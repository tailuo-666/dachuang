from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests

from rag.ssh_service import (
    build_ssh_service_config,
    discover_openai_model,
    ensure_ssh_openai_base_url,
    is_ssh_tunnel_enabled,
)


DEFAULT_MODEL = "paddle-ocr-vl"
DEFAULT_PROMPT = "Please extract all visible text from the image in reading order."


def print_step(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def request_json(method: str, url: str, timeout: float = 30.0, **kwargs: Any) -> dict[str, Any]:
    resp = requests.request(method=method, url=url, timeout=timeout, **kwargs)
    print(f"[HTTP] {method} {url} -> {resp.status_code}")
    preview = (resp.text or "")[:800]
    if preview:
        print(f"[BODY] {preview}")
    try:
        return resp.json()
    except Exception:
        return {"_raw_text": resp.text, "_status_code": resp.status_code}


def normalize_base_url(base_url: str) -> str:
    cleaned = str(base_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def service_root_from_base_url(base_url: str) -> str:
    normalized = normalize_base_url(base_url)
    if normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def build_ocr_ssh_config(args: argparse.Namespace) -> dict[str, Any]:
    config = build_ssh_service_config(
        "ocr",
        default_remote_port=args.remote_port,
        default_local_port=args.local_port,
    )
    overrides = {
        "ssh_host": args.ssh_host,
        "ssh_port": args.ssh_port,
        "ssh_username": args.ssh_username,
        "ssh_password": args.ssh_password,
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "local_host": args.local_host,
        "local_port": args.local_port,
    }
    for key, value in overrides.items():
        if value not in (None, ""):
            config[key] = value
    return config


def resolve_base_url(args: argparse.Namespace) -> str:
    explicit_base_url = normalize_base_url(args.base_url or os.getenv("RAG_OCR_BASE_URL", ""))
    if explicit_base_url:
        return explicit_base_url

    ssh_config = build_ocr_ssh_config(args)
    if is_ssh_tunnel_enabled(ssh_config):
        return ensure_ssh_openai_base_url("ocr", ssh_config)

    raise ValueError(
        "No OCR endpoint configured. Provide --base-url or SSH credentials such as "
        "--ssh-host/--ssh-username/--ssh-password."
    )


def read_local_file_as_data_url(file_path: str) -> str:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_pdf_page_to_png(pdf_path: str, page_number: int) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF (fitz) is required when using --pdf-path.") from exc

    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if page_number < 1:
        raise ValueError("--pdf-page must be >= 1")

    with fitz.open(path) as doc:
        if page_number > doc.page_count:
            raise ValueError(f"PDF page {page_number} exceeds total pages {doc.page_count}.")
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=200)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.close()
        pix.save(tmp.name)
        return tmp.name


def resolve_image_payload(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.image_url:
        return args.image_url, "remote image url", ""

    if args.image_path:
        path = str(Path(args.image_path).expanduser().resolve())
        return read_local_file_as_data_url(path), path, ""

    if args.pdf_path:
        rendered_path = render_pdf_page_to_png(args.pdf_path, args.pdf_page)
        return (
            read_local_file_as_data_url(rendered_path),
            f"{args.pdf_path} (rendered page {args.pdf_page})",
            rendered_path,
        )

    raise ValueError("Provide one of --image-url, --image-path, or --pdf-path for the OCR request.")


def test_health(base_url: str) -> dict[str, Any]:
    print_step("1) Test /health")
    service_root = service_root_from_base_url(base_url)
    data = request_json("GET", f"{service_root}/health", timeout=10.0)
    print("[OK] /health responded")
    return data


def test_models(base_url: str) -> dict[str, Any]:
    print_step("2) Test /v1/models")
    data = request_json("GET", f"{normalize_base_url(base_url)}/models", timeout=20.0)
    print("[OK] /v1/models responded")
    return data


def test_ocr_chat(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    image_url: str,
    max_tokens: int,
) -> dict[str, Any]:
    print_step("3) Test /v1/chat/completions with image")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an OCR assistant. Return extracted text concisely.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    data = request_json(
        "POST",
        f"{normalize_base_url(base_url)}/chat/completions",
        timeout=180.0,
        headers=headers,
        json=payload,
    )
    if "choices" not in data:
        raise RuntimeError("OCR request did not return choices. Check the response preview above.")
    return data


def pick_model(args: argparse.Namespace, base_url: str) -> str:
    explicit_model = str(args.model or os.getenv("RAG_OCR_MODEL", "")).strip()
    if explicit_model:
        return explicit_model

    discovered = discover_openai_model(
        normalize_base_url(base_url),
        api_key=str(args.api_key or os.getenv("RAG_OCR_API_KEY", "EMPTY")).strip() or "EMPTY",
    )
    if discovered:
        return discovered
    return DEFAULT_MODEL


def summarize_choices(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    return content.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test remote OCR service over an SSH tunnel.")
    parser.add_argument("--base-url", default=os.getenv("RAG_OCR_BASE_URL", ""))
    parser.add_argument("--ssh-host", default=os.getenv("RAG_OCR_SSH_HOST", os.getenv("RAG_SSH_HOST", "")))
    parser.add_argument("--ssh-port", type=int, default=int(os.getenv("RAG_OCR_SSH_PORT", os.getenv("RAG_SSH_PORT", "8888"))))
    parser.add_argument("--ssh-username", default=os.getenv("RAG_OCR_SSH_USERNAME", os.getenv("RAG_SSH_USERNAME", "")))
    parser.add_argument("--ssh-password", default=os.getenv("RAG_OCR_SSH_PASSWORD", os.getenv("RAG_SSH_PASSWORD", "")))
    parser.add_argument("--remote-host", default=os.getenv("RAG_OCR_REMOTE_HOST", "127.0.0.1"))
    parser.add_argument("--remote-port", type=int, default=int(os.getenv("RAG_OCR_REMOTE_PORT", "8002")))
    parser.add_argument("--local-host", default=os.getenv("RAG_OCR_LOCAL_HOST", "127.0.0.1"))
    parser.add_argument("--local-port", type=int, default=int(os.getenv("RAG_OCR_LOCAL_PORT", "18002")))
    parser.add_argument("--model", default=os.getenv("RAG_OCR_MODEL", ""))
    parser.add_argument("--api-key", default=os.getenv("RAG_OCR_API_KEY", "EMPTY"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-url", default="")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--pdf-path", default="")
    parser.add_argument("--pdf-page", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--print-json", action="store_true", help="Print the full OCR JSON response.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        base_url = resolve_base_url(args)
        model = pick_model(args, base_url)
        image_payload_url, image_desc, temp_image_path = resolve_image_payload(args)

        print_step("Resolved OCR target")
        print(json.dumps(
            {
                "base_url": normalize_base_url(base_url),
                "model": model,
                "image_source": image_desc,
                "mode": "ssh" if not (args.base_url or "").strip() else "direct",
            },
            ensure_ascii=False,
            indent=2,
        ))

        test_health(base_url)
        models_data = test_models(base_url)

        if isinstance(models_data.get("data"), list):
            model_ids = [item.get("id") for item in models_data["data"] if isinstance(item, dict)]
            print(f"[INFO] available models: {model_ids}")

        ocr_data = test_ocr_chat(
            base_url=base_url,
            model=model,
            api_key=str(args.api_key or "EMPTY").strip() or "EMPTY",
            prompt=args.prompt,
            image_url=image_payload_url,
            max_tokens=args.max_tokens,
        )

        result_text = summarize_choices(ocr_data)
        print_step("4) OCR response preview")
        if result_text:
            print(result_text[:2000])
        else:
            print("[WARN] response has no text content in choices[0].message.content")

        if args.print_json:
            print_step("Full OCR JSON")
            print(json.dumps(ocr_data, ensure_ascii=False, indent=2))

        print("\n[PASS] Remote OCR connectivity test passed")
        return 0
    except Exception as exc:
        print(f"\n[FAIL] {exc!r}")
        return 1
    finally:
        temp_image = locals().get("temp_image_path", "")
        if temp_image:
            try:
                Path(temp_image).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
