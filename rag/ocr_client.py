from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

try:
    from .ssh_service import (
        build_ssh_service_config,
        discover_openai_model,
        ensure_ssh_openai_base_url,
        is_ssh_tunnel_enabled,
    )
except ImportError:
    from ssh_service import (
        build_ssh_service_config,
        discover_openai_model,
        ensure_ssh_openai_base_url,
        is_ssh_tunnel_enabled,
    )


DEFAULT_OCR_MODEL = "paddle-ocr-vl"
DEFAULT_OCR_PROMPT = (
    "Extract all visible text from this page in reading order. "
    "Return plain text only. Do not explain the page, do not summarize it, "
    "and do not include OCR position markers."
)


def normalize_ocr_base_url(base_url: str) -> str:
    cleaned = str(base_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def resolve_ocr_ssh_config() -> dict[str, Any]:
    return build_ssh_service_config(
        "ocr",
        default_remote_port=int(os.getenv("RAG_OCR_REMOTE_PORT", "8002")),
        default_local_port=int(os.getenv("RAG_OCR_LOCAL_PORT", "18002")),
    )


def resolve_ocr_base_url() -> str:
    explicit_base_url = normalize_ocr_base_url(str(os.getenv("RAG_OCR_BASE_URL") or ""))
    if explicit_base_url:
        return explicit_base_url

    ssh_config = resolve_ocr_ssh_config()
    if is_ssh_tunnel_enabled(ssh_config):
        try:
            return ensure_ssh_openai_base_url("ocr", ssh_config)
        except Exception as exc:
            print(f"Failed to establish OCR SSH tunnel: {exc}")

    ocr_host = str(os.getenv("RAG_OCR_HOST") or "").strip()
    ocr_port = str(os.getenv("RAG_OCR_PORT") or "").strip()
    ocr_scheme = str(os.getenv("RAG_OCR_SCHEME", "http")).strip() or "http"
    if ocr_host and ocr_port:
        return f"{ocr_scheme}://{ocr_host}:{ocr_port}/v1"

    return ""


def resolve_ocr_model(base_url: str = "") -> str:
    configured = str(os.getenv("RAG_OCR_MODEL") or "").strip()
    if configured:
        return configured

    resolved_base_url = normalize_ocr_base_url(base_url) or resolve_ocr_base_url()
    if resolved_base_url:
        discovered = discover_openai_model(
            resolved_base_url,
            api_key=str(os.getenv("RAG_OCR_API_KEY", "EMPTY")).strip() or "EMPTY",
        )
        if discovered:
            return discovered

    return DEFAULT_OCR_MODEL


def file_to_data_url(path: str, mime_type: str | None = None) -> str:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"OCR input file not found: {resolved_path}")

    guessed_mime_type, _ = mimetypes.guess_type(str(resolved_path))
    effective_mime_type = mime_type or guessed_mime_type or "application/octet-stream"
    encoded = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
    return f"data:{effective_mime_type};base64,{encoded}"


def bytes_to_data_url(payload: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class RemoteOCRClient:
    """Minimal OpenAI-compatible multimodal OCR client."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        max_tokens: int = 2048,
        retry_max_tokens: int = 4096,
        prompt: str = DEFAULT_OCR_PROMPT,
    ) -> None:
        self.base_url = normalize_ocr_base_url(base_url)
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "EMPTY").strip() or "EMPTY"
        self.timeout = float(timeout)
        self.max_tokens = max(256, int(max_tokens))
        self.retry_max_tokens = max(self.max_tokens, int(retry_max_tokens))
        self.prompt = str(prompt or DEFAULT_OCR_PROMPT).strip() or DEFAULT_OCR_PROMPT

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _request_chat_completion(self, *, image_url: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        if not self.base_url:
            raise ValueError("OCR base_url is empty.")
        if not self.model:
            raise ValueError("OCR model is empty.")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            timeout=self.timeout,
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an OCR assistant. Return plain text only and preserve reading order."
                        ),
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
                "max_tokens": int(max_tokens),
            },
        )
        response.raise_for_status()
        return response.json()

    def _extract_message_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_value = str(item.get("text") or "").strip()
                    if text_value:
                        parts.append(text_value)
            return "\n".join(parts).strip()
        return str(content or "").strip()

    def _finish_reason(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        return str(first_choice.get("finish_reason") or "").strip()

    def extract_from_data_url(
        self,
        image_url: str,
        *,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        effective_prompt = str(prompt or self.prompt).strip() or self.prompt
        effective_max_tokens = max(256, int(max_tokens or self.max_tokens))

        payload = self._request_chat_completion(
            image_url=image_url,
            prompt=effective_prompt,
            max_tokens=effective_max_tokens,
        )
        finish_reason = self._finish_reason(payload)
        if finish_reason == "length" and effective_max_tokens < self.retry_max_tokens:
            payload = self._request_chat_completion(
                image_url=image_url,
                prompt=effective_prompt,
                max_tokens=self.retry_max_tokens,
            )

        return {
            "text": self._extract_message_text(payload),
            "raw_response": payload,
            "finish_reason": self._finish_reason(payload),
        }

    def extract_from_image_path(
        self,
        image_path: str,
        *,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return self.extract_from_data_url(
            file_to_data_url(image_path, mime_type="image/png"),
            prompt=prompt,
            max_tokens=max_tokens,
        )

    def extract_from_image_bytes(
        self,
        payload: bytes,
        *,
        mime_type: str = "image/png",
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return self.extract_from_data_url(
            bytes_to_data_url(payload, mime_type=mime_type),
            prompt=prompt,
            max_tokens=max_tokens,
        )

    def healthcheck(self) -> dict[str, Any]:
        if not self.base_url:
            raise ValueError("OCR base_url is empty.")
        service_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        response = requests.get(f"{service_root}/health", timeout=min(self.timeout, 30.0))
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"status_code": response.status_code, "text": response.text}

    def list_models(self) -> dict[str, Any]:
        if not self.base_url:
            raise ValueError("OCR base_url is empty.")
        response = requests.get(
            f"{self.base_url}/models",
            timeout=min(self.timeout, 30.0),
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()


def create_default_ocr_client(**overrides: Any) -> RemoteOCRClient:
    base_url = normalize_ocr_base_url(str(overrides.get("base_url") or resolve_ocr_base_url()))
    model = str(overrides.get("model") or resolve_ocr_model(base_url)).strip()
    api_key = str(overrides.get("api_key") or os.getenv("RAG_OCR_API_KEY", "EMPTY")).strip() or "EMPTY"
    timeout = float(overrides.get("timeout", 180.0))
    max_tokens = int(overrides.get("max_tokens", 2048))
    retry_max_tokens = int(overrides.get("retry_max_tokens", 4096))
    prompt = str(overrides.get("prompt") or DEFAULT_OCR_PROMPT).strip() or DEFAULT_OCR_PROMPT

    if not base_url:
        raise ValueError("OCR base_url is empty. Configure RAG_OCR_BASE_URL or RAG_OCR SSH settings first.")

    return RemoteOCRClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        retry_max_tokens=retry_max_tokens,
        prompt=prompt,
    )
