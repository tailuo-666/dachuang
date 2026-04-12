from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import requests


class DummyPDFProcessor:
    """Main flow smoke test does not need OCR/PDF processing."""

    def __init__(self, output_dir: str = "./md", lang: str = "en", dpi: int = 220) -> None:
        self.output_dir = output_dir
        self.lang = lang
        self.dpi = dpi


def print_step(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def set_env_if_value(key: str, value: str | None) -> None:
    if value is not None and str(value).strip():
        os.environ[key] = str(value).strip()


def set_or_clear_env(key: str, value: str | None) -> None:
    cleaned = str(value).strip() if value is not None else ""
    if cleaned:
        os.environ[key] = cleaned
    else:
        os.environ.pop(key, None)


def configure_stdout_encoding() -> None:
    stdout = getattr(sys, "stdout", None)
    if stdout and hasattr(stdout, "reconfigure"):
        try:
            stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", "<built-in>")
        return True, str(module_path)
    except Exception as exc:
        return False, repr(exc)


def check_python_imports() -> dict[str, dict[str, Any]]:
    modules = [
        "requests",
        "tavily",
    ]
    results: dict[str, dict[str, Any]] = {}
    for module_name in modules:
        ok, detail = check_import(module_name)
        results[module_name] = {"ok": ok, "detail": detail}
    results["langchain_openai"] = {
        "ok": True,
        "detail": "deferred to LLM runtime check",
    }
    results["langchain_community"] = {
        "ok": True,
        "detail": "deferred to embedding/full-flow runtime check",
    }
    if str(os.getenv("RAG_EMBEDDING_BASE_URL") or "").strip():
        results["langchain_huggingface"] = {
            "ok": True,
            "detail": "skipped because embedding service mode is enabled",
        }
    else:
        results["langchain_huggingface"] = {
            "ok": True,
            "detail": "deferred to local embedding fallback runtime check",
        }
    return results


def check_openai_compatible_service(base_url: str, *, api_key: str = "EMPTY") -> dict[str, Any]:
    service_url = str(base_url or "").rstrip("/")
    result: dict[str, Any] = {
        "base_url": service_url,
        "http_ok": False,
        "http_status": None,
        "models": [],
    }
    try:
        resp = requests.get(
            f"{service_url}/models",
            timeout=20.0,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        result["http_status"] = resp.status_code
        if resp.ok:
            payload = resp.json()
            result["http_ok"] = True
            result["models"] = [
                item.get("id")
                for item in (payload.get("data") or [])
                if isinstance(item, dict) and item.get("id")
            ]
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def check_llm_runtime() -> dict[str, Any]:
    from rag.llm_factory import create_default_llm, get_default_llm_config

    cfg = get_default_llm_config()
    result: dict[str, Any] = {
        "config": cfg,
        "service_check": check_openai_compatible_service(
            str(cfg.get("base_url") or ""),
            api_key=str(cfg.get("api_key") or "EMPTY"),
        ),
        "invoke_ok": False,
        "invoke_preview": "",
    }
    try:
        llm = create_default_llm()
        response = llm.invoke("请只回复 OK")
        content = response.content if hasattr(response, "content") else str(response)
        result["invoke_ok"] = True
        result["invoke_preview"] = str(content)[:200]
    except Exception as exc:
        result["invoke_error"] = repr(exc)
    return result


def check_embedding_runtime() -> dict[str, Any]:
    from langchain_community.embeddings import FakeEmbeddings

    from rag.rag_system import RAGSystem, VLLMOpenAIEmbeddings

    rag_system = RAGSystem()
    resolved_path = rag_system._resolve_embedding_model_path(None)
    resolved_base_url = rag_system._resolve_embedding_base_url()
    resolved_service_model = rag_system._resolve_embedding_model()
    result: dict[str, Any] = {
        "embedding_base_url": resolved_base_url,
        "embedding_service_model": resolved_service_model,
        "resolved_model_path": resolved_path,
        "path_exists": bool(resolved_path and os.path.exists(resolved_path)),
        "embedding_device": os.getenv("RAG_EMBEDDING_DEVICE", "cuda:0"),
        "embedding_class": "",
        "uses_embedding_service": False,
        "is_fake_embeddings": True,
        "embedding_ok": False,
    }

    if resolved_base_url:
        result["service_check"] = check_openai_compatible_service(
            resolved_base_url,
            api_key=str(os.getenv("RAG_EMBEDDING_API_KEY", "EMPTY")),
        )

    try:
        rag_system.setup_embeddings()
        embedding_obj = rag_system.embeddings
        result["embedding_class"] = type(embedding_obj).__name__
        result["uses_embedding_service"] = isinstance(embedding_obj, VLLMOpenAIEmbeddings)
        result["is_fake_embeddings"] = isinstance(embedding_obj, FakeEmbeddings)
        result["embedding_ok"] = embedding_obj is not None and not result["is_fake_embeddings"]
    except Exception as exc:
        result["embedding_error"] = repr(exc)

    return result


def check_local_rag_artifacts() -> dict[str, Any]:
    cwd = Path.cwd()
    faiss_dir = cwd / "faiss"
    md_dir = cwd / "md"
    md_files = sorted(md_dir.glob("*.md")) if md_dir.exists() else []
    return {
        "cwd": str(cwd),
        "faiss_dir": str(faiss_dir),
        "faiss_exists": faiss_dir.exists(),
        "md_dir": str(md_dir),
        "md_exists": md_dir.exists(),
        "md_file_count": len(md_files),
        "md_examples": [item.name for item in md_files[:5]],
    }


def build_answer_preview(answer: str, limit: int = 400) -> str:
    cleaned = str(answer or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "...(truncated)"


def collect_runtime_env_snapshot() -> dict[str, Any]:
    return {
        "llm_base_url": os.getenv("RAG_LLM_BASE_URL", ""),
        "embedding_base_url": os.getenv("RAG_EMBEDDING_BASE_URL", ""),
        "llm_model": os.getenv("RAG_LLM_MODEL", ""),
        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", ""),
        "ssh_host": os.getenv("RAG_SSH_HOST", ""),
        "ssh_port": os.getenv("RAG_SSH_PORT", ""),
        "ssh_username": os.getenv("RAG_SSH_USERNAME", ""),
        "llm_remote_port": os.getenv("RAG_LLM_REMOTE_PORT", ""),
        "embedding_remote_port": os.getenv("RAG_EMBEDDING_REMOTE_PORT", ""),
        "llm_local_port": os.getenv("RAG_LLM_LOCAL_PORT", ""),
        "embedding_local_port": os.getenv("RAG_EMBEDDING_LOCAL_PORT", ""),
        "tavily_key_present": bool(str(os.getenv("TAVILY_API_KEY") or "").strip()),
    }


def run_full_flow(query: str) -> dict[str, Any]:
    from rag.agent.builder import RagService
    from rag.agent.runtime import context, init_runtime
    from rag.llm_factory import create_default_llm
    from rag.rag_system import setup_rag_system

    rag_system = setup_rag_system()
    if rag_system is None:
        artifact_check = check_local_rag_artifacts()
        return {
            "ok": False,
            "error": "RAGSystem initialization failed. Check ./faiss, ./md, and embedding config.",
            "artifact_check": artifact_check,
        }

    init_runtime(
        rag_system=rag_system,
        pdf_processor=DummyPDFProcessor(),
        llm=create_default_llm(),
    )
    context.reset()

    service = RagService()
    result = service.run(query=query)
    final_evidence = context.final_evidence or {}
    final_items = [
        dict(item)
        for item in (context.final_evidence_items or [])
        if isinstance(item, dict)
    ]
    structured_answer = service.parse_final_response(
        result,
        final_evidence_items=final_items,
    )
    answer = structured_answer.answer
    missing_aspects = list(context.current_missing_aspects or [])
    web_search_result = context.web_search_result or {}
    evidence_lookup = {
        int(item.get("index") or 0): item
        for item in final_items
        if int(item.get("index") or 0) > 0
    }
    referenced_evidence = [
        {
            "index": index,
            "origin": evidence_lookup[index].get("origin", ""),
            "title": evidence_lookup[index].get("title", ""),
            "url": evidence_lookup[index].get("url", ""),
            "aspects": evidence_lookup[index].get("aspects", []) or [],
        }
        for index in structured_answer.evidence_list
        if index in evidence_lookup
    ]

    return {
        "ok": bool(answer),
        "answer": answer,
        "answer_preview": build_answer_preview(answer),
        "evidence_list": structured_answer.evidence_list,
        "retrieval_next_action": result.get("retrieval_next_action"),
        "relevance_reason": result.get("relevance_reason"),
        "relevance_missing_aspects": result.get("relevance_missing_aspects"),
        "missing_aspects_for_crawler": missing_aspects,
        "web_search_status": web_search_result.get("status"),
        "web_search_message": web_search_result.get("message", ""),
        "final_evidence_summary": final_evidence.get("summary", ""),
        "final_evidence_item_count": len(final_items),
        "referenced_evidence": referenced_evidence,
    }


def apply_runtime_args(args: argparse.Namespace) -> None:
    set_or_clear_env("RAG_LLM_BASE_URL", args.llm_base_url)
    set_env_if_value("RAG_LLM_MODEL", args.llm_model)
    set_env_if_value("RAG_LLM_API_KEY", args.llm_api_key)

    set_or_clear_env("RAG_EMBEDDING_BASE_URL", args.embedding_base_url)
    set_env_if_value("RAG_EMBEDDING_MODEL", args.embedding_model)
    set_env_if_value("RAG_EMBEDDING_API_KEY", args.embedding_api_key)
    set_env_if_value("RAG_EMBEDDING_MODEL_PATH", args.embedding_model_path)
    set_env_if_value("RAG_EMBEDDING_DEVICE", args.embedding_device)
    set_env_if_value("TAVILY_API_KEY", args.tavily_api_key)

    if args.use_ssh:
        set_env_if_value("RAG_SSH_HOST", args.ssh_host)
        set_env_if_value("RAG_SSH_PORT", str(args.ssh_port))
        set_env_if_value("RAG_SSH_USERNAME", args.ssh_username)
        set_env_if_value("RAG_SSH_PASSWORD", args.ssh_password)
        set_env_if_value("RAG_LLM_REMOTE_PORT", str(args.llm_remote_port))
        set_env_if_value("RAG_LLM_LOCAL_PORT", str(args.llm_local_port))
        set_env_if_value("RAG_EMBEDDING_REMOTE_PORT", str(args.embedding_remote_port))
        set_env_if_value("RAG_EMBEDDING_LOCAL_PORT", str(args.embedding_local_port))
        # Force the runtime to resolve through SSH instead of stale localhost defaults.
        set_or_clear_env("RAG_LLM_BASE_URL", "")
        set_or_clear_env("RAG_EMBEDDING_BASE_URL", "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the full RAG flow on the server.")
    parser.add_argument("--query", required=True, help="The query to send through the full RAG flow.")
    parser.add_argument("--use-ssh", action="store_true", help="Use SSH tunnel mode for LLM and embedding services.")
    parser.add_argument("--ssh-host", default=os.getenv("RAG_SSH_HOST", ""))
    parser.add_argument("--ssh-port", type=int, default=int(os.getenv("RAG_SSH_PORT", "8888")))
    parser.add_argument("--ssh-username", default=os.getenv("RAG_SSH_USERNAME", ""))
    parser.add_argument("--ssh-password", default=os.getenv("RAG_SSH_PASSWORD", ""))
    parser.add_argument("--llm-remote-port", type=int, default=int(os.getenv("RAG_LLM_REMOTE_PORT", "8001")))
    parser.add_argument("--llm-local-port", type=int, default=int(os.getenv("RAG_LLM_LOCAL_PORT", "18001")))
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("RAG_LLM_BASE_URL", ""),
        help="Direct LLM base_url. Leave empty when using --use-ssh.",
    )
    parser.add_argument("--llm-model", default=os.getenv("RAG_LLM_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--llm-api-key", default=os.getenv("RAG_LLM_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-remote-port", type=int, default=int(os.getenv("RAG_EMBEDDING_REMOTE_PORT", "8000")))
    parser.add_argument("--embedding-local-port", type=int, default=int(os.getenv("RAG_EMBEDDING_LOCAL_PORT", "18000")))
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("RAG_EMBEDDING_BASE_URL", ""),
        help="Direct embedding base_url. Leave empty when using --use-ssh.",
    )
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", ""))
    parser.add_argument("--embedding-api-key", default=os.getenv("RAG_EMBEDDING_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-model-path", default=os.getenv("RAG_EMBEDDING_MODEL_PATH", "/data/202225220617/bge-m3"))
    parser.add_argument("--embedding-device", default=os.getenv("RAG_EMBEDDING_DEVICE", "cuda:0"))
    parser.add_argument("--tavily-api-key", default=os.getenv("TAVILY_API_KEY", ""))
    parser.add_argument("--output-json", default="", help="Optional file path to save the final JSON report.")
    return parser


def main() -> int:
    configure_stdout_encoding()
    parser = build_parser()
    args = parser.parse_args()
    apply_runtime_args(args)

    overall_ok = True
    report: dict[str, Any] = {
        "query": args.query,
        "mode": "ssh" if args.use_ssh else "direct",
        "runtime_env": collect_runtime_env_snapshot(),
    }

    print_step("1) Python import check")
    import_results = check_python_imports()
    print(json.dumps(import_results, ensure_ascii=False, indent=2))
    report["import_check"] = import_results

    print_step("2) LLM runtime check")
    llm_result = check_llm_runtime()
    print(json.dumps(llm_result, ensure_ascii=False, indent=2))
    report["llm_runtime"] = llm_result
    if not llm_result.get("invoke_ok"):
        overall_ok = False

    print_step("3) Embedding runtime check")
    embedding_result = check_embedding_runtime()
    print(json.dumps(embedding_result, ensure_ascii=False, indent=2))
    report["embedding_runtime"] = embedding_result
    if not embedding_result.get("embedding_ok"):
        overall_ok = False

    print_step("4) Local RAG artifact check")
    artifact_result = check_local_rag_artifacts()
    print(json.dumps(artifact_result, ensure_ascii=False, indent=2))
    report["artifact_check"] = artifact_result

    print_step("5) Full RAG flow")
    try:
        rag_result = run_full_flow(args.query)
        print(json.dumps(rag_result, ensure_ascii=False, indent=2))
        report["full_flow"] = rag_result
        if not rag_result.get("ok"):
            overall_ok = False
    except Exception as exc:
        overall_ok = False
        report["full_flow"] = {"flow_error": repr(exc)}
        print(json.dumps(report["full_flow"], ensure_ascii=False, indent=2))
        traceback.print_exc()

    print_step("6) Summary")
    summary = {
        "overall_ok": overall_ok,
        "mode": "ssh" if args.use_ssh else "direct",
        "llm_base_url": os.getenv("RAG_LLM_BASE_URL", ""),
        "embedding_base_url": os.getenv("RAG_EMBEDDING_BASE_URL", ""),
        "ssh_host": os.getenv("RAG_SSH_HOST", ""),
        "llm_remote_port": os.getenv("RAG_LLM_REMOTE_PORT", ""),
        "embedding_remote_port": os.getenv("RAG_EMBEDDING_REMOTE_PORT", ""),
        "tavily_key_present": bool(str(os.getenv("TAVILY_API_KEY") or "").strip()),
        "note_1": "This test does not check OCR.",
        "note_2": "This test does not execute the standalone crawler.",
        "note_3": "missing_aspects_for_crawler shows the aspects that should be passed to the crawler side.",
        "note_4": "If RAGSystem initialization fails, first check faiss_exists and md_file_count.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    report["summary"] = summary

    output_json = str(args.output_json or "").strip()
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_step("7) Report Saved")
        print(json.dumps({"output_json": str(output_path.resolve())}, ensure_ascii=False, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
