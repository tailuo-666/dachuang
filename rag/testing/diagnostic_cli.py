from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from ..crawlers.standalone import apply_runtime_args
    from ..llm_factory import create_default_llm, get_default_llm_config
    from ..rag_system import RAGSystem
    from ..retrieval.evaluator import evaluate_retrieval
    from ..schemas import AcademicQueryPlan, NormalizedDocument
    from ..ssh_service import build_ssh_service_config, ensure_ssh_openai_base_url
    from ..query.optimizer import AcademicQueryPlanner
except ImportError:
    from crawlers.standalone import apply_runtime_args
    from llm_factory import create_default_llm, get_default_llm_config
    from rag_system import RAGSystem
    from retrieval.evaluator import evaluate_retrieval
    from schemas import AcademicQueryPlan, NormalizedDocument
    from ssh_service import build_ssh_service_config, ensure_ssh_openai_base_url
    from query.optimizer import AcademicQueryPlanner


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_results"
DEFAULT_OUTPUT_PREFIX_STEM = "fading_memory_ha_gnn_diag"
DEFAULT_QUERY = (
    "fading memory\uff08\u8870\u51cf\u8bb0\u5fc6\uff09\u2019\u8fd9\u4e00\u6027\u8d28\u5728"
    "\u52a8\u6001\u7cfb\u7edf\u4e2d\u7684\u4f5c\u7528\uff0c\u4e0eHA-GNN\u6a21\u578b"
    "\u4e2d\u5229\u7528\u5386\u53f2\u8bbf\u95ee\u4fe1\u606f\u8fdb\u884c\u9884\u6d4b"
    "\u7684\u673a\u5236\u4e4b\u95f4\uff0c\u6709\u4f55\u76f8\u4f3c\u6027\u4e0e\u672c"
    "\u8d28\u533a\u522b\uff1f"
)
MAX_MD_DOCS_PER_SECTION = 5
MAX_MD_CONTENT_CHARS = 600
MAX_MD_METADATA_CHARS = 500


@dataclass
class DiagnosticPaths:
    output_dir: Path
    output_prefix: str
    step1_md: Path
    step2_md: Path
    step3_md: Path
    trace_json: Path


def configure_stdout_encoding() -> None:
    stdout = getattr(sys, "stdout", None)
    if stdout and hasattr(stdout, "reconfigure"):
        try:
            stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _set_env_if_value(name: str, value: str | None) -> None:
    cleaned = str(value or "").strip()
    if cleaned:
        os.environ[name] = cleaned


def _set_or_clear_env(name: str, value: str | None) -> None:
    cleaned = str(value or "").strip()
    if cleaned:
        os.environ[name] = cleaned
    else:
        os.environ.pop(name, None)


def apply_extended_runtime_args(args: argparse.Namespace) -> None:
    apply_runtime_args(args)
    _set_env_if_value("RAG_LLM_MODEL", getattr(args, "llm_model", ""))
    _set_env_if_value("RAG_LLM_API_KEY", getattr(args, "llm_api_key", ""))
    _set_env_if_value("RAG_EMBEDDING_MODEL", getattr(args, "embedding_model", ""))
    _set_env_if_value("RAG_EMBEDDING_API_KEY", getattr(args, "embedding_api_key", ""))
    _set_env_if_value("RAG_OCR_MODEL", getattr(args, "ocr_model", ""))
    _set_env_if_value("RAG_OCR_API_KEY", getattr(args, "ocr_api_key", ""))

    _set_or_clear_env("RAG_LLM_BASE_URL", getattr(args, "llm_base_url", ""))
    _set_or_clear_env("RAG_EMBEDDING_BASE_URL", getattr(args, "embedding_base_url", ""))
    _set_or_clear_env("RAG_OCR_BASE_URL", getattr(args, "ocr_base_url", ""))

    if getattr(args, "use_ssh", False):
        _set_or_clear_env("RAG_LLM_BASE_URL", "")
        _set_or_clear_env("RAG_EMBEDDING_BASE_URL", "")
        _set_or_clear_env("RAG_OCR_BASE_URL", "")


def make_output_paths(output_dir: Path, output_prefix: str) -> DiagnosticPaths:
    return DiagnosticPaths(
        output_dir=output_dir,
        output_prefix=output_prefix,
        step1_md=output_dir / f"{output_prefix}_step1_optimizer.md",
        step2_md=output_dir / f"{output_prefix}_step2_retrieval.md",
        step3_md=output_dir / f"{output_prefix}_step3_evaluation.md",
        trace_json=output_dir / f"{output_prefix}_trace.json",
    )


def build_default_output_prefix() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{DEFAULT_OUTPUT_PREFIX_STEM}_{timestamp}"


def load_query_from_args(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.query_file:
        path = Path(args.query_file).expanduser().resolve()
        query = path.read_text(encoding="utf-8-sig").strip()
        return query, {"source": "query_file", "path": str(path)}
    if args.query:
        return str(args.query).strip(), {"source": "query_arg", "path": ""}
    raise ValueError("Either --query or --query-file must be provided.")


def detect_text_issues(text: str) -> dict[str, Any]:
    raw_text = str(text or "")
    issue_flags: list[str] = []
    mojibake_markers = ["锛", "鈥", "涓", "鍦", "鐨", "鏈", "鎬", "璁", "闂", "鍙", "鈧"]
    mojibake_marker_count = sum(raw_text.count(marker) for marker in mojibake_markers)

    if "\ufffd" in raw_text:
        issue_flags.append("contains_replacement_char")
    if re.search(r"\?{3,}", raw_text):
        issue_flags.append("contains_repeated_question_marks")
    if "\\u" in raw_text:
        issue_flags.append("contains_literal_unicode_escape")
    if mojibake_marker_count >= 3:
        issue_flags.append("contains_common_mojibake_markers")

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", raw_text))
    latin_count = len(re.findall(r"[A-Za-z]", raw_text))

    return {
        "has_issues": bool(issue_flags),
        "issue_flags": issue_flags,
        "mojibake_marker_count": mojibake_marker_count,
        "cjk_count": cjk_count,
        "latin_count": latin_count,
        "length": len(raw_text),
    }


def coerce_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(value)


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def shorten(text: str, limit: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "...(truncated)"


def serialize_document(doc: NormalizedDocument) -> dict[str, Any]:
    return doc.model_dump()


def markdown_doc_summary(doc: NormalizedDocument, index: int) -> str:
    metadata_preview = shorten(safe_json(doc.metadata or {}), MAX_MD_METADATA_CHARS)
    score = "null" if doc.score is None else f"{float(doc.score):.4f}"
    return "\n".join(
        [
            f"### Chunk {index}",
            f"- source: `{doc.source}`",
            f"- title: `{doc.title}`",
            f"- score: `{score}`",
            f"- origin: `{doc.origin}`",
            f"- aspects: {safe_json(doc.aspects or [])}",
            f"- metadata:",
            "```json",
            metadata_preview,
            "```",
            "- content preview:",
            "```text",
            shorten(doc.content, MAX_MD_CONTENT_CHARS),
            "```",
        ]
    )


def normalize_doc_key(doc: NormalizedDocument) -> str:
    normalized_source = re.sub(r"\s+", " ", str(doc.source or "").strip().lower())
    normalized_content = re.sub(r"\s+", " ", str(doc.content or "").strip().lower())
    return f"{normalized_source}::{normalized_content}"


def annotate_document_branch(doc: NormalizedDocument, branch_name: str) -> NormalizedDocument:
    metadata = dict(doc.metadata or {})
    metadata["diagnostic_branch"] = branch_name
    return doc.model_copy(update={"metadata": metadata})


def combine_branch_documents(
    branch_results: dict[str, dict[str, Any]],
    branch_order: list[str],
    *,
    max_docs: int = 5,
) -> list[NormalizedDocument]:
    combined: list[NormalizedDocument] = []
    seen: set[str] = set()

    for branch_name in branch_order:
        payload = branch_results.get(branch_name) or {}
        if not payload.get("ok"):
            continue
        for doc in payload.get("documents") or []:
            key = normalize_doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            combined.append(annotate_document_branch(doc, branch_name))
            if len(combined) >= max_docs:
                return combined
    return combined


def collect_runtime_snapshot() -> dict[str, Any]:
    llm_config = {}
    try:
        llm_config = get_default_llm_config()
    except Exception as exc:
        llm_config = {"error_type": type(exc).__name__, "error_message": str(exc)}

    rag = RAGSystem()
    embedding_ssh = {}
    embedding_base_url = ""
    embedding_model = ""
    try:
        embedding_ssh = rag._resolve_embedding_ssh_config()
        embedding_base_url = rag._resolve_embedding_base_url()
        embedding_model = rag._resolve_embedding_model()
    except Exception as exc:
        embedding_ssh = {"error_type": type(exc).__name__, "error_message": str(exc)}

    ocr_ssh = build_ssh_service_config(
        "ocr",
        default_remote_port=int(os.getenv("RAG_OCR_REMOTE_PORT", "8002")),
        default_local_port=int(os.getenv("RAG_OCR_LOCAL_PORT", "18002")),
    )

    resolved_ocr_base_url = ""
    try:
        if str(os.getenv("RAG_OCR_BASE_URL") or "").strip():
            resolved_ocr_base_url = str(os.getenv("RAG_OCR_BASE_URL")).strip()
        elif str(os.getenv("RAG_SSH_HOST") or "").strip():
            resolved_ocr_base_url = ensure_ssh_openai_base_url("ocr", ocr_ssh)
    except Exception as exc:
        resolved_ocr_base_url = f"error: {type(exc).__name__}: {exc}"

    return {
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "timestamp": datetime.now().isoformat(),
        "llm": llm_config,
        "embedding": {
            "ssh": embedding_ssh,
            "base_url": embedding_base_url,
            "model": embedding_model,
        },
        "ocr": {
            "ssh": ocr_ssh,
            "base_url": resolved_ocr_base_url,
            "model": str(os.getenv("RAG_OCR_MODEL") or "").strip(),
        },
    }


def run_optimizer_diagnostic(query: str, llm) -> dict[str, Any]:
    planner = AcademicQueryPlanner(llm)
    prompt_value = planner.prompt.invoke({"question": query})
    human_prompt = ""
    if len(prompt_value.messages) > 1:
        human_prompt = coerce_message_content(prompt_value.messages[1].content)

    result: dict[str, Any] = {
        "input_query": query,
        "input_text_diagnostics": detect_text_issues(query),
        "planner_input_question": query,
        "prompt_human_message_excerpt": shorten(human_prompt, 1600),
        "raw_response": "",
        "raw_response_text_diagnostics": {},
        "parsed_json": None,
        "used_fallback": False,
        "error_type": "",
        "error_message": "",
    }

    try:
        response = llm.invoke(prompt_value)
        raw_response = coerce_message_content(getattr(response, "content", response))
        parsed_json = planner._extract_json(raw_response)
        plan = planner._coerce_plan(query, parsed_json)
        result.update(
            {
                "raw_response": raw_response,
                "raw_response_text_diagnostics": detect_text_issues(raw_response),
                "parsed_json": parsed_json,
                "final_plan": plan.model_dump(),
            }
        )
        return result
    except Exception as exc:
        fallback_plan = planner._fallback_plan(query)
        result.update(
            {
                "used_fallback": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "final_plan": fallback_plan.model_dump(),
            }
        )
        return result


def run_single_branch(
    rag_system: RAGSystem,
    branch_name: str,
    query: str,
    runner,
) -> dict[str, Any]:
    try:
        docs = runner()
        normalized_docs = [
            doc if isinstance(doc, NormalizedDocument) else NormalizedDocument(**doc)
            for doc in docs
        ]
        return {
            "ok": True,
            "query": query,
            "doc_count": len(normalized_docs),
            "documents": normalized_docs,
        }
    except Exception as exc:
        return {
            "ok": False,
            "query": query,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "documents": [],
        }


def run_retrieval_diagnostic(plan: AcademicQueryPlan) -> dict[str, Any]:
    rag_system = RAGSystem()
    result: dict[str, Any] = {
        "initialized": False,
        "initialize_error_type": "",
        "initialize_error_message": "",
        "retrieval_queries": {
            "retrieval_query_zh": plan.retrieval_query_zh,
            "retrieval_query_en": plan.retrieval_query_en,
            "keywords_en": list(plan.keywords_en),
            "bm25_query": "",
        },
        "main_retrieval": {
            "ok": False,
            "error_type": "",
            "error_message": "",
            "doc_count": 0,
            "documents": [],
            "debug": {},
        },
        "branches": {},
        "fallback_evaluation_input": {
            "input_source": "none",
            "branch_order": ["bm25_en", "dense_zh", "dense_en"],
            "doc_count": 0,
            "documents": [],
        },
    }

    try:
        result["initialized"] = bool(rag_system.initialize())
    except Exception as exc:
        result["initialize_error_type"] = type(exc).__name__
        result["initialize_error_message"] = str(exc)
        return result

    if not result["initialized"]:
        result["initialize_error_message"] = "RAGSystem.initialize() returned False."
        return result

    bm25_query = rag_system._build_bm25_query(plan)
    result["retrieval_queries"]["bm25_query"] = bm25_query

    try:
        docs, debug = rag_system.retrieve_with_query_plan(plan, final_top_k=5)
        result["main_retrieval"] = {
            "ok": True,
            "error_type": "",
            "error_message": "",
            "doc_count": len(docs),
            "documents": docs,
            "debug": debug,
        }
        return result
    except Exception as exc:
        result["main_retrieval"]["error_type"] = type(exc).__name__
        result["main_retrieval"]["error_message"] = str(exc)

    branch_results = {
        "bm25_en": run_single_branch(
            rag_system,
            "bm25_en",
            bm25_query,
            lambda: rag_system._run_bm25_branch(plan)[1],
        ),
        "dense_zh": run_single_branch(
            rag_system,
            "dense_zh",
            plan.retrieval_query_zh,
            lambda: rag_system._run_dense_branch(plan.retrieval_query_zh),
        ),
        "dense_en": run_single_branch(
            rag_system,
            "dense_en",
            plan.retrieval_query_en,
            lambda: rag_system._run_dense_branch(plan.retrieval_query_en),
        ),
    }
    result["branches"] = branch_results

    fallback_docs = combine_branch_documents(
        branch_results,
        result["fallback_evaluation_input"]["branch_order"],
        max_docs=5,
    )
    result["fallback_evaluation_input"] = {
        "input_source": "branch_fallback" if fallback_docs else "none",
        "branch_order": ["bm25_en", "dense_zh", "dense_en"],
        "doc_count": len(fallback_docs),
        "documents": fallback_docs,
    }
    return result


def run_evaluation_diagnostic(
    plan: AcademicQueryPlan,
    retrieval_trace: dict[str, Any],
) -> dict[str, Any]:
    if retrieval_trace["main_retrieval"]["ok"]:
        evaluation_input_source = "main_retrieval"
        docs = retrieval_trace["main_retrieval"]["documents"]
    else:
        evaluation_input_source = retrieval_trace["fallback_evaluation_input"]["input_source"]
        docs = retrieval_trace["fallback_evaluation_input"]["documents"]

    evaluation = evaluate_retrieval(plan, docs)
    return {
        "input_source": evaluation_input_source,
        "input_doc_count": len(docs),
        "result": evaluation.model_dump(),
    }


def render_optimizer_markdown(
    optimizer_trace: dict[str, Any],
    query_source: dict[str, Any],
    paths: DiagnosticPaths,
) -> str:
    return "\n".join(
        [
            f"# Step 1 Optimizer Diagnostic: {paths.output_prefix}",
            "",
            "## Input",
            f"- query source: `{query_source['source']}`",
            f"- query path: `{query_source.get('path') or ''}`",
            f"- output trace: `{paths.trace_json}`",
            "",
            "```text",
            optimizer_trace["input_query"],
            "```",
            "",
            "## Text Diagnostics",
            "```json",
            safe_json(optimizer_trace["input_text_diagnostics"]),
            "```",
            "",
            "## Planner Input",
            "```text",
            optimizer_trace["planner_input_question"],
            "```",
            "",
            "## Prompt Excerpt",
            "```text",
            optimizer_trace["prompt_human_message_excerpt"],
            "```",
            "",
            "## Raw LLM Response",
            "```text",
            optimizer_trace["raw_response"] or "(empty)",
            "```",
            "",
            "## Raw Response Diagnostics",
            "```json",
            safe_json(optimizer_trace.get("raw_response_text_diagnostics") or {}),
            "```",
            "",
            "## Parse Result",
            f"- used fallback: `{optimizer_trace['used_fallback']}`",
            f"- error type: `{optimizer_trace['error_type']}`",
            f"- error message: `{optimizer_trace['error_message']}`",
            "```json",
            safe_json(optimizer_trace.get("parsed_json")),
            "```",
            "",
            "## Final Query Plan",
            "```json",
            safe_json(optimizer_trace["final_plan"]),
            "```",
        ]
    )


def render_retrieval_markdown(
    plan: AcademicQueryPlan,
    retrieval_trace: dict[str, Any],
    paths: DiagnosticPaths,
) -> str:
    lines = [
        f"# Step 2 Retrieval Diagnostic: {paths.output_prefix}",
        "",
        "## Retrieval Inputs",
        "```json",
        safe_json(retrieval_trace["retrieval_queries"]),
        "```",
        "",
        "## Initialization",
        f"- initialized: `{retrieval_trace['initialized']}`",
        f"- initialize error type: `{retrieval_trace['initialize_error_type']}`",
        f"- initialize error message: `{retrieval_trace['initialize_error_message']}`",
        "",
        "## Main Retrieval",
        f"- ok: `{retrieval_trace['main_retrieval']['ok']}`",
        f"- error type: `{retrieval_trace['main_retrieval']['error_type']}`",
        f"- error message: `{retrieval_trace['main_retrieval']['error_message']}`",
        f"- doc count: `{retrieval_trace['main_retrieval']['doc_count']}`",
        "```json",
        safe_json(retrieval_trace["main_retrieval"].get("debug") or {}),
        "```",
    ]

    main_docs = retrieval_trace["main_retrieval"].get("documents") or []
    if main_docs:
        lines.extend(["", "## Main Retrieval Chunks"])
        for idx, doc in enumerate(main_docs[:MAX_MD_DOCS_PER_SECTION], start=1):
            lines.extend(["", markdown_doc_summary(doc, idx)])

    if retrieval_trace["branches"]:
        lines.extend(["", "## Branch Fallback"])
        for branch_name in ["bm25_en", "dense_zh", "dense_en"]:
            branch = retrieval_trace["branches"].get(branch_name) or {}
            lines.extend(
                [
                    "",
                    f"### {branch_name}",
                    f"- ok: `{branch.get('ok', False)}`",
                    f"- query: `{branch.get('query', '')}`",
                    f"- error type: `{branch.get('error_type', '')}`",
                    f"- error message: `{branch.get('error_message', '')}`",
                    f"- doc count: `{branch.get('doc_count', 0)}`",
                ]
            )
            for idx, doc in enumerate((branch.get("documents") or [])[:MAX_MD_DOCS_PER_SECTION], start=1):
                lines.extend(["", markdown_doc_summary(doc, idx)])

        fallback_input = retrieval_trace["fallback_evaluation_input"]
        lines.extend(
            [
                "",
                "## Fallback Evaluation Input",
                f"- input source: `{fallback_input['input_source']}`",
                f"- branch order: `{', '.join(fallback_input['branch_order'])}`",
                f"- doc count: `{fallback_input['doc_count']}`",
            ]
        )
        for idx, doc in enumerate((fallback_input.get("documents") or [])[:MAX_MD_DOCS_PER_SECTION], start=1):
            lines.extend(["", markdown_doc_summary(doc, idx)])

    lines.extend(
        [
            "",
            "## Plan Reference",
            "```json",
            safe_json(plan.model_dump()),
            "```",
        ]
    )
    return "\n".join(lines)


def render_evaluation_markdown(
    evaluation_trace: dict[str, Any],
    paths: DiagnosticPaths,
) -> str:
    result = evaluation_trace["result"]
    scored_docs = [
        NormalizedDocument(**doc) if not isinstance(doc, NormalizedDocument) else doc
        for doc in result.get("scored_docs") or []
    ]

    lines = [
        f"# Step 3 Evaluation Diagnostic: {paths.output_prefix}",
        "",
        "## Evaluation Input",
        f"- input source: `{evaluation_trace['input_source']}`",
        f"- input doc count: `{evaluation_trace['input_doc_count']}`",
        "",
        "## Evaluation Result",
        "```json",
        safe_json(
            {
                key: value
                for key, value in result.items()
                if key != "scored_docs"
            }
        ),
        "```",
        "",
        "## Aspect to Best Chunks",
        "```json",
        safe_json(result.get("aspect_best_chunks") or {}),
        "```",
    ]

    if scored_docs:
        lines.extend(["", "## Scored Docs"])
        for idx, doc in enumerate(scored_docs[:MAX_MD_DOCS_PER_SECTION], start=1):
            lines.extend(["", markdown_doc_summary(doc, idx)])

    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a three-stage RAG diagnostic flow: optimizer -> local retrieval -> evaluator."
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="UTF-8 query string. Prefer --query-file for mixed Chinese/English input.")
    query_group.add_argument("--query-file", help="UTF-8 text file that contains the diagnostic query.")

    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default="")

    parser.add_argument("--use-ssh", action="store_true", help="Use SSH tunnel mode for LLM, OCR, and embedding.")
    parser.add_argument("--ssh-host", default=os.getenv("RAG_SSH_HOST", ""))
    parser.add_argument("--ssh-port", type=int, default=int(os.getenv("RAG_SSH_PORT", "8888")))
    parser.add_argument("--ssh-username", default=os.getenv("RAG_SSH_USERNAME", ""))
    parser.add_argument("--ssh-password", default=os.getenv("RAG_SSH_PASSWORD", ""))

    parser.add_argument("--llm-remote-port", type=int, default=int(os.getenv("RAG_LLM_REMOTE_PORT", "8001")))
    parser.add_argument("--llm-local-port", type=int, default=int(os.getenv("RAG_LLM_LOCAL_PORT", "18001")))
    parser.add_argument("--embedding-remote-port", type=int, default=int(os.getenv("RAG_EMBEDDING_REMOTE_PORT", "8000")))
    parser.add_argument("--embedding-local-port", type=int, default=int(os.getenv("RAG_EMBEDDING_LOCAL_PORT", "18000")))
    parser.add_argument("--ocr-remote-port", type=int, default=int(os.getenv("RAG_OCR_REMOTE_PORT", "8002")))
    parser.add_argument("--ocr-local-port", type=int, default=int(os.getenv("RAG_OCR_LOCAL_PORT", "18002")))

    parser.add_argument("--llm-base-url", default=os.getenv("RAG_LLM_BASE_URL", ""))
    parser.add_argument("--embedding-base-url", default=os.getenv("RAG_EMBEDDING_BASE_URL", ""))
    parser.add_argument("--ocr-base-url", default=os.getenv("RAG_OCR_BASE_URL", ""))

    parser.add_argument("--llm-model", default=os.getenv("RAG_LLM_MODEL", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("RAG_LLM_API_KEY", ""))
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", ""))
    parser.add_argument("--embedding-api-key", default=os.getenv("RAG_EMBEDDING_API_KEY", ""))
    parser.add_argument("--ocr-model", default=os.getenv("RAG_OCR_MODEL", ""))
    parser.add_argument("--ocr-api-key", default=os.getenv("RAG_OCR_API_KEY", ""))
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)

    query, query_source = load_query_from_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_prefix = str(args.output_prefix or "").strip() or build_default_output_prefix()
    paths = make_output_paths(output_dir, output_prefix)

    apply_extended_runtime_args(args)

    llm = create_default_llm()
    optimizer_trace = run_optimizer_diagnostic(query, llm)
    plan = AcademicQueryPlan(**optimizer_trace["final_plan"])
    retrieval_trace = run_retrieval_diagnostic(plan)
    evaluation_trace = run_evaluation_diagnostic(plan, retrieval_trace)

    trace_payload = {
        "query_source": query_source,
        "query": query,
        "default_query_reference": DEFAULT_QUERY,
        "runtime": collect_runtime_snapshot(),
        "optimizer": {
            **optimizer_trace,
        },
        "retrieval": {
            **retrieval_trace,
            "main_retrieval": {
                **retrieval_trace["main_retrieval"],
                "documents": [
                    serialize_document(doc)
                    for doc in retrieval_trace["main_retrieval"].get("documents") or []
                ],
            },
            "branches": {
                branch_name: {
                    **payload,
                    "documents": [serialize_document(doc) for doc in payload.get("documents") or []],
                }
                for branch_name, payload in (retrieval_trace.get("branches") or {}).items()
            },
            "fallback_evaluation_input": {
                **retrieval_trace["fallback_evaluation_input"],
                "documents": [
                    serialize_document(doc)
                    for doc in retrieval_trace["fallback_evaluation_input"].get("documents") or []
                ],
            },
        },
        "evaluation": evaluation_trace,
        "artifacts": {
            "output_dir": str(paths.output_dir),
            "output_prefix": paths.output_prefix,
            "step1_optimizer_md": str(paths.step1_md),
            "step2_retrieval_md": str(paths.step2_md),
            "step3_evaluation_md": str(paths.step3_md),
            "trace_json": str(paths.trace_json),
        },
    }

    write_text(paths.step1_md, render_optimizer_markdown(optimizer_trace, query_source, paths))
    write_text(paths.step2_md, render_retrieval_markdown(plan, retrieval_trace, paths))
    write_text(paths.step3_md, render_evaluation_markdown(evaluation_trace, paths))
    write_text(paths.trace_json, safe_json(trace_payload))

    print(safe_json(trace_payload["artifacts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
