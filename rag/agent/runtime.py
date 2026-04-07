from __future__ import annotations

import copy
from typing import Any

try:
    from ..llm_factory import create_default_llm
except ImportError:
    from llm_factory import create_default_llm


_rag_system = None
_pdf_processor = None
_llm = None
_progress_callback = None


class ResearchContext:
    """Shared in-memory context kept for compatibility with the existing API layer."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.original_query = ""
        self.query_plan: dict[str, Any] | None = None
        self.retrieval_result: dict[str, Any] | None = None
        self.web_search_result: dict[str, Any] | None = None
        self.final_evidence: dict[str, Any] | None = None
        self.final_evidence_items: list[dict[str, Any]] = []
        self.current_missing_aspects: list[str] = []

    def set_final_evidence(self, bundle: dict[str, Any] | None) -> None:
        self.final_evidence = copy.deepcopy(bundle) if bundle else None
        evidence_items = []
        if bundle:
            evidence_items = [dict(item) for item in (bundle.get("all_evidence") or [])]
        self.final_evidence_items = evidence_items

    def set_web_search_result(self, payload: dict[str, Any] | None) -> None:
        self.web_search_result = copy.deepcopy(payload) if payload else None

    def set_current_missing_aspects(self, aspects: list[str] | None) -> None:
        self.current_missing_aspects = [str(item).strip() for item in (aspects or []) if str(item).strip()]


context = ResearchContext()


def init_runtime(rag_system, pdf_processor, llm=None):
    global _rag_system, _pdf_processor, _llm
    _rag_system = rag_system
    _pdf_processor = pdf_processor
    _llm = llm or create_default_llm()
    print("Academic RAG runtime initialized")


def get_rag_system():
    return _rag_system


def get_pdf_processor():
    return _pdf_processor


def get_llm():
    return _llm or create_default_llm()


def set_progress_callback(callback):
    global _progress_callback
    _progress_callback = callback


def log_progress(message: str):
    print(message)
    if _progress_callback:
        _progress_callback(message)
