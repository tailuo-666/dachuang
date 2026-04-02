from __future__ import annotations

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
        self.crawl_result: dict[str, Any] | None = None
        self.papers: list[dict[str, Any]] = []

    def set_sources(self, docs: list[dict[str, Any]]) -> None:
        self.papers = [dict(doc) for doc in docs]

    def extend_sources(self, docs: list[dict[str, Any]]) -> None:
        existing = {str(doc.get("source", "")).strip(): dict(doc) for doc in self.papers}
        for doc in docs:
            key = str(doc.get("source", "")).strip() or str(doc.get("metadata", {}).get("pdf_link", "")).strip()
            if not key:
                key = f"source_{len(existing) + 1}"
            existing[key] = dict(doc)
        self.papers = list(existing.values())


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

