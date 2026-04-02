from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

try:
    from ..crawlers.arxiv import ArxivCrawlerIntegrated
    from ..schemas import NormalizedDocument, RetrievalPayload
    from .runtime import context, get_pdf_processor, get_rag_system, log_progress
except ImportError:
    from crawlers.arxiv import ArxivCrawlerIntegrated
    from schemas import NormalizedDocument, RetrievalPayload
    from agent.runtime import context, get_pdf_processor, get_rag_system, log_progress


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    safe = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [str(item) for item in value]
        else:
            safe[key] = str(value)
    return safe


def _normalize_langchain_doc(doc) -> NormalizedDocument:
    metadata = _safe_metadata(getattr(doc, "metadata", {}) or {})
    source = (
        str(metadata.get("source", "")).strip()
        or str(metadata.get("title", "")).strip()
        or "unknown"
    )
    content = str(getattr(doc, "page_content", "")).strip()
    return NormalizedDocument(
        content=content,
        source=source,
        score=None,
        metadata=metadata,
    )


@tool
def retrieve_local_kb(search_query: str) -> str:
    """Retrieve documents from the local academic knowledge base. Always use this tool first."""
    rag_system = get_rag_system()
    log_progress(f"正在检索本地知识库: {search_query}")

    if rag_system is None or getattr(rag_system, "retriever", None) is None:
        payload = RetrievalPayload(
            status="error",
            message="RAG系统未初始化，无法执行本地检索。",
            query=search_query,
            doc_count=0,
            docs=[],
        )
        return payload.model_dump_json()

    try:
        docs = rag_system.retriever.invoke(search_query)
        normalized_docs = [_normalize_langchain_doc(doc) for doc in docs[:6]]
        payload = RetrievalPayload(
            status="success",
            message=f"本地知识库返回 {len(normalized_docs)} 条候选文档。",
            query=search_query,
            doc_count=len(normalized_docs),
            docs=normalized_docs,
        )
        context.retrieval_result = payload.model_dump()
        context.extend_sources([doc.model_dump() for doc in normalized_docs])
        return payload.model_dump_json()
    except Exception as exc:
        payload = RetrievalPayload(
            status="error",
            message=f"本地检索失败: {exc}",
            query=search_query,
            doc_count=0,
            docs=[],
        )
        return payload.model_dump_json()


@tool
def crawl_academic_sources(query_en: str, keywords_en: list[str]) -> str:
    """Crawl academic sources, especially arXiv, using the optimized English query plan."""
    log_progress(f"正在执行学术爬虫: {query_en}")
    crawler = ArxivCrawlerIntegrated("./paper_results")
    payload = crawler.crawl_and_collect(query_en=query_en, keywords_en=keywords_en, max_pages=3)

    pdf_processor = get_pdf_processor()
    rag_system = get_rag_system()

    if payload.papers:
        crawler.save_to_csv([paper.model_dump() for paper in payload.papers], "paper_result.csv")
        crawler.save_formatted_papers(
            [crawler.format_paper(paper.model_dump()) for paper in payload.papers],
            "formatted_papers.txt",
        )

    if payload.papers and pdf_processor and rag_system:
        try:
            downloaded_count = crawler.download_papers(
                papers=[paper.model_dump() for paper in payload.papers],
                max_downloads=3,
            )
            indexed_doc_count = 0
            if downloaded_count > 0:
                processed = pdf_processor.process_pdf_folder("./paper_results")
                if processed:
                    rag_system.update_rag_system()
                    refreshed_docs = rag_system.retriever.invoke(query_en)
                    indexed_doc_count = len(refreshed_docs[:3])
                    refreshed_normalized = [_normalize_langchain_doc(doc) for doc in refreshed_docs[:3]]
                    if refreshed_normalized:
                        payload = payload.model_copy(
                            update={
                                "message": f"{payload.message} 已下载 {downloaded_count} 篇论文并刷新知识库。",
                                "evidence_docs": refreshed_normalized,
                                "downloaded_count": downloaded_count,
                                "indexed_doc_count": indexed_doc_count,
                            }
                        )
            else:
                payload = payload.model_copy(update={"downloaded_count": 0, "indexed_doc_count": 0})
        except Exception as exc:
            payload = payload.model_copy(
                update={
                    "status": "partial_success",
                    "message": f"{payload.message} PDF 下载或入库失败: {exc}",
                }
            )

    context.crawl_result = payload.model_dump()
    context.set_sources([doc.model_dump() for doc in payload.evidence_docs])
    return payload.model_dump_json()
