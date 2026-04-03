from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

try:
    from ..crawlers.arxiv import ArxivCrawlerIntegrated
    from ..schemas import AcademicQueryPlan, NormalizedDocument, RetrievalPayload
    from .runtime import context, get_pdf_processor, get_rag_system, log_progress
except ImportError:
    from crawlers.arxiv import ArxivCrawlerIntegrated
    from schemas import AcademicQueryPlan, NormalizedDocument, RetrievalPayload
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


def _build_runtime_query_plan(search_query: str) -> AcademicQueryPlan:
    query_plan_raw = context.query_plan or {}
    if query_plan_raw:
        try:
            return AcademicQueryPlan(**query_plan_raw)
        except Exception:
            pass

    fallback_query = str(search_query or "").strip()
    return AcademicQueryPlan(
        original_query=fallback_query,
        normalized_query_zh=fallback_query,
        retrieval_query_zh=fallback_query,
        retrieval_query_en=fallback_query,
        crawler_query_en=fallback_query,
        keywords_zh=[],
        keywords_en=[],
    )


def _build_retrieval_message(debug: dict[str, Any], returned_count: int) -> str:
    branch_counts = debug.get("branch_counts", {})
    return (
        "本地混合检索完成: "
        f"BM25={branch_counts.get('bm25_en', 0)}, "
        f"Dense-ZH={branch_counts.get('dense_zh', 0)}, "
        f"Dense-EN={branch_counts.get('dense_en', 0)}, "
        f"RRF top20={debug.get('rrf_pool_count', 0)}, "
        f"返回={returned_count}"
    )


@tool
def retrieve_local_kb(search_query: str) -> str:
    """Retrieve documents from the local academic knowledge base. Always use this tool first."""
    rag_system = get_rag_system()
    log_progress(f"正在检索本地知识库: {search_query}")

    if (
        rag_system is None
        or getattr(rag_system, "vectorstore", None) is None
        or not hasattr(rag_system, "retrieve_with_query_plan")
    ):
        payload = RetrievalPayload(
            status="error",
            message="RAG系统未初始化，无法执行本地检索。",
            query=search_query,
            doc_count=0,
            docs=[],
        )
        return payload.model_dump_json()

    try:
        query_plan = _build_runtime_query_plan(search_query)
        normalized_docs, debug = rag_system.retrieve_with_query_plan(query_plan, final_top_k=5)
        payload = RetrievalPayload(
            status="success",
            message=_build_retrieval_message(debug, len(normalized_docs)),
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
                    refreshed_plan = _build_runtime_query_plan(query_en)
                    refreshed_normalized, debug = rag_system.retrieve_with_query_plan(
                        refreshed_plan,
                        final_top_k=5,
                    )
                    indexed_doc_count = len(refreshed_normalized)
                    if refreshed_normalized:
                        payload = payload.model_copy(
                            update={
                                "message": (
                                    f"{payload.message} 已下载{downloaded_count} 篇论文并刷新知识库。"
                                    f"{_build_retrieval_message(debug, len(refreshed_normalized))}"
                                ),
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
