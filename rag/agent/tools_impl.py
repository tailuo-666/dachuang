from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import tool

try:
    from ..crawlers.arxiv import ArxivCrawlerIntegrated
    from ..schemas import AcademicQueryPlan, CrawlPayload, NormalizedDocument, RetrievalPayload
    from .runtime import context, get_pdf_processor, get_rag_system, log_progress
except ImportError:
    from crawlers.arxiv import ArxivCrawlerIntegrated
    from schemas import AcademicQueryPlan, CrawlPayload, NormalizedDocument, RetrievalPayload
    from agent.runtime import context, get_pdf_processor, get_rag_system, log_progress


PERSIST_MAX_DOWNLOADS = 3


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
    source = str(metadata.get("source", "")).strip() or str(metadata.get("title", "")).strip() or "unknown"
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


def _dedupe_strings(items: list[str] | None) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(text)
    return deduped


def _resolve_missing_aspects(missing_aspects: list[str] | None) -> list[str]:
    requested = _dedupe_strings(missing_aspects)
    if requested:
        return requested
    return _dedupe_strings(context.current_missing_aspects)


def _empty_crawl_payload(message: str, requested_missing_aspects: list[str]) -> CrawlPayload:
    return CrawlPayload(
        status="empty",
        message=message,
        requested_missing_aspects=requested_missing_aspects,
        covered_missing_aspects=[],
        uncovered_missing_aspects=requested_missing_aspects,
        search_queries=[],
        aspect_evidence=[],
        evidence_docs=[],
        selected_papers=[],
        ingestion_status="skipped",
        pending_ingest_paper_count=0,
    )


def _emit_postprocess_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
    else:
        log_progress(message)


def persist_pending_crawl_results(
    job_snapshot: dict[str, Any] | None,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not job_snapshot:
        return {
            "status": "skipped",
            "message": "No pending crawl ingestion job.",
            "downloaded_count": 0,
            "indexed_doc_count": 0,
        }

    output_dir = str(job_snapshot.get("output_dir") or "./paper_results")
    all_papers = [dict(item) for item in (job_snapshot.get("all_papers") or [])]
    ingest_papers = [dict(item) for item in (job_snapshot.get("ingest_papers") or [])][:PERSIST_MAX_DOWNLOADS]
    manifest_csv = str(job_snapshot.get("manifest_csv") or "paper_result.csv")
    manifest_txt = str(job_snapshot.get("manifest_txt") or "formatted_papers.txt")

    if not all_papers:
        return {
            "status": "skipped",
            "message": "Pending crawl job has no papers to persist.",
            "downloaded_count": 0,
            "indexed_doc_count": 0,
        }

    crawler = ArxivCrawlerIntegrated(output_dir)
    _emit_postprocess_progress(progress_callback, f"开始答后入库，候选论文 {len(ingest_papers)} 篇。")

    crawler.save_to_csv(all_papers, manifest_csv)
    crawler.save_formatted_papers(
        [crawler.format_paper(paper) for paper in all_papers],
        manifest_txt,
    )

    pdf_processor = get_pdf_processor()
    rag_system = get_rag_system()
    if not ingest_papers or pdf_processor is None or rag_system is None:
        reason = "No shortlisted papers" if not ingest_papers else "RAG runtime is not initialized"
        _emit_postprocess_progress(progress_callback, f"答后入库跳过正文处理: {reason}.")
        return {
            "status": "skipped",
            "message": reason,
            "downloaded_count": 0,
            "indexed_doc_count": 0,
        }

    downloaded_count = crawler.download_papers(
        papers=ingest_papers,
        max_downloads=min(len(ingest_papers), PERSIST_MAX_DOWNLOADS),
    )
    indexed_doc_count = 0
    if downloaded_count > 0:
        processed_files = pdf_processor.process_pdf_folder(output_dir)
        indexed_doc_count = len(processed_files or [])
        if processed_files:
            rag_system.update_rag_system()
            _emit_postprocess_progress(
                progress_callback,
                f"答后入库完成，下载 {downloaded_count} 篇，处理 {indexed_doc_count} 个 PDF，并刷新索引。",
            )
        else:
            _emit_postprocess_progress(
                progress_callback,
                f"已下载 {downloaded_count} 篇论文，但没有新的 PDF 被转成 md。",
            )
    else:
        _emit_postprocess_progress(progress_callback, "答后入库未下载到新的 PDF，跳过索引刷新。")

    return {
        "status": "completed" if downloaded_count > 0 else "skipped",
        "message": "Post-answer crawl ingestion finished.",
        "downloaded_count": downloaded_count,
        "indexed_doc_count": indexed_doc_count,
    }


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
            message="RAG 系统未初始化，无法执行本地检索。",
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
def crawl_academic_sources(missing_aspects: list[str]) -> str:
    """Crawl academic sources with missing_aspects and return compact aspect-oriented evidence."""
    requested_missing_aspects = _resolve_missing_aspects(missing_aspects)
    context.set_current_missing_aspects(requested_missing_aspects)

    if not requested_missing_aspects:
        payload = _empty_crawl_payload("missing_aspects 为空，跳过学术爬虫。", [])
        context.set_pending_ingestion_job(None)
        context.crawl_result = payload.model_dump()
        return payload.model_dump_json()

    log_progress(f"正在执行学术爬虫，missing_aspects={requested_missing_aspects}")
    crawler = ArxivCrawlerIntegrated("./paper_results")
    query_plan = _build_runtime_query_plan(" ".join(requested_missing_aspects))

    try:
        payload, ingestion_job = crawler.crawl_and_collect(
            missing_aspects=requested_missing_aspects,
            query_plan=query_plan,
            max_pages=3,
        )
    except Exception as exc:
        payload = _empty_crawl_payload(f"学术爬虫执行失败: {exc}", requested_missing_aspects)
        context.set_pending_ingestion_job(None)
        context.crawl_result = payload.model_dump()
        return payload.model_dump_json()

    context.set_pending_ingestion_job(ingestion_job)
    context.crawl_result = payload.model_dump()
    context.extend_sources([doc.model_dump() for doc in payload.evidence_docs])
    return payload.model_dump_json()
