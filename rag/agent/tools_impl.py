from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

try:
    from ..retrieval.evaluator import evaluate_retrieval
    from ..schemas import (
        AcademicQueryPlan,
        FinalEvidenceBundle,
        NormalizedDocument,
        RetrievalPayload,
        WebSearchPayload,
        WebSearchQuery,
        WebSearchResultItem,
    )
    from .evidence import final_evidence_item_to_normalized_doc
    from .runtime import context, get_rag_system, log_progress
except ImportError:
    from retrieval.evaluator import evaluate_retrieval
    from schemas import (
        AcademicQueryPlan,
        FinalEvidenceBundle,
        NormalizedDocument,
        RetrievalPayload,
        WebSearchPayload,
        WebSearchQuery,
        WebSearchResultItem,
    )
    from agent.evidence import final_evidence_item_to_normalized_doc
    from agent.runtime import context, get_rag_system, log_progress


MAX_TAVILY_RESULTS_PER_ASPECT = 3


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
    title = str(metadata.get("title") or metadata.get("source") or "").strip()
    url = str(metadata.get("url") or metadata.get("pdf_link") or "").strip()
    source = str(metadata.get("source", "")).strip() or title or url or "unknown"
    content = str(getattr(doc, "page_content", "")).strip()
    metadata["title"] = title or source
    metadata["url"] = url
    metadata["origin"] = "local_kb"
    return NormalizedDocument(
        content=content,
        source=source,
        score=None,
        title=title or source,
        url=url,
        origin="local_kb",
        aspects=[],
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
        required_aspects=[],
    )


def _build_retrieval_message(debug: dict[str, Any], returned_count: int) -> str:
    branch_counts = debug.get("branch_counts", {})
    return (
        "本地混合检索完成 "
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


def _empty_web_search_payload(message: str, requested_missing_aspects: list[str]) -> WebSearchPayload:
    return WebSearchPayload(
        status="empty",
        message=message,
        requested_missing_aspects=requested_missing_aspects,
        covered_missing_aspects=[],
        uncovered_missing_aspects=requested_missing_aspects,
        search_queries=[],
        results=[],
        evidence_docs=[],
    )


def _get_tavily_api_key() -> str:
    return (
        str(os.getenv("TAVILY_API_KEY") or "").strip()
        or str(os.getenv("RAG_TAVILY_API_KEY") or "").strip()
    )


def _create_tavily_client(api_key: str):
    from tavily import TavilyClient

    try:
        return TavilyClient(api_key=api_key)
    except TypeError:
        return TavilyClient(api_key)


def _build_web_result_item(aspect: str, raw_item: dict[str, Any]) -> WebSearchResultItem | None:
    title = str(raw_item.get("title") or "").strip()
    content = str(raw_item.get("content") or raw_item.get("raw_content") or "").strip()
    url = str(raw_item.get("url") or "").strip()
    if not content:
        return None
    try:
        score = float(raw_item.get("score")) if raw_item.get("score") is not None else None
    except (TypeError, ValueError):
        score = None
    return WebSearchResultItem(
        aspect=str(aspect or "").strip(),
        title=title,
        content=content,
        url=url,
        score=score,
        favicon=str(raw_item.get("favicon") or "").strip(),
    )


def _merge_web_results_to_docs(results: list[WebSearchResultItem]) -> list[NormalizedDocument]:
    merged: dict[str, dict[str, Any]] = {}
    for item in results:
        key = item.url or item.title or f"{item.aspect}:{item.content[:80]}"
        entry = merged.get(key)
        if entry is None:
            entry = {
                "title": item.title.strip() or item.url.strip() or "tavily",
                "url": item.url.strip(),
                "content": item.content.strip(),
                "source": item.title.strip() or item.url.strip() or "tavily",
                "score": item.score,
                "aspects": [],
                "favicon": item.favicon.strip(),
            }
            merged[key] = entry
        if item.content and len(item.content) > len(entry["content"]):
            entry["content"] = item.content.strip()
        entry["score"] = max(float(entry["score"] or 0.0), float(item.score or 0.0)) or None
        entry["aspects"] = _dedupe_strings(list(entry["aspects"]) + [item.aspect])

    documents: list[NormalizedDocument] = []
    for entry in merged.values():
        metadata = {
            "title": entry["title"],
            "url": entry["url"],
            "origin": "tavily_web",
            "provider": "tavily",
            "aspects": list(entry["aspects"]),
            "favicon": entry["favicon"],
            "tavily_score": entry["score"],
        }
        documents.append(
            NormalizedDocument(
                content=entry["content"],
                source=entry["source"],
                score=entry["score"],
                title=entry["title"],
                url=entry["url"],
                origin="tavily_web",
                aspects=list(entry["aspects"]),
                metadata=metadata,
            )
        )
    documents.sort(key=lambda doc: float(doc.score or 0.0), reverse=True)
    return documents


def _build_missing_aspect_plan(
    query_plan: AcademicQueryPlan,
    requested_missing_aspects: list[str],
) -> AcademicQueryPlan:
    joined = " ".join(requested_missing_aspects)
    return query_plan.model_copy(
        update={
            "retrieval_query_en": joined or query_plan.retrieval_query_en,
            "crawler_query_en": joined or query_plan.crawler_query_en,
            "required_aspects": list(requested_missing_aspects),
        }
    )


def _load_current_local_evidence_docs() -> list[NormalizedDocument]:
    bundle_raw = context.final_evidence or {}
    if not bundle_raw:
        return []

    try:
        bundle = FinalEvidenceBundle(**bundle_raw)
    except Exception:
        return []

    return [final_evidence_item_to_normalized_doc(item) for item in bundle.local_evidence]


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
def search_web_with_tavily(missing_aspects: list[str]) -> str:
    """Search Tavily with the current missing_aspects and return normalized web evidence."""
    requested_missing_aspects = _resolve_missing_aspects(missing_aspects)
    context.set_current_missing_aspects(requested_missing_aspects)

    if not requested_missing_aspects:
        payload = _empty_web_search_payload("missing_aspects 为空，跳过 Tavily 搜索。", [])
        context.set_web_search_result(payload.model_dump())
        return payload.model_dump_json()

    api_key = _get_tavily_api_key()
    if not api_key:
        payload = _empty_web_search_payload(
            "未配置 TAVILY_API_KEY，无法执行 Tavily 搜索。",
            requested_missing_aspects,
        )
        context.set_web_search_result(payload.model_dump())
        return payload.model_dump_json()

    try:
        client = _create_tavily_client(api_key)
    except Exception as exc:
        payload = _empty_web_search_payload(
            f"Tavily 客户端初始化失败: {exc}",
            requested_missing_aspects,
        )
        context.set_web_search_result(payload.model_dump())
        return payload.model_dump_json()

    query_plan = _build_runtime_query_plan(" ".join(requested_missing_aspects))
    search_queries: list[WebSearchQuery] = []
    result_items: list[WebSearchResultItem] = []

    log_progress(f"正在执行 Tavily 搜索，missing_aspects={requested_missing_aspects}")
    for aspect in requested_missing_aspects:
        search_queries.append(WebSearchQuery(aspect=aspect, query=aspect))
        try:
            response = client.search(query=aspect, search_depth="advanced")
        except Exception as exc:
            log_progress(f"Tavily 搜索失败，aspect={aspect}: {exc}")
            continue

        raw_results = response.get("results") if isinstance(response, dict) else []
        kept = 0
        for raw_item in raw_results or []:
            item = _build_web_result_item(aspect, raw_item)
            if item is None:
                continue
            result_items.append(item)
            kept += 1
            if kept >= MAX_TAVILY_RESULTS_PER_ASPECT:
                break

    evidence_docs = _merge_web_results_to_docs(result_items)
    combined_docs = [*_load_current_local_evidence_docs(), *evidence_docs]
    coverage_plan = _build_missing_aspect_plan(query_plan, requested_missing_aspects)

    covered_missing_aspects: list[str] = []
    uncovered_missing_aspects = list(requested_missing_aspects)
    if combined_docs:
        evaluation = evaluate_retrieval(coverage_plan, combined_docs)
        covered_missing_aspects = list(evaluation.covered_aspects)
        uncovered_missing_aspects = [
            aspect for aspect in requested_missing_aspects if aspect not in covered_missing_aspects
        ]

    if evidence_docs:
        status = "success" if not uncovered_missing_aspects else "partial_success"
    else:
        status = "empty"

    message = (
        f"Tavily returned {len(result_items)} results, "
        f"normalized into {len(evidence_docs)} evidence docs, "
        f"covered {len(covered_missing_aspects)}/{len(requested_missing_aspects)} missing_aspects."
    )
    if uncovered_missing_aspects:
        message = f"{message} Uncovered: {', '.join(uncovered_missing_aspects[:4])}."

    payload = WebSearchPayload(
        status=status,
        message=message,
        requested_missing_aspects=requested_missing_aspects,
        covered_missing_aspects=covered_missing_aspects,
        uncovered_missing_aspects=uncovered_missing_aspects,
        search_queries=search_queries,
        results=result_items,
        evidence_docs=evidence_docs,
    )
    context.set_web_search_result(payload.model_dump())
    return payload.model_dump_json()
