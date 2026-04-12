from __future__ import annotations

from typing import Any

try:
    from ..schemas import (
        FinalEvidenceBundle,
        FinalEvidenceItem,
        NormalizedDocument,
        RelevanceEvaluation,
    )
except ImportError:
    from schemas import (
        FinalEvidenceBundle,
        FinalEvidenceItem,
        NormalizedDocument,
        RelevanceEvaluation,
    )


LOCAL_EVIDENCE_MIN_SCORE = 0.60
LOCAL_EVIDENCE_MAX_ITEMS = 3


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def _coerce_title(doc: NormalizedDocument) -> str:
    metadata = doc.metadata or {}
    return str(doc.title or metadata.get("title") or doc.source or "untitled").strip()


def _coerce_url(doc: NormalizedDocument) -> str:
    metadata = doc.metadata or {}
    return str(doc.url or metadata.get("url") or metadata.get("pdf_link") or "").strip()


def annotate_local_documents(
    docs: list[NormalizedDocument | dict[str, Any]],
    evaluation: RelevanceEvaluation,
) -> list[NormalizedDocument]:
    normalized_docs = [
        doc if isinstance(doc, NormalizedDocument) else NormalizedDocument(**doc)
        for doc in docs
    ]

    matched_aspects_by_index: dict[int, list[str]] = {}
    for aspect, chunk_indexes in (evaluation.aspect_best_chunks or {}).items():
        for index in chunk_indexes:
            matched_aspects_by_index.setdefault(index, []).append(aspect)

    annotated_docs: list[NormalizedDocument] = []
    for index, doc in enumerate(normalized_docs):
        metadata = dict(doc.metadata or {})
        matched_aspects = _dedupe_strings(
            list(metadata.get("aspects") or []) + matched_aspects_by_index.get(index, [])
        )
        title = _coerce_title(doc)
        url = _coerce_url(doc)
        origin = str(doc.origin or metadata.get("origin") or "local_kb").strip() or "local_kb"

        metadata["title"] = title
        metadata["url"] = url
        metadata["origin"] = origin
        metadata["aspects"] = matched_aspects

        annotated_docs.append(
            doc.model_copy(
                update={
                    "title": title,
                    "url": url,
                    "origin": origin,
                    "aspects": matched_aspects,
                    "metadata": metadata,
                }
            )
        )
    return annotated_docs


def normalized_doc_to_final_evidence_item(
    doc: NormalizedDocument | dict[str, Any],
    *,
    default_origin: str,
) -> FinalEvidenceItem:
    normalized = doc if isinstance(doc, NormalizedDocument) else NormalizedDocument(**doc)
    metadata = dict(normalized.metadata or {})
    title = _coerce_title(normalized)
    url = _coerce_url(normalized)
    origin = str(normalized.origin or metadata.get("origin") or default_origin).strip() or default_origin
    aspects = _dedupe_strings(list(normalized.aspects or []) + list(metadata.get("aspects") or []))

    metadata["title"] = title
    metadata["url"] = url
    metadata["origin"] = origin
    metadata["aspects"] = aspects
    metadata["index"] = _safe_int(metadata.get("index"))

    return FinalEvidenceItem(
        index=_safe_int(metadata.get("index")),
        origin=origin,
        content=str(normalized.content or "").strip(),
        source=str(normalized.source or title or url or origin).strip() or origin,
        title=title,
        url=url,
        aspects=aspects,
        score=normalized.score,
        metadata=metadata,
    )


def final_evidence_item_to_normalized_doc(item: FinalEvidenceItem | dict[str, Any]) -> NormalizedDocument:
    evidence = item if isinstance(item, FinalEvidenceItem) else FinalEvidenceItem(**item)
    metadata = dict(evidence.metadata or {})
    metadata["index"] = evidence.index
    metadata["title"] = evidence.title
    metadata["url"] = evidence.url
    metadata["origin"] = evidence.origin
    metadata["aspects"] = list(evidence.aspects or [])
    return NormalizedDocument(
        content=evidence.content,
        source=evidence.source,
        score=evidence.score,
        title=evidence.title,
        url=evidence.url,
        origin=evidence.origin,
        aspects=list(evidence.aspects or []),
        metadata=metadata,
    )


def select_local_evidence(
    docs: list[NormalizedDocument | dict[str, Any]],
    *,
    min_score: float = LOCAL_EVIDENCE_MIN_SCORE,
    max_items: int = LOCAL_EVIDENCE_MAX_ITEMS,
) -> list[FinalEvidenceItem]:
    candidates: list[FinalEvidenceItem] = []
    for doc in docs:
        item = normalized_doc_to_final_evidence_item(doc, default_origin="local_kb")
        if _safe_float(item.score) < min_score:
            continue
        if not item.aspects:
            continue
        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            -_safe_float(item.score),
            item.title.lower(),
            item.source.lower(),
        )
    )
    return candidates[:max_items]


def build_final_evidence_bundle(
    *,
    query: str,
    local_evidence: list[FinalEvidenceItem | dict[str, Any]] | None = None,
    web_evidence: list[FinalEvidenceItem | dict[str, Any]] | None = None,
    uncovered_aspects: list[str] | None = None,
    note: str = "",
) -> FinalEvidenceBundle:
    raw_local_items = [
        item if isinstance(item, FinalEvidenceItem) else FinalEvidenceItem(**item)
        for item in (local_evidence or [])
    ]
    raw_web_items = [
        item if isinstance(item, FinalEvidenceItem) else FinalEvidenceItem(**item)
        for item in (web_evidence or [])
    ]
    uncovered = _dedupe_strings(uncovered_aspects)
    local_items: list[FinalEvidenceItem] = []
    web_items: list[FinalEvidenceItem] = []
    all_items: list[FinalEvidenceItem] = []

    next_index = 1
    for item in raw_local_items:
        metadata = dict(item.metadata or {})
        metadata["index"] = next_index
        assigned_item = item.model_copy(update={"index": next_index, "metadata": metadata})
        local_items.append(assigned_item)
        all_items.append(assigned_item)
        next_index += 1
    for item in raw_web_items:
        metadata = dict(item.metadata or {})
        metadata["index"] = next_index
        assigned_item = item.model_copy(update={"index": next_index, "metadata": metadata})
        web_items.append(assigned_item)
        all_items.append(assigned_item)
        next_index += 1

    summary_parts = [
        f"local_evidence={len(local_items)}",
        f"web_evidence={len(web_items)}",
        f"uncovered_aspects={len(uncovered)}",
    ]
    if uncovered:
        summary_parts.append(f"missing={', '.join(uncovered[:4])}")
    if note:
        summary_parts.append(note)

    return FinalEvidenceBundle(
        query=str(query or "").strip(),
        summary="; ".join(summary_parts),
        local_evidence=local_items,
        web_evidence=web_items,
        all_evidence=all_items,
        uncovered_aspects=uncovered,
    )
