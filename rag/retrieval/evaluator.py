from __future__ import annotations

import re
from statistics import mean
from typing import Any

try:
    from ..schemas import AcademicQueryPlan, NormalizedDocument, RelevanceEvaluation
except ImportError:
    from schemas import AcademicQueryPlan, NormalizedDocument, RelevanceEvaluation


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-\+_/\.]*")


def normalize_english_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").lower()).strip()
    cleaned = re.sub(r"[^a-z0-9\-\+_/\. ]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def english_tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(normalize_english_text(text)))


def _keyword_match(text: str, tokens: set[str], keyword: str) -> float:
    normalized_keyword = normalize_english_text(keyword)
    if not normalized_keyword:
        return 0.0
    if " " in normalized_keyword:
        if normalized_keyword in text:
            return 1.0
        phrase_tokens = english_tokens(normalized_keyword)
        if not phrase_tokens:
            return 0.0
        overlap = len(phrase_tokens & tokens) / len(phrase_tokens)
        return round(overlap * 0.8, 4)
    return 1.0 if normalized_keyword in tokens else 0.0


def evaluate_retrieval(
    query_plan: AcademicQueryPlan | dict[str, Any],
    docs: list[NormalizedDocument | dict[str, Any]],
) -> RelevanceEvaluation:
    plan = query_plan if isinstance(query_plan, AcademicQueryPlan) else AcademicQueryPlan(**query_plan)
    normalized_docs = [
        doc if isinstance(doc, NormalizedDocument) else NormalizedDocument(**doc)
        for doc in docs
    ]

    keywords = [keyword for keyword in plan.keywords_en if keyword]
    if not keywords:
        keywords = list(english_tokens(plan.retrieval_query_en))
    if not keywords:
        keywords = list(english_tokens(plan.crawler_query_en))

    scored_docs: list[NormalizedDocument] = []
    coverage_scores: list[float] = []
    informative_sources = set()

    for doc in normalized_docs:
        metadata_blob = " ".join(
            str(value)
            for key, value in (doc.metadata or {}).items()
            if key in {"title", "abstract", "keywords", "source"}
        )
        haystack = normalize_english_text(f"{doc.content} {doc.source} {metadata_blob}")
        tokens = english_tokens(haystack)

        if keywords:
            matched = sum(_keyword_match(haystack, tokens, keyword) for keyword in keywords)
            coverage = matched / len(keywords)
        else:
            coverage = 0.0

        enriched_doc = doc.model_copy(update={"score": round(max(doc.score or 0.0, coverage), 4)})
        scored_docs.append(enriched_doc)
        coverage_scores.append(coverage)
        if coverage > 0:
            informative_sources.add(enriched_doc.source)

    top_scores = sorted(coverage_scores, reverse=True)
    top1 = round(top_scores[0], 4) if top_scores else 0.0
    top3 = round(mean(top_scores[:3]), 4) if top_scores else 0.0
    unique_sources = len(informative_sources)

    sufficient = top1 >= 0.50 or (top3 >= 0.35 and unique_sources >= 2)
    reason = (
        f"keyword_coverage_top1={top1:.2f}, "
        f"avg_top3={top3:.2f}, "
        f"unique_sources={unique_sources}, "
        f"keywords={len(keywords)}"
    )

    return RelevanceEvaluation(
        sufficient=sufficient,
        top1_coverage=top1,
        avg_top3_coverage=top3,
        unique_sources=unique_sources,
        reason=reason,
        scored_docs=scored_docs,
    )
