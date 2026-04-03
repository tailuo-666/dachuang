from __future__ import annotations

import re
from statistics import mean
from typing import Any

try:
    from ..schemas import AcademicQueryPlan, NormalizedDocument, RelevanceEvaluation
except ImportError:
    from schemas import AcademicQueryPlan, NormalizedDocument, RelevanceEvaluation


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-\+_/\.]*")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")

TOP_K = 3

STRONG_THRESHOLD = 0.75
COVERED_BEST_THRESHOLD = 0.60
COVERED_COMBINED_THRESHOLD = 0.75
WEAK_THRESHOLD = 0.35

SUFFICIENT_COVERAGE_THRESHOLD = 0.75
SUFFICIENT_STRENGTH_THRESHOLD = 0.60
SUFFICIENT_NOISE_THRESHOLD = 0.40

RETRIEVE_MORE_NOISE_THRESHOLD = 0.70
NOISE_CHUNK_THRESHOLD = 0.35


def normalize_english_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").lower()).strip()
    cleaned = re.sub(r"[^a-z0-9\-\+_/\. ]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def english_tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(normalize_english_text(text)))


def ordered_english_tokens(text: str) -> list[str]:
    normalized = normalize_english_text(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_PATTERN.findall(normalized):
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def mixed_tokens(text: str) -> set[str]:
    raw = str(text or "")
    english = english_tokens(raw)
    chinese = {token.strip().lower() for token in CJK_PATTERN.findall(raw) if token.strip()}
    return english | chinese


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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _token_overlap_score(aspect: str, doc_tokens: set[str]) -> float:
    aspect_tokens = mixed_tokens(aspect)
    if not aspect_tokens:
        return 0.0
    overlap = len(aspect_tokens & doc_tokens) / len(aspect_tokens)
    return round(min(1.0, max(0.0, overlap)), 4)


def _prepare_chunk(doc: NormalizedDocument) -> dict[str, Any]:
    metadata_blob = " ".join(
        str(value)
        for key, value in (doc.metadata or {}).items()
        if key in {"title", "abstract", "keywords", "source"}
    )
    raw_text = f"{doc.content} {doc.source} {metadata_blob}".strip()
    normalized_text = normalize_english_text(raw_text)
    metadata = doc.metadata or {}
    retrieval_debug = metadata.get("retrieval_debug") if isinstance(metadata.get("retrieval_debug"), dict) else {}
    branch_hits_raw = retrieval_debug.get("branch_hits") if isinstance(retrieval_debug, dict) else []
    branch_ranks_raw = retrieval_debug.get("branch_ranks") if isinstance(retrieval_debug, dict) else {}

    branch_hits = set()
    if isinstance(branch_hits_raw, list):
        branch_hits = {str(branch).strip() for branch in branch_hits_raw if str(branch).strip()}
    branch_ranks: dict[str, float] = {}
    if isinstance(branch_ranks_raw, dict):
        for branch_name, rank in branch_ranks_raw.items():
            score = _safe_float(rank)
            if score > 0:
                branch_ranks[str(branch_name)] = score

    dense_hits = int("dense_zh" in branch_hits) + int("dense_en" in branch_hits)
    dense_signal = dense_hits / 2.0
    bm25_signal = 1.0 if "bm25_en" in branch_hits else 0.0

    rank_signals: list[float] = []
    for branch_name, rank in branch_ranks.items():
        rank_score = max(0.0, 1.0 - (rank - 1.0) / 15.0)
        if branch_name in {"dense_zh", "dense_en"}:
            rank_score = min(1.0, rank_score + 0.1)
        rank_signals.append(rank_score)
    rank_signal = mean(rank_signals) if rank_signals else 0.0

    raw_rrf = _safe_float(metadata.get("rrf_score", retrieval_debug.get("rrf_score", 0.0)))
    rrf_signal = min(1.0, max(0.0, raw_rrf / 0.03)) if raw_rrf > 0 else 0.0

    retrieval_confidence = min(
        1.0,
        max(
            0.0,
            0.45 * dense_signal
            + 0.15 * bm25_signal
            + 0.25 * rank_signal
            + 0.15 * rrf_signal,
        ),
    )

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "english_tokens": english_tokens(normalized_text),
        "mixed_tokens": mixed_tokens(raw_text),
        "retrieval_confidence": round(retrieval_confidence, 4),
    }


def _context_similarity(aspect: str, prepared_chunk: dict[str, Any]) -> float:
    overlap = _token_overlap_score(aspect, prepared_chunk["mixed_tokens"])

    normalized_aspect = normalize_english_text(aspect)
    normalized_text = str(prepared_chunk["normalized_text"])
    english_phrase_match = 1.0 if normalized_aspect and normalized_aspect in normalized_text else 0.0

    chinese_terms = [token for token in CJK_PATTERN.findall(aspect) if token]
    raw_text = str(prepared_chunk["raw_text"])
    chinese_phrase_match = 1.0 if chinese_terms and any(term in raw_text for term in chinese_terms) else 0.0

    return round(min(1.0, max(overlap, english_phrase_match, chinese_phrase_match)), 4)


def _compute_lexical_from_prepared(aspect: str, prepared_chunk: dict[str, Any]) -> float:
    keyword_score = _keyword_match(prepared_chunk["normalized_text"], prepared_chunk["english_tokens"], aspect)
    overlap_score = _token_overlap_score(aspect, prepared_chunk["mixed_tokens"])
    if keyword_score > 0:
        return round(min(1.0, max(keyword_score, overlap_score)), 4)
    return round(min(1.0, max(0.0, overlap_score)), 4)


def _compute_semantic_from_prepared(aspect: str, prepared_chunk: dict[str, Any]) -> float:
    similarity = _context_similarity(aspect, prepared_chunk)
    return round(min(1.0, max(0.0, similarity)), 4)


def _compute_support_from_prepared(
    aspect: str,
    prepared_chunk: dict[str, Any],
    lexical_score: float | None = None,
    context_similarity: float | None = None,
) -> float:
    lexical = lexical_score if lexical_score is not None else _compute_lexical_from_prepared(aspect, prepared_chunk)
    context = (
        context_similarity if context_similarity is not None else _compute_semantic_from_prepared(aspect, prepared_chunk)
    )
    retrieval_confidence = float(prepared_chunk.get("retrieval_confidence", 0.0))
    support = 0.5 * lexical + 0.4 * context + 0.1 * retrieval_confidence
    return round(min(1.0, max(0.0, support)), 4)


def compute_lexical_score(aspect: str, chunk: NormalizedDocument | dict[str, Any]) -> float:
    doc = chunk if isinstance(chunk, NormalizedDocument) else NormalizedDocument(**chunk)
    prepared = _prepare_chunk(doc)
    return _compute_lexical_from_prepared(aspect, prepared)


def compute_semantic_score(aspect: str, chunk: NormalizedDocument | dict[str, Any]) -> float:
    doc = chunk if isinstance(chunk, NormalizedDocument) else NormalizedDocument(**chunk)
    prepared = _prepare_chunk(doc)
    return _compute_semantic_from_prepared(aspect, prepared)


def compute_support(aspect: str, chunk: NormalizedDocument | dict[str, Any]) -> float:
    doc = chunk if isinstance(chunk, NormalizedDocument) else NormalizedDocument(**chunk)
    prepared = _prepare_chunk(doc)
    return _compute_support_from_prepared(aspect, prepared)


def _normalize_aspects(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = normalize_english_text(text) or text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _resolve_required_aspects(plan: AcademicQueryPlan) -> list[str]:
    aspects = _normalize_aspects(plan.required_aspects)[:5]
    if aspects:
        return aspects

    aspects = _normalize_aspects(plan.keywords_en)[:5]
    if aspects:
        return aspects

    retrieval_tokens = ordered_english_tokens(plan.retrieval_query_en)
    if retrieval_tokens:
        return retrieval_tokens[:5]

    crawler_tokens = ordered_english_tokens(plan.crawler_query_en)
    if crawler_tokens:
        return crawler_tokens[:5]

    fallback = str(plan.original_query or "").strip()
    return [fallback] if fallback else []


def _classify_aspect(best_score: float, combined_score: float) -> str:
    if best_score >= STRONG_THRESHOLD:
        return "strong"
    if best_score >= COVERED_BEST_THRESHOLD or combined_score >= COVERED_COMBINED_THRESHOLD:
        return "covered"
    if best_score >= WEAK_THRESHOLD:
        return "weak"
    return "missing"


def evaluate_retrieval(
    query_plan: AcademicQueryPlan | dict[str, Any],
    docs: list[NormalizedDocument | dict[str, Any]],
) -> RelevanceEvaluation:
    plan = query_plan if isinstance(query_plan, AcademicQueryPlan) else AcademicQueryPlan(**query_plan)
    normalized_docs = [
        doc if isinstance(doc, NormalizedDocument) else NormalizedDocument(**doc)
        for doc in docs
    ]

    required_aspects = _resolve_required_aspects(plan)[:5]
    prepared_chunks = [_prepare_chunk(doc) for doc in normalized_docs]
    chunk_best_support = [0.0 for _ in prepared_chunks]

    covered_aspects: list[str] = []
    weak_aspects: list[str] = []
    missing_aspects: list[str] = []
    aspect_scores: dict[str, float] = {}
    aspect_best_chunks: dict[str, list[int]] = {}

    for aspect in required_aspects:
        scores: list[float] = []
        for chunk_idx, prepared_chunk in enumerate(prepared_chunks):
            lexical_score = _compute_lexical_from_prepared(aspect, prepared_chunk)
            context_similarity = _compute_semantic_from_prepared(aspect, prepared_chunk)
            support = _compute_support_from_prepared(
                aspect,
                prepared_chunk,
                lexical_score=lexical_score,
                context_similarity=context_similarity,
            )
            scores.append(support)
            if support > chunk_best_support[chunk_idx]:
                chunk_best_support[chunk_idx] = support

        ranked_pairs = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
        top_pairs = ranked_pairs[:TOP_K]
        top_scores = [score for _, score in top_pairs]

        best_score = round(top_scores[0], 4) if top_scores else 0.0
        combined_score = round(min(1.0, sum(top_scores)), 4)
        status = _classify_aspect(best_score, combined_score)

        if status in {"strong", "covered"}:
            covered_aspects.append(aspect)
        elif status == "weak":
            weak_aspects.append(aspect)
        else:
            missing_aspects.append(aspect)

        aspect_scores[aspect] = best_score
        aspect_best_chunks[aspect] = [idx for idx, score in top_pairs if score > 0]

    total_aspects = len(required_aspects)
    covered_count = len(covered_aspects)
    weak_count = len(weak_aspects)

    aspect_coverage = round((covered_count / total_aspects), 4) if total_aspects else 0.0
    support_strength = round(mean(aspect_scores.values()), 4) if aspect_scores else 0.0

    if chunk_best_support:
        noisy_chunks = sum(1 for score in chunk_best_support if score < NOISE_CHUNK_THRESHOLD)
        noise_ratio = round(noisy_chunks / len(chunk_best_support), 4)
    else:
        noise_ratio = 1.0

    sufficient = (
        aspect_coverage >= SUFFICIENT_COVERAGE_THRESHOLD
        and support_strength >= SUFFICIENT_STRENGTH_THRESHOLD
        and noise_ratio <= SUFFICIENT_NOISE_THRESHOLD
    )
    partial_answerable = (
        total_aspects > 0 and ((covered_count + weak_count) / total_aspects) >= 0.5
    )

    if sufficient:
        next_action = "answer"
    elif not normalized_docs or noise_ratio > RETRIEVE_MORE_NOISE_THRESHOLD:
        next_action = "retrieve_more"
    elif missing_aspects:
        next_action = "crawl_more"
    else:
        next_action = "retrieve_more"

    missing_count = len(missing_aspects)
    missing_preview = ", ".join(missing_aspects[:3]) if missing_aspects else "none"
    reason = (
        f"aspect_coverage={aspect_coverage:.2f}, "
        f"support_strength={support_strength:.2f}, "
        f"noise_ratio={noise_ratio:.2f}, "
        f"missing_aspects_count={missing_count}, "
        f"missing_aspects={missing_preview}, "
        f"next_action={next_action}"
    )

    top_scores = sorted(chunk_best_support, reverse=True)
    top1 = round(top_scores[0], 4) if top_scores else 0.0
    top3 = round(mean(top_scores[:3]), 4) if top_scores else 0.0
    unique_sources = len(
        {
            doc.source
            for idx, doc in enumerate(normalized_docs)
            if idx < len(chunk_best_support) and chunk_best_support[idx] > 0
        }
    )

    scored_docs: list[NormalizedDocument] = []
    for idx, doc in enumerate(normalized_docs):
        support_score = chunk_best_support[idx] if idx < len(chunk_best_support) else 0.0
        enriched_doc = doc.model_copy(
            update={"score": round(max(_safe_float(doc.score), support_score), 4)}
        )
        scored_docs.append(enriched_doc)

    return RelevanceEvaluation(
        sufficient=sufficient,
        partial_answerable=partial_answerable,
        aspect_coverage=aspect_coverage,
        support_strength=support_strength,
        noise_ratio=noise_ratio,
        covered_aspects=covered_aspects,
        weak_aspects=weak_aspects,
        missing_aspects=missing_aspects,
        aspect_scores=aspect_scores,
        aspect_best_chunks=aspect_best_chunks,
        next_action=next_action,
        top1_coverage=top1,
        avg_top3_coverage=top3,
        unique_sources=unique_sources,
        reason=reason,
        scored_docs=scored_docs,
    )
