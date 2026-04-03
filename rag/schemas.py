from __future__ import annotations

from typing import Any, Annotated

from langchain.agents import AgentState
from langchain.agents.middleware.types import OmitFromInput
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired


class AcademicQueryPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    original_query: str
    normalized_query_zh: str
    retrieval_query_zh: str
    retrieval_query_en: str
    crawler_query_en: str
    keywords_zh: list[str] = Field(default_factory=list)
    keywords_en: list[str] = Field(default_factory=list)
    required_aspects: list[str] = Field(default_factory=list)


class NormalizedDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: str = "retrieval_result"
    status: str
    message: str
    query: str
    doc_count: int
    docs: list[NormalizedDocument] = Field(default_factory=list)


class CrawlPaper(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    abstract: str = ""
    submission_date: str = ""
    pdf_link: str = ""
    authors: list[str] = Field(default_factory=list)


class CrawlPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: str = "crawl_result"
    status: str
    message: str
    query: str
    query_en: str
    keywords_en: list[str] = Field(default_factory=list)
    papers: list[CrawlPaper] = Field(default_factory=list)
    evidence_docs: list[NormalizedDocument] = Field(default_factory=list)
    downloaded_count: int = 0
    indexed_doc_count: int = 0


class RelevanceEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sufficient: bool
    partial_answerable: bool

    aspect_coverage: float
    support_strength: float
    noise_ratio: float

    covered_aspects: list[str] = Field(default_factory=list)
    weak_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)

    aspect_scores: dict[str, float] = Field(default_factory=dict)
    aspect_best_chunks: dict[str, list[int]] = Field(default_factory=dict)
    next_action: str

    # Legacy debug-only fields kept for compatibility.
    top1_coverage: float
    avg_top3_coverage: float
    unique_sources: int
    reason: str
    scored_docs: list[NormalizedDocument] = Field(default_factory=list)


class ResearchState(AgentState[dict[str, Any]]):
    query_plan: NotRequired[Annotated[dict[str, Any] | None, OmitFromInput]]
    retrieval_result: NotRequired[Annotated[dict[str, Any] | None, OmitFromInput]]
    retrieval_sufficient: NotRequired[Annotated[bool | None, OmitFromInput]]
    retrieval_next_action: NotRequired[Annotated[str | None, OmitFromInput]]
    retrieval_retry_count: NotRequired[Annotated[int, OmitFromInput]]
    relevance_score: NotRequired[Annotated[float | None, OmitFromInput]]
    relevance_reason: NotRequired[Annotated[str | None, OmitFromInput]]
    relevance_aspect_coverage: NotRequired[Annotated[float | None, OmitFromInput]]
    relevance_support_strength: NotRequired[Annotated[float | None, OmitFromInput]]
    relevance_noise_ratio: NotRequired[Annotated[float | None, OmitFromInput]]
    relevance_missing_aspects: NotRequired[Annotated[list[str], OmitFromInput]]
    relevance_weak_aspects: NotRequired[Annotated[list[str], OmitFromInput]]
    crawl_required: NotRequired[Annotated[bool, OmitFromInput]]
    crawl_used: NotRequired[Annotated[bool, OmitFromInput]]
    final_sources: NotRequired[Annotated[list[dict[str, Any]], OmitFromInput]]
