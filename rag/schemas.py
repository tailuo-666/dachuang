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
    top1_coverage: float
    avg_top3_coverage: float
    unique_sources: int
    reason: str
    scored_docs: list[NormalizedDocument] = Field(default_factory=list)


class ResearchState(AgentState[dict[str, Any]]):
    query_plan: NotRequired[Annotated[dict[str, Any] | None, OmitFromInput]]
    retrieval_result: NotRequired[Annotated[dict[str, Any] | None, OmitFromInput]]
    retrieval_sufficient: NotRequired[Annotated[bool | None, OmitFromInput]]
    relevance_score: NotRequired[Annotated[float | None, OmitFromInput]]
    relevance_reason: NotRequired[Annotated[str | None, OmitFromInput]]
    crawl_required: NotRequired[Annotated[bool, OmitFromInput]]
    crawl_used: NotRequired[Annotated[bool, OmitFromInput]]
    final_sources: NotRequired[Annotated[list[dict[str, Any]], OmitFromInput]]
