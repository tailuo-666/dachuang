from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

try:
    from ..llm_factory import create_default_llm
    from ..schemas import AcademicQueryPlan, AspectRewritePayload, QueuedMissingAspect
    from .arxiv import ArxivCrawlerIntegrated
except ImportError:
    from llm_factory import create_default_llm
    from schemas import AcademicQueryPlan, AspectRewritePayload, QueuedMissingAspect
    from crawlers.arxiv import ArxivCrawlerIntegrated


DEFAULT_QUEUE_PATH = "./paper_results/pending_aspects.json"


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


class MissingAspectQueryOptimizer:
    """Rewrite missing aspects into arXiv-friendly abstract search queries."""

    def __init__(self, llm=None) -> None:
        self.llm = llm or create_default_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You rewrite missing-aspect phrases into short English abstract-search queries for academic papers. "
                        "Return JSON only."
                    ),
                ),
                (
                    "human",
                    """
Return strict JSON:
{
  "original_aspect": "...",
  "optimized_query_en": "...",
  "keywords_en": ["..."]
}

Rules:
1. The input aspect may look like a definition request such as "definition of LLM".
2. Rewrite it into a phrase that is more likely to appear in an academic title or abstract.
3. Keep the meaning aligned with the original aspect.
4. `optimized_query_en` should be short and search-like.
5. `keywords_en` should be deduplicated academic phrases, at most 6 items.
6. Return JSON only.

Aspect: {aspect}
                    """.strip(),
                ),
            ]
        )

    def rewrite(self, aspect: str) -> AspectRewritePayload:
        cleaned_aspect = str(aspect or "").strip()
        if not cleaned_aspect:
            return AspectRewritePayload(
                original_aspect="",
                optimized_query_en="",
                keywords_en=[],
            )

        try:
            prompt_value = self.prompt.invoke({"aspect": cleaned_aspect})
            response = self.llm.invoke(prompt_value)
            raw_text = response.content if hasattr(response, "content") else str(response)
            data = self._extract_json(raw_text)
            optimized_query_en = str(data.get("optimized_query_en") or cleaned_aspect).strip()
            keywords_en = self._normalize_string_list(data.get("keywords_en"))
            if not keywords_en:
                keywords_en = self._extract_keywords(optimized_query_en)
            return AspectRewritePayload(
                original_aspect=cleaned_aspect,
                optimized_query_en=optimized_query_en,
                keywords_en=keywords_en[:6],
            )
        except Exception:
            return AspectRewritePayload(
                original_aspect=cleaned_aspect,
                optimized_query_en=cleaned_aspect,
                keywords_en=self._extract_keywords(cleaned_aspect)[:6],
            )

    def _extract_json(self, text: str) -> dict[str, Any]:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned)
            cleaned = cleaned.rstrip("`").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _normalize_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return _dedupe_strings([str(item or "").strip() for item in value])

    def _extract_keywords(self, text: str) -> list[str]:
        matches = re.findall(r"[A-Za-z][A-Za-z0-9\-\+\.]*(?:\s+[A-Za-z][A-Za-z0-9\-\+\.]*)*", text or "")
        return _dedupe_strings(matches)


class MissingAspectQueueStore:
    """Persist standalone crawler pending aspects on disk."""

    def __init__(self, queue_path: str = DEFAULT_QUEUE_PATH) -> None:
        self.queue_path = queue_path
        parent_dir = os.path.dirname(queue_path) or "."
        os.makedirs(parent_dir, exist_ok=True)

    def list_items(self) -> list[QueuedMissingAspect]:
        if not os.path.exists(self.queue_path):
            return []
        try:
            with open(self.queue_path, "r", encoding="utf-8") as file:
                raw_items = json.load(file)
        except Exception:
            return []

        queued_items: list[QueuedMissingAspect] = []
        for raw_item in raw_items if isinstance(raw_items, list) else []:
            try:
                queued_items.append(QueuedMissingAspect(**raw_item))
            except Exception:
                continue
        return queued_items

    def save_items(self, items: list[QueuedMissingAspect]) -> None:
        with open(self.queue_path, "w", encoding="utf-8") as file:
            json.dump([item.model_dump() for item in items], file, ensure_ascii=False, indent=2)

    def enqueue(self, aspects: list[str]) -> list[QueuedMissingAspect]:
        existing = {item.aspect.lower(): item for item in self.list_items()}
        for aspect in _dedupe_strings(aspects):
            key = aspect.lower()
            if key not in existing:
                existing[key] = QueuedMissingAspect(aspect=aspect)
        queued_items = list(existing.values())
        self.save_items(queued_items)
        return queued_items

    def remove(self, aspects: list[str]) -> list[QueuedMissingAspect]:
        remove_keys = {str(aspect or "").strip().lower() for aspect in aspects if str(aspect or "").strip()}
        remaining = [item for item in self.list_items() if item.aspect.lower() not in remove_keys]
        self.save_items(remaining)
        return remaining


class StandaloneMissingAspectCrawler:
    """Independent arXiv crawler driven by queued missing aspects."""

    def __init__(
        self,
        output_dir: str = "./paper_results",
        queue_path: str = DEFAULT_QUEUE_PATH,
        llm=None,
    ) -> None:
        self.output_dir = output_dir
        self.queue_store = MissingAspectQueueStore(queue_path)
        self.optimizer = MissingAspectQueryOptimizer(llm=llm)
        self.crawler = ArxivCrawlerIntegrated(output_dir=output_dir)

    def enqueue_aspects(self, aspects: list[str]) -> list[QueuedMissingAspect]:
        return self.queue_store.enqueue(aspects)

    def list_pending_aspects(self) -> list[QueuedMissingAspect]:
        return self.queue_store.list_items()

    def run_aspects(self, aspects: list[str], *, max_pages: int = 3) -> dict[str, Any]:
        requested_aspects = _dedupe_strings(aspects)
        rewrites = [self.optimizer.rewrite(aspect) for aspect in requested_aspects]
        search_query_overrides = {
            rewrite.original_aspect: rewrite.optimized_query_en
            for rewrite in rewrites
            if rewrite.optimized_query_en
        }
        keywords_en = _dedupe_strings(
            [keyword for rewrite in rewrites for keyword in rewrite.keywords_en]
        )
        joined_query = " ".join([rewrite.optimized_query_en for rewrite in rewrites if rewrite.optimized_query_en])
        query_plan = AcademicQueryPlan(
            original_query="; ".join(requested_aspects),
            normalized_query_zh="; ".join(requested_aspects),
            retrieval_query_zh="; ".join(requested_aspects),
            retrieval_query_en=joined_query or " ".join(requested_aspects),
            crawler_query_en=joined_query or " ".join(requested_aspects),
            keywords_zh=[],
            keywords_en=keywords_en,
            required_aspects=requested_aspects,
        )
        payload, ingestion_job = self.crawler.crawl_and_collect(
            missing_aspects=requested_aspects,
            query_plan=query_plan,
            search_query_overrides=search_query_overrides,
            max_pages=max_pages,
        )
        return {
            "requested_aspects": requested_aspects,
            "rewrites": [rewrite.model_dump() for rewrite in rewrites],
            "payload": payload.model_dump(),
            "ingestion_job": ingestion_job,
        }

    def run_pending(self, *, max_pages: int = 3) -> dict[str, Any]:
        pending_items = self.list_pending_aspects()
        if not pending_items:
            return {
                "requested_aspects": [],
                "rewrites": [],
                "payload": {
                    "status": "skipped",
                    "message": "No pending missing_aspects in queue.",
                },
                "ingestion_job": None,
            }

        requested_aspects = [item.aspect for item in pending_items]
        result = self.run_aspects(requested_aspects, max_pages=max_pages)
        self.queue_store.remove(requested_aspects)
        return result
