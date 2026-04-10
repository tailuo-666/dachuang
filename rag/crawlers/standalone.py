from __future__ import annotations

import argparse
import json
import os
import re
import sys
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


def _set_env_if_value(name: str, value: str | None) -> None:
    cleaned = str(value or "").strip()
    if cleaned:
        os.environ[name] = cleaned


def _set_or_clear_env(name: str, value: str | None) -> None:
    cleaned = str(value or "").strip()
    if cleaned:
        os.environ[name] = cleaned
    else:
        os.environ.pop(name, None)


def apply_runtime_args(args: argparse.Namespace) -> None:
    _set_or_clear_env("RAG_LLM_BASE_URL", getattr(args, "llm_base_url", ""))
    _set_or_clear_env("RAG_EMBEDDING_BASE_URL", getattr(args, "embedding_base_url", ""))
    _set_or_clear_env("RAG_OCR_BASE_URL", getattr(args, "ocr_base_url", ""))

    if getattr(args, "use_ssh", False):
        _set_env_if_value("RAG_SSH_HOST", getattr(args, "ssh_host", ""))
        _set_env_if_value("RAG_SSH_PORT", str(getattr(args, "ssh_port", "")))
        _set_env_if_value("RAG_SSH_USERNAME", getattr(args, "ssh_username", ""))
        _set_env_if_value("RAG_SSH_PASSWORD", getattr(args, "ssh_password", ""))

        _set_env_if_value("RAG_LLM_REMOTE_PORT", str(getattr(args, "llm_remote_port", "")))
        _set_env_if_value("RAG_LLM_LOCAL_PORT", str(getattr(args, "llm_local_port", "")))
        _set_env_if_value("RAG_EMBEDDING_REMOTE_PORT", str(getattr(args, "embedding_remote_port", "")))
        _set_env_if_value("RAG_EMBEDDING_LOCAL_PORT", str(getattr(args, "embedding_local_port", "")))
        _set_env_if_value("RAG_OCR_REMOTE_PORT", str(getattr(args, "ocr_remote_port", "")))
        _set_env_if_value("RAG_OCR_LOCAL_PORT", str(getattr(args, "ocr_local_port", "")))

        # When using SSH, clear direct base URLs so downstream factories resolve through tunnels.
        _set_or_clear_env("RAG_LLM_BASE_URL", "")
        _set_or_clear_env("RAG_EMBEDDING_BASE_URL", "")
        _set_or_clear_env("RAG_OCR_BASE_URL", "")


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
{{
  "original_aspect": "...",
  "optimized_query_en": "...",
  "keywords_en": ["..."]
}}

Rules:
1. Rewrite the input aspect into a short English query that is likely to match wording used in academic paper abstracts or titles.
2. Keep the rewritten query semantically aligned with the original aspect. Do not add unrelated tasks, applications, datasets, benchmarks, or methods.
3. The input may be a definition-style request such as "definition of LLM", "what is CNN", or "explain RNN".
4. For definition-style inputs, do not keep request phrases such as:
   - "definition of"
   - "what is"
   - "explain"
   - "introduction to"
   Instead, rewrite toward wording more likely to appear in abstracts, using this priority:
   a. the canonical academic term itself
   b. "<canonical term> is" or "<canonical term> are" when this sounds natural in academic prose
   c. "overview of <canonical term>" only if the bare term would be too vague
5. Prefer canonical academic terminology and expanded forms of abbreviations when useful for retrieval. For example:
   - LLM -> large language model / large language models
   - CNN -> convolutional neural network / convolutional neural networks
   - RNN -> recurrent neural network / recurrent neural networks
6. `optimized_query_en` must be short, search-like, and usually 2 to 6 words.
7. Prefer phrases that could realistically appear verbatim in an abstract sentence or title.
8. Avoid conversational wording, question forms, and long descriptions.
9. `keywords_en` must contain only deduplicated academic phrases useful for retrieval, with at most 6 items.
10. If both an abbreviation and its full form are standard in the literature, include both in `keywords_en`.
11. Prefer plural forms for broad concept classes when natural in academic writing, such as "large language models" or "recurrent neural networks".
12. Do not output any explanation or commentary. Output strict JSON only.

Examples:

Input: definition of LLM
Output:
{{
  "original_aspect": "definition of LLM",
  "optimized_query_en": "large language models are",
  "keywords_en": [
    "large language models",
    "LLM",
    "autoregressive language models",
    "transformer language models"
  ]
}}

Input: definition of CNN
Output:
{{
  "original_aspect": "definition of CNN",
  "optimized_query_en": "convolutional neural networks are",
  "keywords_en": [
    "convolutional neural networks",
    "CNN",
    "deep convolutional neural networks",
    "convolutional architectures"
  ]
}}

Input: definition of RNN
Output:
{{
  "original_aspect": "definition of RNN",
  "optimized_query_en": "recurrent neural networks are",
  "keywords_en": [
    "recurrent neural networks",
    "RNN",
    "recurrent architectures",
    "sequence models"
  ]
}}

Input: what is transformer
Output:
{{
  "original_aspect": "what is transformer",
  "optimized_query_en": "transformer architecture",
  "keywords_en": [
    "transformer",
    "transformer architecture",
    "self-attention",
    "attention mechanism"
  ]
}}

Input: explain attention mechanism
Output:
{{
  "original_aspect": "explain attention mechanism",
  "optimized_query_en": "attention mechanism",
  "keywords_en": [
    "attention mechanism",
    "self-attention",
    "attention models"
  ]
}}

Input: introduction to graph neural networks
Output:
{{
  "original_aspect": "introduction to graph neural networks",
  "optimized_query_en": "graph neural networks",
  "keywords_en": [
    "graph neural networks",
    "GNN",
    "graph representation learning",
    "message passing neural networks"
  ]
}}

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
        except Exception as exc:
            print(
                "Warning: failed to rewrite missing aspect "
                f"{cleaned_aspect!r} via LLM ({type(exc).__name__}: {exc}); "
                "falling back to original aspect."
            )
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
        md_output_dir: str = "./md",
        queue_path: str = DEFAULT_QUEUE_PATH,
        llm=None,
        pdf_processor=None,
        rag_system=None,
    ) -> None:
        self.output_dir = output_dir
        self.md_output_dir = md_output_dir
        self.queue_store = MissingAspectQueueStore(queue_path)
        self.optimizer = MissingAspectQueryOptimizer(llm=llm)
        self.crawler = ArxivCrawlerIntegrated(output_dir=output_dir)
        self.pdf_processor = pdf_processor
        self.rag_system = rag_system

    def enqueue_aspects(self, aspects: list[str]) -> list[QueuedMissingAspect]:
        return self.queue_store.enqueue(aspects)

    def list_pending_aspects(self) -> list[QueuedMissingAspect]:
        return self.queue_store.list_items()

    def run_aspects(
        self,
        aspects: list[str],
        *,
        max_pages: int = 3,
        max_new_papers: int = 5,
        auto_ingest: bool = True,
    ) -> dict[str, Any]:
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
        ingestion_result = None
        if auto_ingest and ingestion_job:
            ingestion_result = self.crawler.execute_ingestion_job(
                ingestion_job,
                md_output_dir=self.md_output_dir,
                max_new_papers=max_new_papers,
                pdf_processor=self.pdf_processor,
                rag_system=self.rag_system,
            )
        return {
            "requested_aspects": requested_aspects,
            "rewrites": [rewrite.model_dump() for rewrite in rewrites],
            "payload": payload.model_dump(),
            "ingestion_job": ingestion_job,
            "ingestion_result": ingestion_result,
        }

    def run_pending(
        self,
        *,
        max_pages: int = 3,
        max_new_papers: int = 5,
        auto_ingest: bool = True,
    ) -> dict[str, Any]:
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
                "ingestion_result": None,
            }

        requested_aspects = [item.aspect for item in pending_items]
        result = self.run_aspects(
            requested_aspects,
            max_pages=max_pages,
            max_new_papers=max_new_papers,
            auto_ingest=auto_ingest,
        )
        ingestion_status = str((result.get("ingestion_result") or {}).get("status") or "").strip().lower()
        if not auto_ingest or ingestion_status in {"", "success", "partial_success", "skipped"}:
            self.queue_store.remove(requested_aspects)
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone missing-aspect arXiv crawler.")
    parser.add_argument("--aspect", action="append", default=[], help="Missing aspect to search and ingest.")
    parser.add_argument("--pending", action="store_true", help="Run the crawler on queued pending aspects.")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--max-new-papers", type=int, default=5)
    parser.add_argument("--output-dir", default="./paper_results")
    parser.add_argument("--md-output-dir", default="./md")
    parser.add_argument("--queue-path", default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--no-auto-ingest", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--use-ssh", action="store_true", help="Use SSH tunnel mode for LLM, OCR, and embedding services.")
    parser.add_argument("--ssh-host", default=os.getenv("RAG_SSH_HOST", ""))
    parser.add_argument("--ssh-port", type=int, default=int(os.getenv("RAG_SSH_PORT", "8888")))
    parser.add_argument("--ssh-username", default=os.getenv("RAG_SSH_USERNAME", ""))
    parser.add_argument("--ssh-password", default=os.getenv("RAG_SSH_PASSWORD", ""))
    parser.add_argument("--llm-remote-port", type=int, default=int(os.getenv("RAG_LLM_REMOTE_PORT", "8001")))
    parser.add_argument("--llm-local-port", type=int, default=int(os.getenv("RAG_LLM_LOCAL_PORT", "18001")))
    parser.add_argument("--embedding-remote-port", type=int, default=int(os.getenv("RAG_EMBEDDING_REMOTE_PORT", "8000")))
    parser.add_argument("--embedding-local-port", type=int, default=int(os.getenv("RAG_EMBEDDING_LOCAL_PORT", "18000")))
    parser.add_argument("--ocr-remote-port", type=int, default=int(os.getenv("RAG_OCR_REMOTE_PORT", "8002")))
    parser.add_argument("--ocr-local-port", type=int, default=int(os.getenv("RAG_OCR_LOCAL_PORT", "18002")))
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("RAG_LLM_BASE_URL", ""),
        help="Direct LLM base_url. Leave empty when using --use-ssh.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("RAG_EMBEDDING_BASE_URL", ""),
        help="Direct embedding base_url. Leave empty when using --use-ssh.",
    )
    parser.add_argument(
        "--ocr-base-url",
        default=os.getenv("RAG_OCR_BASE_URL", ""),
        help="Direct OCR base_url. Leave empty when using --use-ssh.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_runtime_args(args)

    if not args.pending and not args.aspect:
        parser.error("Provide at least one --aspect or pass --pending.")

    crawler = StandaloneMissingAspectCrawler(
        output_dir=args.output_dir,
        md_output_dir=args.md_output_dir,
        queue_path=args.queue_path,
    )
    if args.pending:
        result = crawler.run_pending(
            max_pages=args.max_pages,
            max_new_papers=args.max_new_papers,
            auto_ingest=not args.no_auto_ingest,
        )
    else:
        result = crawler.run_aspects(
            args.aspect,
            max_pages=args.max_pages,
            max_new_papers=args.max_new_papers,
            auto_ingest=not args.no_auto_ingest,
        )

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print((result.get("payload") or {}).get("message") or result)
        ingestion_result = result.get("ingestion_result") or {}
        if ingestion_result:
            print(ingestion_result.get("message", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
