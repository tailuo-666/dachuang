from __future__ import annotations

import csv
import math
import os
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup
from lxml import html

try:
    from ..retrieval.evaluator import compute_support, evaluate_retrieval
    from ..schemas import (
        AcademicQueryPlan,
        AspectEvidence,
        CrawlPaper,
        CrawlPayload,
        CrawlSearchQuery,
        NormalizedDocument,
        SelectedPaper,
    )
except ImportError:
    from retrieval.evaluator import compute_support, evaluate_retrieval
    from schemas import (
        AcademicQueryPlan,
        AspectEvidence,
        CrawlPaper,
        CrawlPayload,
        CrawlSearchQuery,
        NormalizedDocument,
        SelectedPaper,
    )


SUMMARY_CHUNK_SIZE = 450
SUMMARY_CHUNK_OVERLAP = 80
MAX_EVIDENCE_PER_ASPECT = 2
MAX_TOTAL_EVIDENCE = 8


class ArxivCrawlerIntegrated:
    """Structured arXiv crawler used by the single-query academic workflow."""

    def __init__(self, output_dir: str = "./paper_results"):
        self.output_dir = output_dir
        self.all_papers: list[dict[str, Any]] = []
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_phrase(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        cleaned = re.sub(r'["\']+', "", cleaned)
        return cleaned

    def _quote_if_needed(self, phrase: str) -> str:
        phrase = self._sanitize_phrase(phrase)
        if not phrase:
            return ""
        if " " in phrase:
            return f'"{phrase}"'
        return phrase

    def _extract_english_terms(self, text: str) -> list[str]:
        matches = re.findall(r"[A-Za-z][A-Za-z0-9\-\+\.]*(?:\s+[A-Za-z][A-Za-z0-9\-\+\.]*)*", text or "")
        deduped: list[str] = []
        seen: set[str] = set()
        for match in matches:
            candidate = self._sanitize_phrase(match)
            lowered = candidate.lower()
            if not candidate or lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(candidate)
        return deduped[:8]

    def _normalize_aspects(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = self._sanitize_phrase(str(item or ""))
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(text)
        return deduped

    def _paper_key(self, paper: dict[str, Any]) -> str:
        pdf_link = self._sanitize_phrase(str(paper.get("pdf_link", ""))).lower()
        title = self._sanitize_phrase(str(paper.get("title", ""))).lower()
        submission = self._sanitize_phrase(str(paper.get("submission_date", ""))).lower()
        return pdf_link or f"{title}::{submission}"

    def _resolve_aspect_query_text(self, aspect: str, query_plan: AcademicQueryPlan | None) -> str:
        cleaned_aspect = self._sanitize_phrase(aspect)
        if self._extract_english_terms(cleaned_aspect):
            return cleaned_aspect
        if query_plan is not None:
            keywords_en = [self._sanitize_phrase(item) for item in query_plan.keywords_en if self._sanitize_phrase(item)]
            if keywords_en:
                return " ".join(keywords_en[:4])
            fallback = self._sanitize_phrase(query_plan.crawler_query_en)
            if fallback:
                return fallback
        return cleaned_aspect

    def generate_search_query(
        self,
        user_question: str = "",
        *,
        query_en: str | None = None,
        keywords_en: list[str] | None = None,
    ) -> str:
        keywords = [self._sanitize_phrase(item) for item in (keywords_en or []) if self._sanitize_phrase(item)]
        if not keywords:
            keywords = self._extract_english_terms(query_en or user_question)

        if keywords:
            primary_terms = [self._quote_if_needed(term) for term in keywords[:4]]
            return " AND ".join([term for term in primary_terms if term])

        fallback = self._sanitize_phrase(query_en or user_question)
        if fallback:
            return fallback

        return '"retrieval augmented generation" AND "academic search"'

    def build_search_url(self, query: str, start: int = 0, size: int = 50) -> str:
        encoded_query = requests.utils.quote(query)
        return (
            "https://arxiv.org/search/"
            f"?query={encoded_query}&searchtype=abstract&abstracts=show"
            f"&order=-announced_date_first&size={size}&start={start}"
        )

    def get_total_results(self, url: str) -> int:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            tree = html.fromstring(response.content)
            result_string = "".join(tree.xpath('//*[@id="main-container"]/div[1]/div[1]/h1/text()')).strip()
            match = re.search(r"of ([\d,]+) results", result_string)
            return int(match.group(1).replace(",", "")) if match else 0
        except Exception as exc:
            print(f"Failed to fetch result count from arXiv: {exc}")
            return 0

    def fetch_paper_info(self, url: str) -> list[dict[str, Any]]:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            papers = []
            for article in soup.find_all("li", class_="arxiv-result"):
                try:
                    title_element = article.find("p", class_=re.compile(r"title"))
                    authors_element = article.find("p", class_=re.compile(r"authors"))
                    abstract_element = article.find("span", class_=re.compile(r"abstract-full"))
                    meta_element = article.find("p", class_=re.compile(r"is-size-7"))
                    pdf_link_element = article.find("a", string=re.compile(r"pdf", re.I))

                    title = title_element.get_text(" ", strip=True) if title_element else "Untitled"
                    authors_text = authors_element.get_text(" ", strip=True) if authors_element else ""
                    authors_text = authors_text.replace("Authors:", "").strip()
                    authors = [author.strip() for author in authors_text.split(",") if author.strip()]
                    abstract = abstract_element.get_text(" ", strip=True) if abstract_element else ""
                    abstract = re.sub(r"^Abstract:\s*", "", abstract)
                    submitted = meta_element.get_text(" ", strip=True) if meta_element else ""
                    submission_date = submitted.split(";")[0].replace("Submitted", "").strip()

                    pdf_link = ""
                    if pdf_link_element and pdf_link_element.get("href"):
                        pdf_link = pdf_link_element["href"]
                        if pdf_link.startswith("/"):
                            pdf_link = f"https://arxiv.org{pdf_link}"

                    papers.append(
                        {
                            "title": title,
                            "authors": authors,
                            "abstract": abstract,
                            "submission_date": submission_date,
                            "pdf_link": pdf_link,
                        }
                    )
                except Exception as exc:
                    print(f"Failed to parse a paper card from arXiv: {exc}")
            return papers
        except Exception as exc:
            print(f"Failed to fetch arXiv paper info: {exc}")
            return []

    def crawl_papers(
        self,
        user_question: str = "",
        max_pages: int = 5,
        *,
        query_en: str | None = None,
        keywords_en: list[str] | None = None,
        search_query: str | None = None,
    ) -> list[dict[str, Any]]:
        print("Searching arXiv for relevant papers...")
        effective_query = search_query or self.generate_search_query(
            user_question=user_question,
            query_en=query_en,
            keywords_en=keywords_en,
        )
        print(f"Generated arXiv query: {effective_query}")

        if not effective_query:
            self.all_papers = []
            return []

        base_url = self.build_search_url(effective_query)
        total_results = self.get_total_results(base_url)
        if total_results == 0:
            print("No relevant arXiv papers found.")
            self.all_papers = []
            return []

        total_pages = min(math.ceil(total_results / 50), max_pages)
        self.all_papers = []
        for page in range(total_pages):
            start = page * 50
            page_url = self.build_search_url(effective_query, start=start)
            page_papers = self.fetch_paper_info(page_url)
            self.all_papers.extend(page_papers)
            time.sleep(1)

        print(f"Finished arXiv crawl with {len(self.all_papers)} papers.")
        return self.all_papers

    def build_search_queries_for_aspects(
        self,
        missing_aspects: list[str],
        *,
        query_plan: AcademicQueryPlan | None = None,
        search_query_overrides: dict[str, str] | None = None,
    ) -> list[CrawlSearchQuery]:
        queries: list[CrawlSearchQuery] = []
        for aspect in self._normalize_aspects(missing_aspects):
            override_query = ""
            if search_query_overrides:
                override_query = self._sanitize_phrase(search_query_overrides.get(aspect, ""))
            query_text = override_query or self._resolve_aspect_query_text(aspect, query_plan)
            keywords_en = self._extract_english_terms(query_text)
            search_query = self.generate_search_query(query_en=query_text, keywords_en=keywords_en or None)
            if not search_query and query_plan is not None:
                search_query = self.generate_search_query(
                    query_en=query_plan.crawler_query_en,
                    keywords_en=query_plan.keywords_en,
                )
            if search_query:
                queries.append(CrawlSearchQuery(aspect=aspect, query=search_query))
        return queries

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = SUMMARY_CHUNK_SIZE,
        overlap: int = SUMMARY_CHUNK_OVERLAP,
    ) -> list[str]:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return []
        if len(cleaned) <= chunk_size:
            return [cleaned]

        chunks: list[str] = []
        start = 0
        total_length = len(cleaned)
        while start < total_length:
            end = min(total_length, start + chunk_size)
            if end < total_length:
                split_candidates = [
                    cleaned.rfind(". ", start, end),
                    cleaned.rfind("; ", start, end),
                    cleaned.rfind(", ", start, end),
                    cleaned.rfind(" ", start, end),
                ]
                split_at = max(split_candidates)
                if split_at > start + max(120, chunk_size // 3):
                    end = split_at + 1
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= total_length:
                break
            start = max(0, end - overlap)
        return chunks

    def papers_to_evidence_docs(self, papers: list[dict[str, Any]]) -> list[NormalizedDocument]:
        evidence = []
        for paper in papers:
            title = str(paper.get("title", "")).strip()
            abstract = str(paper.get("abstract", "")).strip()
            submission_date = str(paper.get("submission_date", "")).strip()
            pdf_link = str(paper.get("pdf_link", "")).strip()
            authors = paper.get("authors", []) or []
            content = "\n".join(
                [
                    f"Title: {title}",
                    f"Abstract: {abstract}",
                    f"Submitted: {submission_date}",
                    f"PDF: {pdf_link}",
                ]
            ).strip()
            evidence.append(
                NormalizedDocument(
                    content=content,
                    source=title or pdf_link or "arxiv",
                    score=None,
                    metadata={
                        "title": title,
                        "abstract": abstract,
                        "submission_date": submission_date,
                        "pdf_link": pdf_link,
                        "authors": authors,
                        "source": "arxiv",
                    },
                )
            )
        return evidence

    def papers_to_summary_chunks(self, papers: list[dict[str, Any]]) -> list[NormalizedDocument]:
        chunks: list[NormalizedDocument] = []
        for paper in papers:
            title = str(paper.get("title", "")).strip()
            abstract = str(paper.get("abstract", "")).strip()
            submission_date = str(paper.get("submission_date", "")).strip()
            pdf_link = str(paper.get("pdf_link", "")).strip()
            authors = [str(author).strip() for author in (paper.get("authors", []) or []) if str(author).strip()]

            abstract_chunks = self._chunk_text(abstract or title or "No abstract available")
            if not abstract_chunks:
                abstract_chunks = [title or "Untitled paper"]

            total_chunks = len(abstract_chunks)
            for chunk_index, abstract_chunk in enumerate(abstract_chunks, start=1):
                content_lines = [f"Title: {title or 'Untitled'}"]
                if authors:
                    content_lines.append(f"Authors: {', '.join(authors)}")
                if submission_date:
                    content_lines.append(f"Submitted: {submission_date}")
                if pdf_link:
                    content_lines.append(f"PDF: {pdf_link}")
                content_lines.append(f"Abstract: {abstract_chunk}")
                chunk_content = "\n".join(content_lines).strip()

                chunks.append(
                    NormalizedDocument(
                        content=chunk_content,
                        source=title or pdf_link or "arxiv",
                        score=None,
                        metadata={
                            "title": title,
                            "abstract": abstract,
                            "submission_date": submission_date,
                            "pdf_link": pdf_link,
                            "authors": authors,
                            "source": "arxiv",
                            "paper_title": title,
                            "paper_source": pdf_link or title,
                            "chunk_index": chunk_index,
                            "chunk_count": total_chunks,
                        },
                    )
                )
        return chunks

    def _build_aspect_eval_plan(
        self,
        missing_aspects: list[str],
        query_plan: AcademicQueryPlan | None,
    ) -> AcademicQueryPlan:
        required_aspects = self._normalize_aspects(missing_aspects)
        if query_plan is not None:
            return query_plan.model_copy(
                update={
                    "required_aspects": required_aspects,
                    "retrieval_query_en": " ".join(required_aspects) or query_plan.retrieval_query_en,
                    "crawler_query_en": " ".join(required_aspects) or query_plan.crawler_query_en,
                }
            )

        fallback = " ".join(required_aspects)
        return AcademicQueryPlan(
            original_query=fallback,
            normalized_query_zh=fallback,
            retrieval_query_zh=fallback,
            retrieval_query_en=fallback,
            crawler_query_en=fallback,
            keywords_zh=[],
            keywords_en=self._extract_english_terms(fallback),
            required_aspects=required_aspects,
        )

    def _select_aspect_evidence(
        self,
        missing_aspects: list[str],
        summary_chunks: list[NormalizedDocument],
        *,
        query_plan: AcademicQueryPlan | None = None,
        max_per_aspect: int = MAX_EVIDENCE_PER_ASPECT,
        max_total: int = MAX_TOTAL_EVIDENCE,
    ) -> tuple[list[AspectEvidence], list[NormalizedDocument], list[SelectedPaper], list[str], list[str]]:
        requested_aspects = self._normalize_aspects(missing_aspects)
        if not requested_aspects or not summary_chunks:
            return [], [], [], [], requested_aspects

        evaluation = evaluate_retrieval(
            self._build_aspect_eval_plan(requested_aspects, query_plan),
            summary_chunks,
        )
        status_by_aspect: dict[str, str] = {}
        for aspect in requested_aspects:
            if aspect in evaluation.covered_aspects:
                status_by_aspect[aspect] = "covered"
            elif aspect in evaluation.weak_aspects:
                status_by_aspect[aspect] = "weak"
            else:
                status_by_aspect[aspect] = "missing"

        candidates_by_aspect: dict[str, list[dict[str, Any]]] = {}
        for aspect in requested_aspects:
            aspect_candidates: list[dict[str, Any]] = []
            for chunk_index in evaluation.aspect_best_chunks.get(aspect, []):
                if chunk_index >= len(summary_chunks):
                    continue
                score = round(compute_support(aspect, summary_chunks[chunk_index]), 4)
                if score <= 0:
                    continue
                aspect_candidates.append(
                    {
                        "aspect": aspect,
                        "coverage_status": status_by_aspect[aspect],
                        "score": score,
                        "doc": summary_chunks[chunk_index],
                    }
                )
            candidates_by_aspect[aspect] = aspect_candidates[:max_per_aspect]

        selected_candidates: list[dict[str, Any]] = []
        for round_index in range(max_per_aspect):
            for aspect in requested_aspects:
                candidates = candidates_by_aspect.get(aspect, [])
                if round_index < len(candidates):
                    selected_candidates.append(candidates[round_index])
                    if len(selected_candidates) >= max_total:
                        break
            if len(selected_candidates) >= max_total:
                break

        aspect_evidence: list[AspectEvidence] = []
        evidence_doc_map: dict[str, NormalizedDocument] = {}
        selected_paper_map: dict[str, SelectedPaper] = {}
        for item in selected_candidates:
            doc = item["doc"]
            metadata = dict(doc.metadata or {})
            paper_title = str(metadata.get("paper_title") or metadata.get("title") or doc.source).strip()
            paper_source = str(metadata.get("paper_source") or metadata.get("pdf_link") or doc.source).strip()
            pdf_link = str(metadata.get("pdf_link", "")).strip()
            authors = [str(author).strip() for author in metadata.get("authors", []) or [] if str(author).strip()]
            submission_date = str(metadata.get("submission_date", "")).strip()

            aspect_evidence.append(
                AspectEvidence(
                    aspect=item["aspect"],
                    coverage_status=item["coverage_status"],
                    chunk=doc.content,
                    score=item["score"],
                    paper_title=paper_title,
                    paper_source=paper_source,
                    pdf_link=pdf_link,
                    authors=authors,
                    submission_date=submission_date,
                )
            )

            doc_key = f"{doc.source}::{doc.content}"
            previous_doc = evidence_doc_map.get(doc_key)
            previous_aspects = []
            if previous_doc is not None:
                previous_aspects = [
                    str(value).strip()
                    for value in (previous_doc.metadata or {}).get("matched_aspects", [])
                    if str(value).strip()
                ]
            matched_aspects = sorted({item["aspect"], *previous_aspects})
            updated_metadata = dict(metadata)
            updated_metadata["matched_aspects"] = matched_aspects
            updated_doc = doc.model_copy(
                update={
                    "score": max(float(doc.score or 0.0), float(item["score"])),
                    "metadata": updated_metadata,
                }
            )
            evidence_doc_map[doc_key] = updated_doc

            paper_key = self._paper_key(
                {
                    "title": paper_title,
                    "pdf_link": pdf_link,
                    "submission_date": submission_date,
                }
            )
            if paper_key not in selected_paper_map:
                selected_paper_map[paper_key] = SelectedPaper(
                    title=paper_title,
                    abstract=str(metadata.get("abstract", "")).strip(),
                    submission_date=submission_date,
                    pdf_link=pdf_link,
                    authors=authors,
                    matched_aspects=[item["aspect"]],
                )
            else:
                existing = selected_paper_map[paper_key]
                existing.matched_aspects = sorted({*existing.matched_aspects, item["aspect"]})

        covered_missing_aspects = [aspect for aspect in requested_aspects if status_by_aspect.get(aspect) == "covered"]
        uncovered_missing_aspects = [aspect for aspect in requested_aspects if status_by_aspect.get(aspect) != "covered"]
        return (
            aspect_evidence,
            list(evidence_doc_map.values()),
            list(selected_paper_map.values()),
            covered_missing_aspects,
            uncovered_missing_aspects,
        )

    def _build_ingest_shortlist(
        self,
        papers: list[dict[str, Any]],
        selected_papers: list[SelectedPaper],
        *,
        max_downloads: int = 3,
    ) -> list[dict[str, Any]]:
        if not papers:
            return []
        if not selected_papers:
            return papers[:max_downloads]

        selected_keys = {
            self._paper_key(
                {
                    "title": paper.title,
                    "submission_date": paper.submission_date,
                    "pdf_link": paper.pdf_link,
                }
            )
            for paper in selected_papers
        }
        shortlist = [paper for paper in papers if self._paper_key(paper) in selected_keys]
        return shortlist[:max_downloads] if shortlist else papers[:max_downloads]

    def crawl_and_collect(
        self,
        *,
        missing_aspects: list[str],
        query_plan: AcademicQueryPlan | None = None,
        search_query_overrides: dict[str, str] | None = None,
        max_pages: int = 3,
        max_per_aspect: int = MAX_EVIDENCE_PER_ASPECT,
        max_total_evidence: int = MAX_TOTAL_EVIDENCE,
    ) -> tuple[CrawlPayload, dict[str, Any] | None]:
        requested_aspects = self._normalize_aspects(missing_aspects)
        if not requested_aspects:
            payload = CrawlPayload(
                status="empty",
                message="missing_aspects is empty, skip arXiv crawl.",
                requested_missing_aspects=[],
                covered_missing_aspects=[],
                uncovered_missing_aspects=[],
                search_queries=[],
                aspect_evidence=[],
                evidence_docs=[],
                selected_papers=[],
                ingestion_status="skipped",
                pending_ingest_paper_count=0,
            )
            return payload, None

        search_queries = self.build_search_queries_for_aspects(
            requested_aspects,
            query_plan=query_plan,
            search_query_overrides=search_query_overrides,
        )
        if not search_queries and query_plan is not None:
            fallback_query = self.generate_search_query(
                query_en=query_plan.crawler_query_en,
                keywords_en=query_plan.keywords_en,
            )
            if fallback_query:
                search_queries = [CrawlSearchQuery(aspect=aspect, query=fallback_query) for aspect in requested_aspects]

        deduped_papers: dict[str, dict[str, Any]] = {}
        query_cache: dict[str, list[dict[str, Any]]] = {}
        for search in search_queries:
            if search.query not in query_cache:
                query_cache[search.query] = self.crawl_papers(search_query=search.query, max_pages=max_pages)
            for paper in query_cache[search.query]:
                paper_key = self._paper_key(paper)
                if paper_key not in deduped_papers:
                    deduped_papers[paper_key] = paper

        papers_raw = list(deduped_papers.values())
        self.all_papers = papers_raw
        summary_chunks = self.papers_to_summary_chunks(papers_raw)
        aspect_evidence, evidence_docs, selected_papers, covered_missing_aspects, uncovered_missing_aspects = (
            self._select_aspect_evidence(
                requested_aspects,
                summary_chunks,
                query_plan=query_plan,
                max_per_aspect=max_per_aspect,
                max_total=max_total_evidence,
            )
        )
        ingest_shortlist = self._build_ingest_shortlist(papers_raw, selected_papers, max_downloads=3)

        if not papers_raw:
            status = "empty"
        elif covered_missing_aspects and not uncovered_missing_aspects:
            status = "success"
        elif aspect_evidence:
            status = "partial_success"
        else:
            status = "empty"

        message = (
            f"arXiv found {len(papers_raw)} papers, "
            f"returned {len(aspect_evidence)} summary chunks, "
            f"covered {len(covered_missing_aspects)}/{len(requested_aspects)} missing_aspects."
        )
        if uncovered_missing_aspects:
            message = f"{message} Uncovered: {', '.join(uncovered_missing_aspects[:4])}."

        payload = CrawlPayload(
            status=status,
            message=message,
            requested_missing_aspects=requested_aspects,
            covered_missing_aspects=covered_missing_aspects,
            uncovered_missing_aspects=uncovered_missing_aspects,
            search_queries=search_queries,
            aspect_evidence=aspect_evidence,
            evidence_docs=evidence_docs,
            selected_papers=selected_papers,
            ingestion_status="pending" if ingest_shortlist else "skipped",
            pending_ingest_paper_count=len(ingest_shortlist),
        )

        ingestion_job = None
        if papers_raw:
            ingestion_job = {
                "output_dir": self.output_dir,
                "requested_missing_aspects": requested_aspects,
                "search_queries": [item.model_dump() for item in search_queries],
                "all_papers": [dict(paper) for paper in papers_raw],
                "selected_papers": [paper.model_dump() for paper in selected_papers],
                "ingest_papers": [dict(paper) for paper in ingest_shortlist],
                "manifest_csv": "paper_result.csv",
                "manifest_txt": "formatted_papers.txt",
            }
        return payload, ingestion_job

    def save_to_csv(self, papers: list[dict[str, Any]] | None = None, filename: str = "paper_result.csv") -> bool:
        if papers is None:
            papers = self.all_papers
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["title", "authors", "abstract", "submission_date", "pdf_link"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for paper in papers:
                    writer.writerow(paper)
            print(f"Saved arXiv paper manifest to {filepath}")
            return True
        except Exception as exc:
            print(f"Failed to save CSV manifest: {exc}")
            return False

    def read_csv(self, filename: str) -> list[dict[str, Any]]:
        try:
            filepath = os.path.join(self.output_dir, filename)
            papers = []
            with open(filepath, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    papers.append(
                        {
                            "title": row.get("title", ""),
                            "submission_date": row.get("submission_date", ""),
                            "pdf_link": row.get("pdf_link", ""),
                            "abstract": row.get("abstract", ""),
                            "authors": row.get("authors", ""),
                        }
                    )
            return papers
        except Exception as exc:
            print(f"Failed to read CSV manifest: {exc}")
            return []

    def extract_year(self, submission: str) -> str:
        match = re.search(r"\d{4}", submission or "")
        return match.group(0) if match else "Unknown"

    def format_paper(self, paper: dict[str, Any]) -> str:
        title = str(paper.get("title", "")).strip()
        year = self.extract_year(str(paper.get("submission_date", "")))
        pdf_link = str(paper.get("pdf_link", "")).strip()
        return f"+ {title}, arXiv {year}, [[paper]]({pdf_link})."

    def generate_paper_list(self, filename: str) -> list[str]:
        papers = self.read_csv(filename)
        return [self.format_paper(paper) for paper in papers]

    def save_formatted_papers(self, papers: list[str] | None = None, filename: str = "formatted_papers.txt") -> bool:
        if papers is None:
            papers = self.generate_paper_list("paper_result.csv")
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as file:
                for paper in papers:
                    file.write(paper + "\n")
            print(f"Saved formatted paper list to {filepath}")
            return True
        except Exception as exc:
            print(f"Failed to save formatted paper list: {exc}")
            return False

    def extract_papers_from_file(self, filename: str) -> list[dict[str, Any]]:
        try:
            filepath = os.path.join(self.output_dir, filename)
            paper_data = []
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                pattern = r"\+\s(.+?),\s(?:arxiv|arXiv)\s\d{4},\s\[\[paper\]\]\((https://arxiv\.org/pdf/[^)]+)\)\."
                matches = re.findall(pattern, content)
                for title, paper_link in matches:
                    paper_data.append({"title": title.strip(), "paper_link": paper_link})
            print(f"Extracted {len(paper_data)} papers from formatted list.")
            return paper_data
        except Exception as exc:
            print(f"Failed to extract papers from file: {exc}")
            return []

    def _clean_filename(self, filename: str) -> str:
        illegal_chars = r'[<>:"/\\|?*]'
        clean_name = re.sub(illegal_chars, "", filename or "")
        clean_name = re.sub(r"\s+", " ", clean_name).strip()
        if len(clean_name) > 80:
            clean_name = clean_name[:80]
        return clean_name

    def download_paper(self, paper_link: str, filepath: str) -> bool:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(paper_link, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"Failed to download PDF, status code: {response.status_code}")
                return False
            with open(filepath, "wb") as file:
                file.write(response.content)
            return True
        except Exception as exc:
            print(f"Download error: {exc}")
            return False

    def download_papers(
        self,
        papers: list[dict[str, Any]] | None = None,
        max_downloads: int = 10,
        source: str = "memory",
    ) -> int:
        if papers is None:
            if source == "file":
                papers = self.extract_papers_from_file("formatted_papers.txt")
            else:
                papers = self.all_papers[:max_downloads]
        else:
            papers = papers[:max_downloads]

        if not papers:
            print("No papers available for download.")
            return 0

        print(f"Preparing to download {len(papers)} papers.")
        success_count = 0
        for index, paper in enumerate(papers, start=1):
            paper_link = paper.get("pdf_link") or paper.get("paper_link")
            if not paper_link:
                continue

            clean_title = self._clean_filename(str(paper.get("title", f"paper_{index}")))
            filepath = os.path.join(self.output_dir, f"{clean_title}.pdf")
            if os.path.exists(filepath):
                print(f"File already exists, skipping: {os.path.basename(filepath)}")
                success_count += 1
                continue

            print(f"[{index}/{len(papers)}] Downloading: {paper.get('title', '')}")
            if self.download_paper(str(paper_link), filepath):
                print(f"Downloaded: {os.path.basename(filepath)}")
                success_count += 1
            else:
                print(f"Download failed: {os.path.basename(filepath)}")
            time.sleep(1)

        print(f"Download finished. Success {success_count}/{len(papers)}.")
        return success_count
