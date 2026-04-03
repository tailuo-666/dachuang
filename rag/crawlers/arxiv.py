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
    from ..schemas import CrawlPaper, CrawlPayload, NormalizedDocument
except ImportError:
    from schemas import CrawlPaper, CrawlPayload, NormalizedDocument


class ArxivCrawlerIntegrated:
    """Structured arXiv crawler used by the single-query academic workflow."""

    def __init__(self, output_dir: str = "./paper_results"):
        self.output_dir = output_dir
        self.all_papers: list[dict[str, Any]] = []
        self._ensure_directories()

    def _ensure_directories(self):
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
        deduped = []
        seen = set()
        for match in matches:
            candidate = self._sanitize_phrase(match)
            lowered = candidate.lower()
            if not candidate or lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(candidate)
        return deduped[:8]

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
            print(f"获取总结果数失败: {exc}")
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
                    print(f"解析单篇论文信息失败: {exc}")
            return papers
        except Exception as exc:
            print(f"获取论文信息失败: {exc}")
            return []

    def crawl_papers(
        self,
        user_question: str = "",
        max_pages: int = 5,
        *,
        query_en: str | None = None,
        keywords_en: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        print("正在根据问题搜索相关论文...")
        search_query = self.generate_search_query(
            user_question=user_question,
            query_en=query_en,
            keywords_en=keywords_en,
        )
        print(f"生成的搜索查询: {search_query}")

        base_url = self.build_search_url(search_query)
        total_results = self.get_total_results(base_url)
        if total_results == 0:
            print("未找到相关论文")
            self.all_papers = []
            return []

        total_pages = min(math.ceil(total_results / 50), max_pages)
        self.all_papers = []

        for page in range(total_pages):
            start = page * 50
            print(f"爬取页面 {page + 1}/{total_pages}, 起始位置: {start}")
            page_url = self.build_search_url(search_query, start=start)
            page_papers = self.fetch_paper_info(page_url)
            self.all_papers.extend(page_papers)
            time.sleep(1)

        print(f"爬取完成！共获取 {len(self.all_papers)} 篇相关论文")
        return self.all_papers

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

    def crawl_and_collect(
        self,
        *,
        query_en: str,
        keywords_en: list[str],
        max_pages: int = 3,
    ) -> CrawlPayload:
        papers_raw = self.crawl_papers(query_en, max_pages=max_pages, query_en=query_en, keywords_en=keywords_en)
        papers = [CrawlPaper(**paper) for paper in papers_raw]
        evidence_docs = self.papers_to_evidence_docs(papers_raw)
        status = "success" if papers else "empty"
        message = f"爬虫找到 {len(papers)} 篇论文，可直接使用标题与摘要作为补充证据。"
        return CrawlPayload(
            status=status,
            message=message,
            query=self.generate_search_query(query_en=query_en, keywords_en=keywords_en),
            query_en=query_en,
            keywords_en=keywords_en,
            papers=papers,
            evidence_docs=evidence_docs,
            downloaded_count=0,
            indexed_doc_count=0,
        )

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
            print(f"论文信息已保存到 {filepath}")
            return True
        except Exception as exc:
            print(f"保存CSV文件失败: {exc}")
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
            print(f"读取CSV文件失败: {exc}")
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
            print(f"格式化论文列表已保存到 {filepath}")
            return True
        except Exception as exc:
            print(f"保存格式化论文失败: {exc}")
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
            print(f"从文件中成功提取 {len(paper_data)} 篇论文")
            return paper_data
        except Exception as exc:
            print(f"读取文件时出错: {exc}")
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
                print(f"下载失败，状态码: {response.status_code}")
                return False
            with open(filepath, "wb") as file:
                file.write(response.content)
            return True
        except Exception as exc:
            print(f"下载出错: {exc}")
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
            print("没有论文可下载")
            return 0

        print(f"准备下载 {len(papers)} 篇论文")
        success_count = 0
        for index, paper in enumerate(papers, start=1):
            paper_link = paper.get("pdf_link") or paper.get("paper_link")
            if not paper_link:
                continue

            clean_title = self._clean_filename(str(paper.get("title", f"paper_{index}")))
            filepath = os.path.join(self.output_dir, f"{clean_title}.pdf")
            if os.path.exists(filepath):
                print(f"文件已存在，跳过: {os.path.basename(filepath)}")
                success_count += 1
                continue

            print(f"\n[{index}/{len(papers)}] 正在下载: {paper.get('title', '')}")
            if self.download_paper(str(paper_link), filepath):
                print(f"✓ 下载成功: {os.path.basename(filepath)}")
                success_count += 1
            else:
                print(f"✗ 下载失败: {os.path.basename(filepath)}")
            time.sleep(1)

        print(f"\n下载完成！成功下载 {success_count}/{len(papers)} 篇论文")
        return success_count
