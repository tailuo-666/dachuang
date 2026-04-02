from __future__ import annotations

from typing import Any


def retrieval_docs_high_match() -> list[dict[str, Any]]:
    return [
        {
            "content": (
                "This paper studies retrieval augmented generation, query rewriting, "
                "vector database indexing, and relevance evaluation for academic search."
            ),
            "source": "paper_a.md",
            "score": None,
            "metadata": {"title": "RAG Query Rewriting for Academic Search"},
        },
        {
            "content": (
                "Academic retrieval systems often combine BM25, vector retrieval, "
                "and terminology normalization for multilingual RAG workflows."
            ),
            "source": "paper_b.md",
            "score": None,
            "metadata": {"title": "Hybrid Retrieval in Multilingual RAG"},
        },
    ]


def retrieval_docs_low_match() -> list[dict[str, Any]]:
    return [
        {
            "content": "This paper is about image compression and video codecs.",
            "source": "paper_c.md",
            "score": None,
            "metadata": {"title": "A Survey on Video Compression"},
        }
    ]


def fake_crawl_papers() -> list[dict[str, Any]]:
    return [
        {
            "title": "Academic Query Optimization for RAG",
            "abstract": "We study multilingual query normalization and academic retrieval.",
            "submission_date": "12 Jan 2025",
            "pdf_link": "https://arxiv.org/pdf/2501.00001",
            "authors": ["Alice", "Bob"],
        }
    ]

