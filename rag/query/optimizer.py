from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

try:
    from ..schemas import AcademicQueryPlan
except ImportError:
    from schemas import AcademicQueryPlan


COMMON_TRANSLATIONS = {
    "检索增强生成": "retrieval augmented generation",
    "机器学习": "machine learning",
    "深度学习": "deep learning",
    "强化学习": "reinforcement learning",
    "向量数据库": "vector database",
    "向量检索": "vector retrieval",
    "多模态": "multimodal",
    "知识图谱": "knowledge graph",
    "提示词工程": "prompt engineering",
    "大语言模型": "large language model",
    "文档重排序": "document reranking",
    "相关性评估": "relevance evaluation",
    "信息检索": "information retrieval",
    "问答系统": "question answering system",
    "问答": "question answering",
    "学术搜索": "academic search",
    "论文爬虫": "paper crawler",
    "中文": "Chinese",
    "英文": "English",
    "翻译": "translation",
    "术语": "terminology",
    "查询重写": "query rewriting",
    "召回": "recall",
    "精确率": "precision",
    "鲁棒性": "robustness",
    "agent": "agent",
    "rag": "RAG",
    "llm": "LLM",
    "ocr": "OCR",
    "bm25": "BM25",
    "arxiv": "arXiv",
}

CHINESE_STOPWORDS = {
    "什么",
    "如何",
    "怎么",
    "为什么",
    "一个",
    "以及",
    "对于",
    "问题",
    "研究",
    "进行",
    "有关",
    "这个",
    "那个",
    "并且",
    "或者",
    "可以",
    "需要",
    "场景",
    "部分",
}

COMPARISON_PATTERN = re.compile(r"(为什么|为何|区别|对比|比较|优于|更好|优势|vs|VS|比)")


class AcademicQueryPlanner:
    """Build a single-query academic retrieval plan."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = self._build_chain()

    def _build_chain(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the query planner for an academic RAG workflow. "
                        "Convert a Chinese or mixed Chinese-English user question into one JSON object for "
                        "local knowledge-base retrieval, retrieval relevance evaluation, Tavily search, and "
                        "a standalone academic crawler. Return JSON only."
                    ),
                ),
                (
                    "human",
                    """
Return strict JSON with exactly these fields:
{
  "original_query": "...",
  "normalized_query_zh": "...",
  "retrieval_query_zh": "...",
  "retrieval_query_en": "...",
  "crawler_query_en": "...",
  "keywords_zh": ["..."],
  "keywords_en": ["..."],
  "required_aspects": ["..."]
}

Rules:
1. `normalized_query_zh` should be a polished Chinese version of the original intent.
2. `retrieval_query_zh` is for local Chinese retrieval and should stay concise and natural.
3. `retrieval_query_en` is for semantic retrieval and must preserve the original meaning instead of literal word-by-word translation.
4. `crawler_query_en` is only a short fallback academic search string for title/abstract search.
5. `keywords_zh` and `keywords_en` must be deduplicated and ordered by importance.
6. `required_aspects` is the most important field. It must be English search-ready noun phrases, not sentences, and at most 5 items.
7. Every required aspect must be independently searchable and independently judgeable later.
8. Prefer aspects framed as definition, mechanism, difference, reason, method, limitation, or evidence.
9. For comparison questions such as "why A is better than B", prefer:
   - definition of A
   - definition of B
   - differences between A and B
   - advantages of A over B
10. Preserve standard English academic terms and abbreviations such as RAG, LLM, BM25, OCR, and arXiv.
11. Avoid invented translations. Keep standard English terms when they are known.
12. `keywords_en` should prefer academic phrases such as "query rewriting", "information retrieval", and "question answering".
13. `crawler_query_en` should be shorter than `retrieval_query_en` and remove weak verbs like "how", "improve", "use", and "study" when possible.
14. `required_aspects` must be directly usable as Tavily search inputs later.
15. Do not output markdown. Do not output explanations.

Example:
{
  "original_query": "Transformer 为什么比 RNN 好？",
  "normalized_query_zh": "Transformer 相比 RNN 的优势来源是什么？",
  "retrieval_query_zh": "Transformer RNN 区别 优势 原因",
  "retrieval_query_en": "why Transformer outperforms RNN",
  "crawler_query_en": "Transformer RNN differences advantages",
  "keywords_zh": ["Transformer", "RNN", "区别", "优势"],
  "keywords_en": ["Transformer", "RNN", "differences", "advantages"],
  "required_aspects": [
    "definition of Transformer",
    "definition of RNN",
    "differences between Transformer and RNN",
    "advantages of Transformer over RNN"
  ]
}

User question: {question}
                    """.strip(),
                ),
            ]
        )

    def build(self, original_query: str) -> AcademicQueryPlan:
        """Return a best-effort query plan."""
        query = (original_query or "").strip()
        if not query:
            return self._fallback_plan("")

        try:
            prompt_value = self.prompt.invoke({"question": query})
            response = self.llm.invoke(prompt_value)
            raw_text = response.content if hasattr(response, "content") else str(response)
            data = self._extract_json(raw_text)
            return self._coerce_plan(query, data)
        except Exception:
            return self._fallback_plan(query)

    def _extract_json(self, text: str) -> dict[str, Any]:
        cleaned = (text or "").strip()
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

    def _coerce_plan(self, original_query: str, data: dict[str, Any]) -> AcademicQueryPlan:
        keywords_zh = self._normalize_string_list(data.get("keywords_zh"))
        keywords_en = self._normalize_string_list(data.get("keywords_en"))

        if not keywords_zh:
            keywords_zh = self._extract_keywords_zh(original_query)
        if not keywords_en:
            keywords_en = self._translate_keywords(keywords_zh, original_query)
        required_aspects = self._normalize_required_aspects(
            value=data.get("required_aspects"),
            original_query=original_query,
            keywords_zh=keywords_zh,
        )

        plan = AcademicQueryPlan(
            original_query=original_query,
            normalized_query_zh=str(data.get("normalized_query_zh") or original_query).strip(),
            retrieval_query_zh=str(data.get("retrieval_query_zh") or original_query).strip(),
            retrieval_query_en=str(data.get("retrieval_query_en") or " ".join(keywords_en)).strip(),
            crawler_query_en=str(data.get("crawler_query_en") or " ".join(keywords_en)).strip(),
            keywords_zh=keywords_zh,
            keywords_en=keywords_en,
            required_aspects=required_aspects,
        )
        if not plan.normalized_query_zh:
            plan.normalized_query_zh = original_query
        if not plan.retrieval_query_zh:
            plan.retrieval_query_zh = plan.normalized_query_zh
        if not plan.retrieval_query_en:
            plan.retrieval_query_en = " ".join(plan.keywords_en) or original_query
        if not plan.crawler_query_en:
            plan.crawler_query_en = plan.retrieval_query_en
        if not plan.required_aspects:
            plan.required_aspects = self._infer_required_aspects(original_query, plan.keywords_zh)
        return plan

    def _fallback_plan(self, original_query: str) -> AcademicQueryPlan:
        keywords_zh = self._extract_keywords_zh(original_query)
        keywords_en = self._translate_keywords(keywords_zh, original_query)
        required_aspects = self._infer_required_aspects(original_query, keywords_zh)
        return AcademicQueryPlan(
            original_query=original_query,
            normalized_query_zh=original_query,
            retrieval_query_zh=original_query,
            retrieval_query_en=" ".join(keywords_en) or original_query,
            crawler_query_en=" ".join(keywords_en) or original_query,
            keywords_zh=keywords_zh,
            keywords_en=keywords_en,
            required_aspects=required_aspects,
        )

    def _extract_keywords_zh(self, query: str) -> list[str]:
        chunks = re.findall(r"[A-Za-z][A-Za-z0-9\-\+\.]*|[\u4e00-\u9fff]{2,}", query)
        deduped = []
        seen = set()
        for chunk in chunks:
            item = chunk.strip()
            lowered = item.lower()
            if not item or lowered in seen or item in CHINESE_STOPWORDS:
                continue
            seen.add(lowered)
            deduped.append(item)
        return deduped[:8]

    def _translate_keywords(self, keywords_zh: list[str], original_query: str) -> list[str]:
        translated = []
        seen = set()
        for keyword in keywords_zh:
            term = self._translate_term(keyword)
            if not term:
                continue
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            translated.append(term)

        if not translated:
            english_tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-\+\.]*", original_query)
            for token in english_tokens:
                lowered = token.lower()
                if lowered not in seen:
                    translated.append(token)
                    seen.add(lowered)

        return translated[:8]

    def _translate_term(self, keyword: str) -> str:
        lowered = keyword.lower()
        if lowered in COMMON_TRANSLATIONS:
            return COMMON_TRANSLATIONS[lowered]
        if keyword in COMMON_TRANSLATIONS:
            return COMMON_TRANSLATIONS[keyword]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-\+\.]*", keyword):
            return keyword
        return keyword

    def _normalize_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        seen = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    def _normalize_required_aspects(self, value: Any, original_query: str, keywords_zh: list[str]) -> list[str]:
        aspects = self._normalize_string_list(value)[:5]
        if aspects:
            return aspects
        return self._infer_required_aspects(original_query, keywords_zh)

    def _infer_required_aspects(self, original_query: str, keywords_zh: list[str]) -> list[str]:
        if len(keywords_zh) >= 2 and COMPARISON_PATTERN.search(original_query or ""):
            left = self._translate_term(keywords_zh[0]) or keywords_zh[0]
            right = self._translate_term(keywords_zh[1]) or keywords_zh[1]
            return [
                f"definition of {left}",
                f"definition of {right}",
                f"differences between {left} and {right}",
                f"advantages of {left} over {right}",
            ][:5]

        base = (original_query or "").strip().rstrip("？?!.。")
        if not base:
            return []
        primary_term = self._translate_term(keywords_zh[0]) if keywords_zh else base
        return [
            f"definition of {primary_term}",
            f"core methods for {primary_term}",
            f"evidence or evaluation for {primary_term}",
        ][:5]
