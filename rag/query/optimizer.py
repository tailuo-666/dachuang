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
    "召回率": "recall",
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
    "一下",
    "以及",
    "对于",
    "问题",
    "研究",
    "进行",
    "有关",
    "一个",
    "这个",
    "那个",
    "并且",
    "或者",
    "可以",
    "需要",
    "场景",
    "部分",
}


class AcademicQueryPlanner:
    """Build a single-query academic retrieval plan."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "你是学术研究场景下的查询优化专家。"
                        "你的任务是把用户的中文或中英混合问题，转成适合本地知识库检索与 arXiv 英文检索的单一查询计划。"
                        "你必须理解学术术语、英文固定搭配、常见缩写和中英混合表达，优先使用学术界常见且稳定的术语表达。"
                        "不要拆分子问题，不要输出解释，只返回 JSON。"
                    ),
                ),
                (
                    "human",
                    """
请基于下面的用户问题，输出严格 JSON，对应字段必须完整：
{{
  "original_query": "...",
  "normalized_query_zh": "...",
  "retrieval_query_zh": "...",
  "retrieval_query_en": "...",
  "crawler_query_en": "...",
  "keywords_zh": ["..."],
  "keywords_en": ["..."],
  "required_aspects": ["..."]
}}

规则：
1. 这是学术研究查询，不要做口语化总结。
2. normalized_query_zh 是术语规范化后的中文问题。
3. retrieval_query_zh 用于本地中文知识库检索，要更精准但仍保持自然。
4. retrieval_query_en 用于英文语义检索，要忠实表达原问题语义，不做逐词直译。
5. crawler_query_en 用于 arXiv 标题/摘要搜索，要更像学术搜索串，优先核心术语和研究对象。
6. keywords_zh 与 keywords_en 要去重、按重要性排序。
7. required_aspects 是“本次检索应该覆盖的文档角度”，必须是英文短语列表，最多 5 个，按重要性排序。
8. required_aspects 要可检索、可判断，不要写成空泛目标；优先“定义/机制/差异/原因/证据”这类角度。
9. 对于比较类问题（如 A 为什么比 B 好），required_aspects 优先包含：
   - A 是什么
   - B 是什么
   - A 与 B 的区别
   - A 更好的原因
10. 提取关键词时优先保留“术语短语”而不是拆成零散单词。例如：
   - 机器学习 -> machine learning，不能拆成 machine 和 learning
   - 深度学习 -> deep learning
   - 强化学习 -> reinforcement learning
   - 知识图谱 -> knowledge graph
   - 信息检索 -> information retrieval
   - 查询重写 -> query rewriting
   - 大语言模型 -> large language model
11. 如果用户问题里已经出现标准英文术语或公认缩写，优先保留并复用，例如 RAG、LLM、BM25、OCR、arXiv，不要改写成生造表达。
12. 严禁机械翻译、按字面误译、按近音误译。像“机器学习”绝不能写成 mechanic learning；拿不准时，保留公认英文术语或原缩写，也不要发明新词。
13. keywords_en 必须尽量输出可检索的学术短语，不要变成单词袋。优先输出 machine learning、query rewriting、academic retrieval、question answering 这种短语。
14. crawler_query_en 比 retrieval_query_en 更短、更像搜索串，尽量去掉 how, improve, use, study 这类弱信息词，保留研究对象、任务、方法和约束。
15. 如果用户原问题已经足够精准，也仍然要给出英文查询。
16. 不允许输出 markdown 代码块，只允许输出 JSON 对象。

Few-shot 示例 1：
用户问题：
机器学习如何用于学术论文推荐？

输出：
{{
  "original_query": "机器学习如何用于学术论文推荐？",
  "normalized_query_zh": "机器学习如何用于学术论文推荐与推荐系统优化？",
  "retrieval_query_zh": "机器学习 学术论文推荐 推荐系统",
  "retrieval_query_en": "how machine learning can be used for academic paper recommendation",
  "crawler_query_en": "machine learning academic paper recommendation recommender systems",
  "keywords_zh": ["机器学习", "学术论文推荐", "推荐系统"],
  "keywords_en": ["machine learning", "academic paper recommendation", "recommender systems"],
  "required_aspects": ["机器学习是什么", "学术论文推荐任务定义", "机器学习用于论文推荐的方法", "推荐效果评估指标"]
}}

Few-shot 示例 2：
用户问题：
Transformer 为什么比 RNN 好？

输出：
{{
  "original_query": "Transformer 为什么比 RNN 好？",
  "normalized_query_zh": "Transformer 相比 RNN 的优势来源是什么？",
  "retrieval_query_zh": "Transformer RNN 区别 优势 原因",
  "retrieval_query_en": "why Transformer outperforms RNN",
  "crawler_query_en": "Transformer RNN differences advantages",
  "keywords_zh": ["Transformer", "RNN", "区别", "优势"],
  "keywords_en": ["Transformer", "RNN", "differences", "advantages"],
  "required_aspects": ["Transformer是什么", "RNN是什么", "两者区别", "Transformer更好的原因"]
}}

Few-shot 示例 3：
用户问题：
知识图谱和大语言模型结合做问答有什么方法？

输出：
{{
  "original_query": "知识图谱和大语言模型结合做问答有什么方法？",
  "normalized_query_zh": "知识图谱与大语言模型结合进行问答的方法有哪些？",
  "retrieval_query_zh": "知识图谱 大语言模型 问答 方法",
  "retrieval_query_en": "methods for combining knowledge graph and large language model for question answering",
  "crawler_query_en": "knowledge graph large language model question answering",
  "keywords_zh": ["知识图谱", "大语言模型", "问答"],
  "keywords_en": ["knowledge graph", "large language model", "question answering"],
  "required_aspects": ["知识图谱是什么", "大语言模型是什么", "知识图谱与大语言模型结合方式", "问答任务中的优势与局限"]
}}

用户问题：
{question}
                    """.strip(),
                ),
            ]
        )
        return prompt

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
        compare_pattern = re.compile(r"(为什么|为何|区别|对比|比较|优于|更好|优势|vs|VS|比)")
        if len(keywords_zh) >= 2 and compare_pattern.search(original_query or ""):
            left = keywords_zh[0]
            right = keywords_zh[1]
            return [
                f"{left}是什么",
                f"{right}是什么",
                f"{left}与{right}的区别",
                f"{left}更好的原因",
            ][:5]

        base = (original_query or "").strip().rstrip("？?!！。")
        if not base:
            return []
        return [
            f"{base}的核心定义",
            f"{base}的关键方法",
            f"{base}的评估指标",
        ][:5]
