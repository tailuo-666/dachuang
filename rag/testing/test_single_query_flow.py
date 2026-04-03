from __future__ import annotations

import unittest

from langchain_core.messages import SystemMessage

try:
    from rag.agent.middleware import AcademicResearchMiddleware
    from rag.crawlers.arxiv import ArxivCrawlerIntegrated
    from rag.query.optimizer import AcademicQueryPlanner
    from rag.rag_system import RAGSystem
    from rag.retrieval.evaluator import evaluate_retrieval
    from rag.schemas import AcademicQueryPlan, NormalizedDocument
    from rag.testing.fixtures import fake_crawl_papers, retrieval_docs_high_match, retrieval_docs_low_match
except ImportError:
    from agent.middleware import AcademicResearchMiddleware
    from crawlers.arxiv import ArxivCrawlerIntegrated
    from query.optimizer import AcademicQueryPlanner
    from rag_system import RAGSystem
    from retrieval.evaluator import evaluate_retrieval
    from schemas import AcademicQueryPlan, NormalizedDocument
    from testing.fixtures import fake_crawl_papers, retrieval_docs_high_match, retrieval_docs_low_match


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _messages):
        return FakeLLMResponse(self.content)


class MiddlewareTool:
    def __init__(self, name: str):
        self.name = name


class DummyRequest:
    def __init__(self, state, tools, system_message=None):
        self.state = state
        self.tools = tools
        self.system_message = system_message

    def override(self, **kwargs):
        cloned = DummyRequest(
            state=kwargs.get("state", self.state),
            tools=kwargs.get("tools", self.tools),
            system_message=kwargs.get("system_message", self.system_message),
        )
        return cloned


class DummyLangChainDoc:
    def __init__(self, content: str, source: str, **metadata):
        self.page_content = content
        self.metadata = {"source": source, **metadata}


class QueryPlannerTests(unittest.TestCase):
    def test_build_query_plan_for_chinese_query(self):
        fake_json = """
        {
          "original_query": "如何优化中文学术RAG检索？",
          "normalized_query_zh": "如何优化中文学术 RAG 检索与相关性评估？",
          "retrieval_query_zh": "中文学术 RAG 检索 查询优化 相关性评估",
          "retrieval_query_en": "optimize Chinese academic RAG retrieval relevance evaluation",
          "crawler_query_en": "Chinese academic RAG retrieval query optimization relevance evaluation",
          "keywords_zh": ["中文学术RAG", "检索", "相关性评估"],
          "keywords_en": ["Chinese academic RAG", "retrieval", "relevance evaluation"],
          "required_aspects": ["RAG定义", "检索流程", "相关性评估方法"]
        }
        """
        planner = AcademicQueryPlanner(FakeLLM(fake_json))
        plan = planner.build("如何优化中文学术RAG检索？")
        self.assertEqual(plan.original_query, "如何优化中文学术RAG检索？")
        self.assertIn("retrieval", plan.retrieval_query_en)
        self.assertGreaterEqual(len(plan.keywords_en), 2)

    def test_build_query_plan_for_mixed_query(self):
        fake_json = """
        {
          "original_query": "RAG query rewriting 怎么提升 academic retrieval？",
          "normalized_query_zh": "RAG query rewriting 如何提升 academic retrieval 质量？",
          "retrieval_query_zh": "RAG query rewriting academic retrieval 质量 提升",
          "retrieval_query_en": "RAG query rewriting for improving academic retrieval quality",
          "crawler_query_en": "RAG query rewriting academic retrieval quality",
          "keywords_zh": ["RAG", "query rewriting", "academic retrieval"],
          "keywords_en": ["RAG", "query rewriting", "academic retrieval"],
          "required_aspects": ["RAG是什么", "query rewriting方法", "academic retrieval评估"]
        }
        """
        planner = AcademicQueryPlanner(FakeLLM(fake_json))
        plan = planner.build("RAG query rewriting 怎么提升 academic retrieval？")
        self.assertIn("query rewriting", plan.keywords_en)
        self.assertIn("academic retrieval", plan.crawler_query_en)

    def test_required_aspects_are_deduped_and_capped(self):
        fake_json = """
        {
          "original_query": "Transformer 为什么比 RNN 好？",
          "normalized_query_zh": "Transformer 为什么比 RNN 更好？",
          "retrieval_query_zh": "Transformer RNN 区别 优势 原因",
          "retrieval_query_en": "why Transformer outperforms RNN",
          "crawler_query_en": "Transformer RNN differences advantages",
          "keywords_zh": ["Transformer", "RNN", "区别", "优势"],
          "keywords_en": ["Transformer", "RNN", "differences", "advantages"],
          "required_aspects": [
            "Transformer是什么",
            "RNN是什么",
            "两者区别",
            "Transformer更好的原因",
            "实验结果证据",
            "Transformer是什么"
          ]
        }
        """
        planner = AcademicQueryPlanner(FakeLLM(fake_json))
        plan = planner.build("Transformer 为什么比 RNN 好？")
        self.assertEqual(len(plan.required_aspects), 5)
        self.assertEqual(plan.required_aspects[0], "Transformer是什么")


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.plan = AcademicQueryPlan(
            original_query="如何优化学术RAG检索？",
            normalized_query_zh="如何优化学术RAG检索？",
            retrieval_query_zh="学术 RAG 检索 查询优化 相关性评估",
            retrieval_query_en="academic RAG retrieval query optimization relevance evaluation",
            crawler_query_en="academic RAG retrieval query optimization relevance evaluation",
            keywords_zh=["学术RAG", "检索", "查询优化", "相关性评估"],
            keywords_en=["academic RAG", "retrieval", "query optimization", "relevance evaluation"],
        )

    def test_sufficient_retrieval_skips_crawler(self):
        result = evaluate_retrieval(self.plan, retrieval_docs_high_match())
        self.assertTrue(result.sufficient)
        self.assertGreaterEqual(result.top1_coverage, 0.60)

    def test_insufficient_retrieval_requires_crawler(self):
        result = evaluate_retrieval(self.plan, retrieval_docs_low_match())
        self.assertFalse(result.sufficient)
        self.assertLess(result.top1_coverage, 0.60)


class ArxivCrawlerTests(unittest.TestCase):
    def test_search_query_uses_keywords(self):
        crawler = ArxivCrawlerIntegrated("./paper_results")
        query = crawler.generate_search_query(
            query_en="academic retrieval query optimization",
            keywords_en=["academic retrieval", "query optimization", "RAG"],
        )
        self.assertIn('"academic retrieval"', query)
        self.assertIn('"query optimization"', query)

    def test_evidence_docs_exist_without_pdf_processing(self):
        crawler = ArxivCrawlerIntegrated("./paper_results")
        docs = crawler.papers_to_evidence_docs(fake_crawl_papers())
        self.assertEqual(len(docs), 1)
        self.assertIn("Abstract:", docs[0].content)


class MiddlewareFilteringTests(unittest.TestCase):
    def setUp(self):
        fake_json = """
        {
          "original_query": "如何优化学术RAG检索？",
          "normalized_query_zh": "如何优化学术RAG检索？",
          "retrieval_query_zh": "学术 RAG 检索 查询优化",
          "retrieval_query_en": "academic RAG retrieval query optimization",
          "crawler_query_en": "academic RAG retrieval query optimization",
          "keywords_zh": ["学术RAG", "检索", "查询优化"],
          "keywords_en": ["academic RAG", "retrieval", "query optimization"]
        }
        """
        self.middleware = AcademicResearchMiddleware(
            FakeLLM(fake_json),
            retrieve_tool_name="retrieve_local_kb",
            crawl_tool_name="crawl_academic_sources",
        )
        self.tools = [MiddlewareTool("retrieve_local_kb"), MiddlewareTool("crawl_academic_sources")]

    def test_wrap_model_call_initially_only_exposes_retriever(self):
        request = DummyRequest(
            state={"query_plan": {}, "messages": []},
            tools=self.tools,
            system_message=SystemMessage(content="base"),
        )

        def handler(inner_request):
            return inner_request

        result = self.middleware.wrap_model_call(request, handler)
        self.assertEqual([tool.name for tool in result.tools], ["retrieve_local_kb"])

    def test_wrap_model_call_enables_crawler_when_needed(self):
        request = DummyRequest(
            state={
                "query_plan": {},
                "retrieval_result": {"status": "success"},
                "crawl_required": True,
                "crawl_used": False,
                "messages": [],
            },
            tools=self.tools,
            system_message=SystemMessage(content="base"),
        )

        def handler(inner_request):
            return inner_request

        result = self.middleware.wrap_model_call(request, handler)
        self.assertEqual([tool.name for tool in result.tools], ["crawl_academic_sources"])


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.plan = AcademicQueryPlan(
            original_query="如何优化学术RAG检索？",
            normalized_query_zh="如何优化学术RAG检索？",
            retrieval_query_zh="中文 检索 查询",
            retrieval_query_en="academic retrieval query",
            crawler_query_en="academic retrieval query",
            keywords_zh=["学术RAG", "检索"],
            keywords_en=["academic retrieval", "BM25"],
        )

    def test_bm25_query_uses_english_query_and_keywords_only(self):
        rag_system = RAGSystem()
        plan = self.plan.model_copy(
            update={
                "retrieval_query_zh": "中文 检索 查询",
                "retrieval_query_en": "academic retrieval",
                "keywords_en": ["academic retrieval", "BM25", "BM25"],
            }
        )
        self.assertEqual(rag_system._build_bm25_query(plan), "academic retrieval BM25")

    def test_rrf_fusion_merges_duplicate_documents(self):
        rag_system = RAGSystem()
        shared_doc = NormalizedDocument(
            content="shared content",
            source="paper_a.md",
            score=None,
            metadata={"title": "Shared"},
        )
        branch_results = {
            "bm25_en": [shared_doc],
            "dense_zh": [shared_doc],
            "dense_en": [
                NormalizedDocument(
                    content="another content",
                    source="paper_b.md",
                    score=None,
                    metadata={"title": "Another"},
                )
            ],
        }

        docs, debug = rag_system._fuse_with_rrf(branch_results, final_top_k=5)
        self.assertEqual(len(docs), 2)
        self.assertEqual(debug["rrf_pool_count"], 2)
        self.assertEqual(docs[0].source, "paper_a.md")
        self.assertIn("retrieval_debug", docs[0].metadata)
        self.assertIn("bm25_en", docs[0].metadata["retrieval_debug"]["branch_hits"])
        self.assertIn("dense_zh", docs[0].metadata["retrieval_debug"]["branch_hits"])

    def test_retrieve_with_query_plan_returns_top5_after_rrf(self):
        class StubRAGSystem(RAGSystem):
            def _run_bm25_branch(self, query_plan):
                return "bm25 english query", [
                    self._normalize_langchain_doc(DummyLangChainDoc(f"bm25 content {idx}", f"bm25_{idx}.md"))
                    for idx in range(1, 16)
                ]

            def _run_dense_branch(self, query, top_k=15):
                prefix = "zh" if query == "中文 检索 查询" else "en"
                return [
                    self._normalize_langchain_doc(DummyLangChainDoc(f"{prefix} content {idx}", f"{prefix}_{idx}.md"))
                    for idx in range(1, top_k + 1)
                ]

        rag_system = StubRAGSystem()
        docs, debug = rag_system.retrieve_with_query_plan(self.plan, final_top_k=5)
        self.assertEqual(len(docs), 5)
        self.assertEqual(debug["branch_counts"]["bm25_en"], 15)
        self.assertEqual(debug["branch_counts"]["dense_zh"], 15)
        self.assertEqual(debug["branch_counts"]["dense_en"], 15)
        self.assertEqual(debug["rrf_pool_count"], 20)


if __name__ == "__main__":
    unittest.main()
