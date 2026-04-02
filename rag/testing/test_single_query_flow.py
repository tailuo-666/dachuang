from __future__ import annotations

import unittest

from langchain_core.messages import SystemMessage

try:
    from rag.agent.middleware import AcademicResearchMiddleware
    from rag.crawlers.arxiv import ArxivCrawlerIntegrated
    from rag.query.optimizer import AcademicQueryPlanner
    from rag.retrieval.evaluator import evaluate_retrieval
    from rag.schemas import AcademicQueryPlan
    from rag.testing.fixtures import fake_crawl_papers, retrieval_docs_high_match, retrieval_docs_low_match
except ImportError:
    from agent.middleware import AcademicResearchMiddleware
    from crawlers.arxiv import ArxivCrawlerIntegrated
    from query.optimizer import AcademicQueryPlanner
    from retrieval.evaluator import evaluate_retrieval
    from schemas import AcademicQueryPlan
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
          "term_replacements": [
            {
              "original_term": "RAG",
              "normalized_term_zh": "检索增强生成",
              "academic_term_en": "retrieval augmented generation"
            }
          ]
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
          "term_replacements": []
        }
        """
        planner = AcademicQueryPlanner(FakeLLM(fake_json))
        plan = planner.build("RAG query rewriting 怎么提升 academic retrieval？")
        self.assertIn("query rewriting", plan.keywords_en)
        self.assertIn("academic retrieval", plan.crawler_query_en)


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
            term_replacements=[],
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
          "keywords_en": ["academic RAG", "retrieval", "query optimization"],
          "term_replacements": []
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


if __name__ == "__main__":
    unittest.main()
