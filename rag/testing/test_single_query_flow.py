from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import unittest
import uuid
from unittest import mock

from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, ToolMessage

try:
    import fitz
except ModuleNotFoundError:
    fitz = None

try:
    from rag.agent.evidence import annotate_local_documents, build_final_evidence_bundle, select_local_evidence
    from rag.agent.middleware import AcademicResearchMiddleware
    import rag.agent.middleware as agent_middleware_module
    from rag.agent.runtime import context as research_context
    import rag.llm_factory as llm_factory_module
    import rag.rag_system as rag_system_module
    from rag.crawlers.arxiv import ArxivCrawlerIntegrated
    from rag.crawlers.standalone import (
        MissingAspectQueryOptimizer,
        StandaloneMissingAspectCrawler,
        apply_runtime_args,
    )
    import rag.ocr_client as ocr_client_module
    from rag.ocr_client import RemoteOCRClient
    from rag.query.optimizer import AcademicQueryPlanner
    from rag.rag_system import RAGSystem, VLLMOpenAIEmbeddings
    from rag.retrieval.evaluator import evaluate_retrieval
    from rag.schemas import (
        AcademicQueryPlan,
        AspectRewritePayload,
        FinalEvidenceBundle,
        FinalEvidenceItem,
        NormalizedDocument,
        RetrievalPayload,
        WebSearchPayload,
        WebSearchQuery,
    )
    from rag.testing.fixtures import fake_crawl_papers
except ImportError:
    from agent.evidence import annotate_local_documents, build_final_evidence_bundle, select_local_evidence
    from agent.middleware import AcademicResearchMiddleware
    import agent.middleware as agent_middleware_module
    from agent.runtime import context as research_context
    import llm_factory as llm_factory_module
    import rag_system as rag_system_module
    from crawlers.arxiv import ArxivCrawlerIntegrated
    from crawlers.standalone import (
        MissingAspectQueryOptimizer,
        StandaloneMissingAspectCrawler,
        apply_runtime_args,
    )
    import ocr_client as ocr_client_module
    from ocr_client import RemoteOCRClient
    from query.optimizer import AcademicQueryPlanner
    from rag_system import RAGSystem, VLLMOpenAIEmbeddings
    from retrieval.evaluator import evaluate_retrieval
    from schemas import (
        AcademicQueryPlan,
        AspectRewritePayload,
        FinalEvidenceBundle,
        FinalEvidenceItem,
        NormalizedDocument,
        RetrievalPayload,
        WebSearchPayload,
        WebSearchQuery,
    )
    from testing.fixtures import fake_crawl_papers

try:
    from rag.pdf_processor import PDFProcessor
except ImportError:
    try:
        from pdf_processor import PDFProcessor
    except ImportError:
        PDFProcessor = None


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _messages):
        return FakeLLMResponse(self.content)


class FakeOCRClient:
    def __init__(self, page_texts: list[str]):
        self.page_texts = list(page_texts)
        self.calls: list[str] = []

    def extract_from_image_path(self, image_path: str, *, prompt=None, max_tokens=None):
        self.calls.append(image_path)
        text = self.page_texts.pop(0) if self.page_texts else ""
        return {
            "text": text,
            "raw_response": {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
            "finish_reason": "stop",
        }

    def extract_from_image_bytes(self, payload: bytes, *, mime_type="image/png", prompt=None, max_tokens=None):
        self.calls.append(f"bytes:{len(payload)}")
        text = self.page_texts.pop(0) if self.page_texts else ""
        return {
            "text": text,
            "raw_response": {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
            "finish_reason": "stop",
        }


class MiddlewareTool:
    def __init__(self, name: str):
        self.name = name


class DummyRequest:
    def __init__(self, state, tools, system_message=None):
        self.state = state
        self.tools = tools
        self.system_message = system_message

    def override(self, **kwargs):
        return DummyRequest(
            state=kwargs.get("state", self.state),
            tools=kwargs.get("tools", self.tools),
            system_message=kwargs.get("system_message", self.system_message),
        )


class DummyLangChainDoc:
    def __init__(self, content: str, source: str, **metadata):
        self.page_content = content
        self.metadata = {"source": source, **metadata}


class MissingAspectQueryOptimizerTests(unittest.TestCase):
    def test_prompt_invoke_allows_literal_json_example(self):
        optimizer = MissingAspectQueryOptimizer(llm=object())

        prompt_value = optimizer.prompt.invoke({"aspect": "the definition of RNN"})

        self.assertIn('"original_aspect"', prompt_value.messages[1].content)
        self.assertIn('Input: definition of LLM', prompt_value.messages[1].content)
        self.assertIn('"optimized_query_en": "large language models are"', prompt_value.messages[1].content)
        self.assertIn("Aspect: the definition of RNN", prompt_value.messages[1].content)

    def test_rewrite_returns_llm_optimized_query_and_deduped_keywords(self):
        fake_json = """
        {
          "original_aspect": "the definition of RNN",
          "optimized_query_en": "recurrent neural network definition",
          "keywords_en": [
            "recurrent neural network",
            "RNN",
            "recurrent neural network"
          ]
        }
        """
        optimizer = MissingAspectQueryOptimizer(llm=FakeLLM(fake_json))

        payload = optimizer.rewrite("the definition of RNN")

        self.assertEqual(payload.original_aspect, "the definition of RNN")
        self.assertEqual(payload.optimized_query_en, "recurrent neural network definition")
        self.assertEqual(payload.keywords_en, ["recurrent neural network", "RNN"])

    def test_rewrite_falls_back_to_original_aspect_and_warns(self):
        class ExplodingLLM:
            def invoke(self, _messages):
                raise RuntimeError("boom")

        optimizer = MissingAspectQueryOptimizer(llm=ExplodingLLM())

        with mock.patch("builtins.print") as mocked_print:
            payload = optimizer.rewrite("the definition of CNN")

        self.assertEqual(payload.original_aspect, "the definition of CNN")
        self.assertEqual(payload.optimized_query_en, "the definition of CNN")
        self.assertEqual(payload.keywords_en, ["the definition of CNN"])
        mocked_print.assert_called_once()
        warning_message = mocked_print.call_args.args[0]
        self.assertIn("the definition of CNN", warning_message)
        self.assertIn("RuntimeError", warning_message)
        self.assertIn("boom", warning_message)
        self.assertIn("falling back to original aspect", warning_message)


class StandaloneCrawlerRewriteFlowTests(unittest.TestCase):
    def _workspace_tempdir(self, prefix: str) -> str:
        path = os.path.join(os.getcwd(), f".tmp_{prefix}_{uuid.uuid4().hex}")
        os.makedirs(path, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_run_aspects_passes_rewritten_queries_to_arxiv_search(self):
        class StubPayload:
            def model_dump(self):
                return {"status": "success", "message": "ok"}

        class StubCrawler:
            def __init__(self):
                self.last_call = None

            def crawl_and_collect(self, **kwargs):
                self.last_call = kwargs
                return StubPayload(), None

        output_dir = self._workspace_tempdir("standalone_rewrite_output")
        md_dir = self._workspace_tempdir("standalone_rewrite_md")
        queue_path = os.path.join(output_dir, "pending_aspects.json")
        crawler = StandaloneMissingAspectCrawler(
            output_dir=output_dir,
            md_output_dir=md_dir,
            queue_path=queue_path,
            llm=object(),
        )
        crawler.optimizer = mock.Mock()
        crawler.optimizer.rewrite.side_effect = [
            AspectRewritePayload(
                original_aspect="the definition of RNN",
                optimized_query_en="recurrent neural network definition",
                keywords_en=["recurrent neural network", "RNN"],
            ),
            AspectRewritePayload(
                original_aspect="the definition of CNN",
                optimized_query_en="convolutional neural network definition",
                keywords_en=["convolutional neural network", "CNN"],
            ),
        ]
        crawler.crawler = StubCrawler()

        result = crawler.run_aspects(
            ["the definition of RNN", "the definition of CNN"],
            max_pages=1,
            auto_ingest=False,
        )

        self.assertEqual(
            crawler.crawler.last_call["search_query_overrides"],
            {
                "the definition of RNN": "recurrent neural network definition",
                "the definition of CNN": "convolutional neural network definition",
            },
        )
        self.assertEqual(
            crawler.crawler.last_call["query_plan"].crawler_query_en,
            "recurrent neural network definition convolutional neural network definition",
        )
        self.assertEqual(
            result["rewrites"],
            [
                {
                    "original_aspect": "the definition of RNN",
                    "optimized_query_en": "recurrent neural network definition",
                    "keywords_en": ["recurrent neural network", "RNN"],
                },
                {
                    "original_aspect": "the definition of CNN",
                    "optimized_query_en": "convolutional neural network definition",
                    "keywords_en": ["convolutional neural network", "CNN"],
                },
            ],
        )


class StandaloneRuntimeConfigTests(unittest.TestCase):
    def test_apply_runtime_args_sets_shared_ssh_env_for_llm_ocr_and_embedding(self):
        args = argparse.Namespace(
            use_ssh=True,
            ssh_host="172.26.19.131",
            ssh_port=8888,
            ssh_username="root",
            ssh_password="123456.a",
            llm_remote_port=8001,
            llm_local_port=18001,
            embedding_remote_port=8000,
            embedding_local_port=18000,
            ocr_remote_port=8002,
            ocr_local_port=18002,
            llm_base_url="http://stale-llm/v1",
            embedding_base_url="http://stale-embedding/v1",
            ocr_base_url="http://stale-ocr/v1",
        )

        with mock.patch.dict(
            os.environ,
            {
                "RAG_LLM_BASE_URL": "http://old-llm/v1",
                "RAG_EMBEDDING_BASE_URL": "http://old-embedding/v1",
                "RAG_OCR_BASE_URL": "http://old-ocr/v1",
            },
            clear=False,
        ):
            apply_runtime_args(args)

            llm_ssh_config = llm_factory_module.get_default_llm_ssh_config()
            ocr_ssh_config = ocr_client_module.resolve_ocr_ssh_config()
            embedding_ssh_config = RAGSystem()._resolve_embedding_ssh_config()
            self.assertEqual(os.environ["RAG_SSH_HOST"], "172.26.19.131")
            self.assertEqual(os.environ["RAG_SSH_USERNAME"], "root")
            self.assertEqual(os.environ["RAG_LLM_REMOTE_PORT"], "8001")
            self.assertEqual(os.environ["RAG_EMBEDDING_REMOTE_PORT"], "8000")
            self.assertEqual(os.environ["RAG_OCR_REMOTE_PORT"], "8002")
            self.assertEqual(os.environ.get("RAG_LLM_BASE_URL"), None)
            self.assertEqual(os.environ.get("RAG_EMBEDDING_BASE_URL"), None)
            self.assertEqual(os.environ.get("RAG_OCR_BASE_URL"), None)
            self.assertEqual(llm_ssh_config["ssh_host"], "172.26.19.131")
            self.assertEqual(llm_ssh_config["remote_port"], 8001)
            self.assertEqual(ocr_ssh_config["ssh_host"], "172.26.19.131")
            self.assertEqual(ocr_ssh_config["remote_port"], 8002)
            self.assertEqual(embedding_ssh_config["ssh_host"], "172.26.19.131")
            self.assertEqual(embedding_ssh_config["remote_port"], 8000)


class QueryPlannerTests(unittest.TestCase):
    def test_prompt_invoke_allows_literal_json_examples(self):
        planner = AcademicQueryPlanner(llm=object())

        prompt_value = planner.prompt.invoke({"question": "Transformer和RNN有什么区别"})

        self.assertIn('"original_query"', prompt_value.messages[1].content)
        self.assertIn('"required_aspects"', prompt_value.messages[1].content)
        self.assertIn("User question: Transformer和RNN有什么区别", prompt_value.messages[1].content)

    def test_build_query_plan_for_chinese_query(self):
        fake_json = """
        {
          "original_query": "如何优化中文学术RAG检索？",
          "normalized_query_zh": "如何优化中文学术RAG检索与相关性评估？",
          "retrieval_query_zh": "中文 学术 RAG 检索 查询优化 相关性评估",
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


class MiddlewarePromptBudgetTests(unittest.TestCase):
    def setUp(self):
        fake_json = """
        {
          "original_query": "如何比较 fading memory 和 HA-GNN？",
          "normalized_query_zh": "如何比较 fading memory 和 HA-GNN？",
          "retrieval_query_zh": "fading memory HA-GNN 历史访问 比较",
          "retrieval_query_en": "fading memory HA-GNN historical access comparison",
          "crawler_query_en": "fading memory HA-GNN historical access comparison",
          "keywords_zh": ["fading memory", "HA-GNN", "历史访问"],
          "keywords_en": ["fading memory", "HA-GNN", "historical access"],
          "required_aspects": ["fading memory definition", "HA-GNN mechanism"]
        }
        """
        self.middleware = AcademicResearchMiddleware(
            FakeLLM(fake_json),
            retrieve_tool_name="retrieve_local_kb",
            web_search_tool_name="search_web_with_tavily",
        )

    def test_prompt_friendly_final_evidence_keeps_full_content_and_all_items(self):
        long_content = "A" * 4000
        bundle = FinalEvidenceBundle(
            query="compare fading memory and HA-GNN",
            summary="summary",
            local_evidence=[
                FinalEvidenceItem(
                    origin="local_kb",
                    content=long_content,
                    source="local_1",
                    title="Local 1",
                    url="https://example.com/local-1",
                    aspects=["fading memory"],
                )
            ],
            web_evidence=[
                FinalEvidenceItem(
                    origin="tavily_web",
                    content=long_content,
                    source=f"web_{idx}",
                    title=f"Web {idx}",
                    url=f"https://example.com/web-{idx}",
                    aspects=["HA-GNN"],
                )
                for idx in range(1, 8)
            ],
            uncovered_aspects=["difference"],
        )

        compact_text = self.middleware._build_prompt_friendly_final_evidence(bundle)
        compact = json.loads(compact_text)

        self.assertEqual(len(compact["local_evidence"]), 1)
        self.assertEqual(len(compact["web_evidence"]), 7)
        self.assertEqual(compact["local_evidence"][0]["content"], long_content)
        self.assertEqual(compact["web_evidence"][-1]["content"], long_content)
        self.assertIn('"content":', compact_text)
        self.assertNotIn("content_excerpt", compact["local_evidence"][0])

    def test_tool_summaries_stay_compact(self):
        long_content = "B" * 3000
        retrieval_payload = RetrievalPayload(
            status="success",
            message="ok",
            query="fading memory",
            doc_count=5,
            docs=[],
        )
        local_items = [
            FinalEvidenceItem(
                origin="local_kb",
                content=long_content,
                source="local",
                title="Local",
                url="https://example.com/local",
                aspects=["fading memory"],
            )
        ]
        retrieval_summary = self.middleware._build_retrieval_tool_summary(
            payload=retrieval_payload,
            local_evidence=local_items,
            missing_aspects=["mechanism of historical access in HA-GNN"],
            next_action="search_web",
        )

        web_payload = WebSearchPayload(
            status="partial_success",
            message="web ok",
            requested_missing_aspects=["fading memory", "HA-GNN"],
            covered_missing_aspects=["fading memory"],
            uncovered_missing_aspects=["HA-GNN"],
            search_queries=[WebSearchQuery(aspect="fading memory", query="fading memory")],
            results=[],
            evidence_docs=[],
        )
        final_bundle = FinalEvidenceBundle(
            query="compare",
            summary="summary",
            local_evidence=[],
            web_evidence=[
                FinalEvidenceItem(
                    origin="tavily_web",
                    content=long_content,
                    source="web",
                    title="Web",
                    url="https://example.com/web",
                    aspects=["HA-GNN"],
                )
            ],
            uncovered_aspects=["HA-GNN"],
        )
        web_summary = self.middleware._build_web_search_tool_summary(
            payload=web_payload,
            final_bundle=final_bundle,
        )

        self.assertLess(len(retrieval_summary), 2000)
        self.assertLess(len(web_summary), 2000)
        self.assertNotIn(long_content, retrieval_summary)
        self.assertNotIn(long_content, web_summary)


class MiddlewareAutoWebSearchTests(unittest.TestCase):
    def setUp(self):
        fake_json = """
        {
          "original_query": "compare fading memory and HA-GNN",
          "normalized_query_zh": "比较 fading memory 和 HA-GNN",
          "retrieval_query_zh": "衰减记忆 动态系统 HA-GNN 历史访问",
          "retrieval_query_en": "fading memory dynamic systems HA-GNN historical access",
          "crawler_query_en": "fading memory HA-GNN",
          "keywords_zh": ["衰减记忆", "HA-GNN"],
          "keywords_en": ["fading memory", "HA-GNN"],
          "required_aspects": [
            "definition of fading memory in dynamic systems",
            "HA-GNN historical access mechanism"
          ]
        }
        """
        self.middleware = AcademicResearchMiddleware(
            FakeLLM(fake_json),
            retrieve_tool_name="retrieve_local_kb",
            web_search_tool_name="search_web_with_tavily",
        )
        self.plan = AcademicQueryPlan(
            original_query="compare fading memory and HA-GNN",
            normalized_query_zh="比较 fading memory 和 HA-GNN",
            retrieval_query_zh="衰减记忆 动态系统 HA-GNN 历史访问",
            retrieval_query_en="fading memory dynamic systems HA-GNN historical access",
            crawler_query_en="fading memory HA-GNN",
            keywords_zh=["衰减记忆", "HA-GNN"],
            keywords_en=["fading memory", "HA-GNN"],
            required_aspects=[
                "definition of fading memory in dynamic systems",
                "HA-GNN historical access mechanism",
            ],
        )
        research_context.reset()
        self.addCleanup(research_context.reset)

    def _make_retrieval_tool_message(self, docs: list[NormalizedDocument]) -> ToolMessage:
        payload = RetrievalPayload(
            status="success",
            message="ok",
            query="衰减记忆 动态系统 HA-GNN 历史访问",
            doc_count=len(docs),
            docs=docs,
        )
        return ToolMessage(
            content=payload.model_dump_json(),
            tool_call_id="call_retrieve_1",
            name="retrieve_local_kb",
        )

    def test_retrieval_hook_auto_merges_web_evidence_when_search_is_needed(self):
        local_doc = NormalizedDocument(
            content="Local fading memory definition.",
            source="local.md",
            score=0.7,
            title="Local Paper",
            url="https://example.com/local",
            origin="local_kb",
            metadata={"title": "Local Paper", "url": "https://example.com/local", "origin": "local_kb"},
        )
        local_item = FinalEvidenceItem(
            origin="local_kb",
            content=local_doc.content,
            source=local_doc.source,
            title=local_doc.title,
            url=local_doc.url,
            aspects=["definition of fading memory in dynamic systems"],
            score=0.7,
            metadata=local_doc.metadata,
        )
        web_doc = NormalizedDocument(
            content="Web evidence for HA-GNN historical access mechanism.",
            source="web source",
            score=0.9,
            title="Web Paper",
            url="https://example.com/web",
            origin="tavily_web",
            aspects=["HA-GNN historical access mechanism"],
            metadata={"title": "Web Paper", "url": "https://example.com/web", "origin": "tavily_web"},
        )
        web_payload = WebSearchPayload(
            status="success",
            message="ok",
            requested_missing_aspects=["HA-GNN historical access mechanism"],
            covered_missing_aspects=["HA-GNN historical access mechanism"],
            uncovered_missing_aspects=[],
            search_queries=[
                WebSearchQuery(
                    aspect="HA-GNN historical access mechanism",
                    query="HA-GNN historical access mechanism",
                )
            ],
            results=[],
            evidence_docs=[web_doc],
        )
        evaluation = mock.Mock(
            sufficient=False,
            reason="local evidence is insufficient",
            support_strength=0.45,
            aspect_coverage=0.5,
            noise_ratio=0.1,
            missing_aspects=["HA-GNN historical access mechanism"],
            weak_aspects=[],
            scored_docs=[local_doc],
        )
        request = mock.Mock()
        request.state = {
            "query_plan": self.plan.model_dump(),
            "final_evidence": None,
        }
        request.tool_call = {"name": "retrieve_local_kb"}

        with mock.patch.object(agent_middleware_module, "evaluate_retrieval", return_value=evaluation):
            with mock.patch.object(agent_middleware_module, "annotate_local_documents", return_value=[local_doc]):
                with mock.patch.object(agent_middleware_module, "select_local_evidence", return_value=[local_item]):
                    with mock.patch.object(
                        self.middleware,
                        "_run_forced_web_search",
                        return_value=web_payload,
                    ) as mocked_search:
                        result = self.middleware._handle_retrieval_result(
                            request,
                            self._make_retrieval_tool_message([local_doc]),
                        )

        self.assertEqual(result.update["retrieval_next_action"], "answer")
        self.assertFalse(result.update["web_search_required"])
        self.assertTrue(result.update["web_search_used"])
        self.assertEqual(result.update["relevance_missing_aspects"], [])
        self.assertEqual(result.update["web_search_result"]["status"], "success")
        self.assertEqual(len(result.update["final_evidence"]["web_evidence"]), 1)
        self.assertEqual(result.update["final_evidence"]["web_evidence"][0]["origin"], "tavily_web")
        mocked_search.assert_called_once_with(["HA-GNN historical access mechanism"])

    def test_retrieval_hook_skips_web_search_when_local_evidence_is_sufficient(self):
        local_doc = NormalizedDocument(
            content="Local evidence covers all aspects.",
            source="local.md",
            score=0.8,
            title="Local Paper",
            url="https://example.com/local",
            origin="local_kb",
            metadata={"title": "Local Paper", "url": "https://example.com/local", "origin": "local_kb"},
        )
        local_item = FinalEvidenceItem(
            origin="local_kb",
            content=local_doc.content,
            source=local_doc.source,
            title=local_doc.title,
            url=local_doc.url,
            aspects=list(self.plan.required_aspects),
            score=0.8,
            metadata=local_doc.metadata,
        )
        evaluation = mock.Mock(
            sufficient=True,
            reason="local evidence is sufficient",
            support_strength=0.8,
            aspect_coverage=1.0,
            noise_ratio=0.0,
            missing_aspects=[],
            weak_aspects=[],
            scored_docs=[local_doc],
        )
        request = mock.Mock()
        request.state = {
            "query_plan": self.plan.model_dump(),
            "final_evidence": None,
        }
        request.tool_call = {"name": "retrieve_local_kb"}

        with mock.patch.object(agent_middleware_module, "evaluate_retrieval", return_value=evaluation):
            with mock.patch.object(agent_middleware_module, "annotate_local_documents", return_value=[local_doc]):
                with mock.patch.object(agent_middleware_module, "select_local_evidence", return_value=[local_item]):
                    with mock.patch.object(self.middleware, "_run_forced_web_search") as mocked_search:
                        result = self.middleware._handle_retrieval_result(
                            request,
                            self._make_retrieval_tool_message([local_doc]),
                        )

        self.assertEqual(result.update["retrieval_next_action"], "answer")
        self.assertFalse(result.update["web_search_required"])
        self.assertFalse(result.update["web_search_used"])
        self.assertIsNone(result.update["web_search_result"])
        self.assertEqual(result.update["relevance_missing_aspects"], [])
        self.assertEqual(result.update["final_evidence"]["web_evidence"], [])
        mocked_search.assert_not_called()

    def test_retrieval_hook_falls_back_to_weak_aspects_when_missing_aspects_is_empty(self):
        local_doc = NormalizedDocument(
            content="Local evidence is weak but partially relevant.",
            source="local.md",
            score=0.55,
            title="Local Paper",
            url="https://example.com/local",
            origin="local_kb",
            metadata={"title": "Local Paper", "url": "https://example.com/local", "origin": "local_kb"},
        )
        local_item = FinalEvidenceItem(
            origin="local_kb",
            content=local_doc.content,
            source=local_doc.source,
            title=local_doc.title,
            url=local_doc.url,
            aspects=["definition of fading memory in dynamic systems"],
            score=0.55,
            metadata=local_doc.metadata,
        )
        web_payload = WebSearchPayload(
            status="success",
            message="ok",
            requested_missing_aspects=["HA-GNN historical access mechanism"],
            covered_missing_aspects=["HA-GNN historical access mechanism"],
            uncovered_missing_aspects=[],
            search_queries=[
                WebSearchQuery(
                    aspect="HA-GNN historical access mechanism",
                    query="HA-GNN historical access mechanism",
                )
            ],
            results=[],
            evidence_docs=[],
        )
        evaluation = mock.Mock(
            sufficient=False,
            reason="support is weak",
            support_strength=0.35,
            aspect_coverage=1.0,
            noise_ratio=0.3,
            missing_aspects=[],
            weak_aspects=["HA-GNN historical access mechanism"],
            scored_docs=[local_doc],
        )
        request = mock.Mock()
        request.state = {
            "query_plan": self.plan.model_dump(),
            "final_evidence": None,
        }
        request.tool_call = {"name": "retrieve_local_kb"}

        with mock.patch.object(agent_middleware_module, "evaluate_retrieval", return_value=evaluation):
            with mock.patch.object(agent_middleware_module, "annotate_local_documents", return_value=[local_doc]):
                with mock.patch.object(agent_middleware_module, "select_local_evidence", return_value=[local_item]):
                    with mock.patch.object(
                        self.middleware,
                        "_run_forced_web_search",
                        return_value=web_payload,
                    ) as mocked_search:
                        result = self.middleware._handle_retrieval_result(
                            request,
                            self._make_retrieval_tool_message([local_doc]),
                        )

        self.assertEqual(result.update["retrieval_next_action"], "answer")
        self.assertEqual(result.update["relevance_missing_aspects"], [])
        mocked_search.assert_called_once_with(["HA-GNN historical access mechanism"])


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.plan = AcademicQueryPlan(
            original_query="Why does Transformer outperform RNN?",
            normalized_query_zh="为什么 Transformer 优于 RNN？",
            retrieval_query_zh="Transformer RNN 区别 优势 原因",
            retrieval_query_en="why Transformer outperforms RNN",
            crawler_query_en="Transformer RNN differences advantages",
            keywords_zh=["Transformer", "RNN", "区别", "优势"],
            keywords_en=["Transformer", "RNN", "differences", "advantages"],
            required_aspects=[
                "definition of Transformer",
                "definition of RNN",
                "differences between Transformer and RNN",
                "advantages of Transformer over RNN",
            ],
        )

    def test_sufficient_retrieval_skips_crawler(self):
        docs = [
            {
                "content": "Definition of Transformer architecture and self-attention mechanism.",
                "source": "doc_transformer.md",
                "metadata": {"title": "Transformer Definition"},
            },
            {
                "content": "Definition of recurrent neural network (RNN) with sequence modeling basics.",
                "source": "doc_rnn.md",
                "metadata": {"title": "RNN Definition"},
            },
            {
                "content": (
                    "Differences between Transformer and RNN include parallel computation, "
                    "long-range dependency handling, and better training efficiency."
                ),
                "source": "doc_diff.md",
                "metadata": {"title": "Transformer vs RNN"},
            },
            {
                "content": (
                    "Advantages of Transformer over RNN are stronger scalability and "
                    "improved performance in large-scale sequence tasks."
                ),
                "source": "doc_adv.md",
                "metadata": {"title": "Advantages"},
            },
        ]
        result = evaluate_retrieval(self.plan, docs)
        self.assertTrue(result.sufficient)
        self.assertEqual(result.next_action, "answer")
        self.assertGreaterEqual(result.aspect_coverage, 0.75)
        self.assertGreaterEqual(result.support_strength, 0.60)
        self.assertLessEqual(result.noise_ratio, 0.40)
        self.assertGreaterEqual(len(result.covered_aspects), 3)

    def test_combined_support_can_cover_aspect(self):
        plan = self.plan.model_copy(
            update={
                "required_aspects": ["advantages of Transformer over RNN"],
                "keywords_en": ["Transformer", "RNN", "advantages"],
            }
        )
        docs = [
            {
                "content": "Transformer has strong advantages in parallel training and long context.",
                "source": "doc_part_a.md",
                "metadata": {"title": "Transformer strengths"},
            },
            {
                "content": "Compared with RNN, performance improves on sequence benchmarks.",
                "source": "doc_part_b.md",
                "metadata": {"title": "RNN comparison"},
            },
        ]
        result = evaluate_retrieval(plan, docs)
        self.assertIn("advantages of Transformer over RNN", result.covered_aspects)
        self.assertLess(result.support_strength, 0.60)
        self.assertFalse(result.sufficient)

    def test_high_noise_prefers_retrieve_more(self):
        docs = [
            {
                "content": "This paper discusses image compression and video codecs only.",
                "source": "noise_1.md",
                "metadata": {"title": "Compression Survey"},
            },
            {
                "content": "An unrelated benchmark on audio denoising models.",
                "source": "noise_2.md",
                "metadata": {"title": "Audio Denoising"},
            },
        ]
        result = evaluate_retrieval(self.plan, docs)
        self.assertFalse(result.sufficient)
        self.assertEqual(result.next_action, "retrieve_more")
        self.assertGreater(result.noise_ratio, 0.70)
        self.assertGreaterEqual(len(result.missing_aspects), 1)

    def test_legacy_debug_metrics_are_kept(self):
        docs = [
            {
                "content": "Definition of Transformer and RNN.",
                "source": "doc_legacy.md",
                "metadata": {"title": "Legacy Metrics"},
            }
        ]
        result = evaluate_retrieval(self.plan, docs)
        payload = result.model_dump()
        self.assertIn("top1_coverage", payload)
        self.assertIn("avg_top3_coverage", payload)
        self.assertIn("unique_sources", payload)


class ArxivCrawlerTests(unittest.TestCase):
    def _workspace_tempdir(self, prefix: str) -> str:
        path = os.path.join(os.getcwd(), f".tmp_{prefix}_{uuid.uuid4().hex}")
        os.makedirs(path, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

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

    def test_download_papers_writes_pdf_sidecar(self):
        class StubCrawler(ArxivCrawlerIntegrated):
            def download_paper(self, paper_link: str, filepath: str) -> bool:
                with open(filepath, "wb") as file:
                    file.write(b"%PDF-1.4 dummy")
                return True

        output_dir = self._workspace_tempdir("download_sidecar")
        crawler = StubCrawler(output_dir)
        count = crawler.download_papers(
            papers=[{"title": "Sample Paper", "pdf_link": "https://arxiv.org/pdf/2501.00001"}],
            max_downloads=1,
        )

        self.assertEqual(count, 1)
        sidecar_path = os.path.join(output_dir, "Sample Paper.metadata.json")
        with open(sidecar_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        self.assertEqual(payload["title"], "Sample Paper")
        self.assertEqual(payload["url"], "https://arxiv.org/pdf/2501.00001")
        self.assertEqual(payload["pdf_link"], "https://arxiv.org/pdf/2501.00001")
        self.assertEqual(payload["origin"], "local_kb")


class LocalMetadataPropagationTests(unittest.TestCase):
    def _workspace_tempdir(self, prefix: str) -> str:
        path = os.path.join(os.getcwd(), f".tmp_{prefix}_{uuid.uuid4().hex}")
        os.makedirs(path, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _create_pdf(self, path: str, text: str, *, title: str = "") -> None:
        if fitz is None:
            raise unittest.SkipTest("fitz is not available in the current test environment")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        if title:
            doc.set_metadata({"title": title})
        doc.save(path)
        doc.close()

    def test_pdf_processor_writes_md_sidecar_from_csv_fallback(self):
        if PDFProcessor is None:
            self.skipTest("PDFProcessor dependencies are not available in the current test environment")
        pdf_dir = self._workspace_tempdir("pdf_input")
        md_dir = self._workspace_tempdir("md_output")
        pdf_path = os.path.join(pdf_dir, "Sample Paper.pdf")
        self._create_pdf(
            pdf_path,
            "Sample paper content with enough academic context to survive indexing filters.",
        )

        manifest_path = os.path.join(pdf_dir, "paper_result.csv")
        with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["title", "authors", "abstract", "submission_date", "pdf_link"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "title": "Sample Paper",
                    "authors": "Alice",
                    "abstract": "Abstract",
                    "submission_date": "2025-01-01",
                    "pdf_link": "https://arxiv.org/pdf/2501.00001",
                }
            )

        processor = PDFProcessor(
            output_dir=md_dir,
            lang="en",
            dpi=72,
            ocr_client=FakeOCRClient(
                [
                    "Sample Paper<|LOC_1|>\nAbstract\nSample paper content with enough academic context.",
                ]
            ),
        )
        processed_files = processor.process_pdf_folder(pdf_dir)

        self.assertEqual(len(processed_files), 1)
        sidecar_path = os.path.join(md_dir, "Sample Paper.metadata.json")
        with open(sidecar_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        self.assertEqual(payload["title"], "Sample Paper")
        self.assertEqual(payload["url"], "https://arxiv.org/pdf/2501.00001")
        self.assertEqual(payload["pdf_link"], "https://arxiv.org/pdf/2501.00001")
        self.assertEqual(payload["origin"], "local_kb")
        with open(os.path.join(md_dir, "Sample Paper.md"), "r", encoding="utf-8") as file:
            md_content = file.read()
        self.assertIn("Sample Paper", md_content)
        self.assertNotIn("<|LOC_", md_content)

    def test_load_md_documents_merges_sidecar_metadata(self):
        md_dir = self._workspace_tempdir("md_sidecar")
        md_path = os.path.join(md_dir, "paper.md")
        with open(md_path, "w", encoding="utf-8") as file:
            file.write("A local paper chunk.")
        sidecar_path = os.path.join(md_dir, "paper.metadata.json")
        with open(sidecar_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "title": "Paper Title",
                    "url": "https://example.com/paper.pdf",
                    "pdf_link": "https://example.com/paper.pdf",
                    "source_file": "paper.pdf",
                    "origin": "local_kb",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        original_md_folder = rag_system_module.MD_OUTPUT_FOLDER
        rag_system_module.MD_OUTPUT_FOLDER = md_dir
        try:
            rag_system = RAGSystem()
            docs = rag_system.load_md_documents()
        finally:
            rag_system_module.MD_OUTPUT_FOLDER = original_md_folder

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["title"], "Paper Title")
        self.assertEqual(docs[0].metadata["url"], "https://example.com/paper.pdf")
        self.assertEqual(docs[0].metadata["pdf_link"], "https://example.com/paper.pdf")
        self.assertEqual(docs[0].metadata["source"], "Paper Title")
        self.assertEqual(docs[0].metadata["source_file"], "paper.pdf")

    def test_load_md_documents_falls_back_without_sidecar(self):
        md_dir = self._workspace_tempdir("md_plain")
        md_path = os.path.join(md_dir, "plain_doc.md")
        with open(md_path, "w", encoding="utf-8") as file:
            file.write("Plain local content.")

        original_md_folder = rag_system_module.MD_OUTPUT_FOLDER
        rag_system_module.MD_OUTPUT_FOLDER = md_dir
        try:
            rag_system = RAGSystem()
            docs = rag_system.load_md_documents()
        finally:
            rag_system_module.MD_OUTPUT_FOLDER = original_md_folder

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["title"], "plain_doc")
        self.assertEqual(docs[0].metadata["url"], "")
        self.assertEqual(docs[0].metadata["source"], "plain_doc")
        self.assertEqual(docs[0].metadata["source_file"], "plain_doc.md")

    def test_local_final_evidence_keeps_title_and_url(self):
        plan = AcademicQueryPlan(
            original_query="What is local retrieval?",
            normalized_query_zh="本地检索是什么？",
            retrieval_query_zh="本地 检索 定义",
            retrieval_query_en="definition of local retrieval",
            crawler_query_en="local retrieval definition",
            keywords_zh=["本地检索", "定义"],
            keywords_en=["local retrieval", "definition"],
            required_aspects=["definition of local retrieval"],
        )
        docs = [
            NormalizedDocument(
                content="Definition of local retrieval in a paper-backed RAG system.",
                source="sample_paper.md",
                score=None,
                metadata={
                    "title": "Local Retrieval Paper",
                    "url": "https://example.com/local-retrieval.pdf",
                    "pdf_link": "https://example.com/local-retrieval.pdf",
                    "origin": "local_kb",
                },
            )
        ]

        evaluation = evaluate_retrieval(plan, docs)
        annotated_docs = annotate_local_documents(evaluation.scored_docs, evaluation)
        local_evidence = select_local_evidence(annotated_docs)
        final_bundle = build_final_evidence_bundle(
            query=plan.original_query,
            local_evidence=local_evidence,
            web_evidence=[],
            uncovered_aspects=[],
            note="local evidence is sufficient for answering",
        )

        self.assertEqual(len(final_bundle.local_evidence), 1)
        item = final_bundle.local_evidence[0]
        self.assertEqual(item.title, "Local Retrieval Paper")
        self.assertEqual(item.url, "https://example.com/local-retrieval.pdf")
        self.assertEqual(item.origin, "local_kb")
        self.assertEqual(item.aspects, ["definition of local retrieval"])


class RAGChunkingTests(unittest.TestCase):
    def test_semantic_chunking_pre_splits_long_docs_before_embedding(self):
        rag_system = RAGSystem()
        rag_system.embeddings = object()
        long_text = "Recurrent neural networks are sequence models with hidden states. " * 220
        documents = [DummyLangChainDoc(long_text, "sample.md", title="Sample Paper")]
        semantic_input_lengths: list[int] = []

        class FakeSemanticChunker:
            def __init__(self, _embeddings, breakpoint_threshold_type=None, breakpoint_threshold_amount=None):
                self.breakpoint_threshold_type = breakpoint_threshold_type
                self.breakpoint_threshold_amount = breakpoint_threshold_amount

            def split_documents(self, docs):
                semantic_input_lengths.extend(len(doc.page_content) for doc in docs)
                return [DummyLangChainDoc("Short academic chunk for indexing.", "sample.md", title="Sample Paper")]

        class FakeVectorStore:
            def __init__(self):
                self.saved_paths: list[str] = []

            def save_local(self, path: str):
                self.saved_paths.append(path)

        fake_vectorstore = FakeVectorStore()

        with mock.patch.object(rag_system_module, "SemanticChunker", FakeSemanticChunker):
            with mock.patch("langchain_community.vectorstores.FAISS.from_documents", return_value=fake_vectorstore):
                vectorstore = rag_system.setup_vector_store_semantic_arxiv(documents)

        self.assertIs(vectorstore, fake_vectorstore)
        self.assertGreater(len(semantic_input_lengths), 1)
        self.assertTrue(
            all(length <= rag_system_module.SEMANTIC_PRECHUNK_SIZE for length in semantic_input_lengths)
        )
        self.assertEqual(fake_vectorstore.saved_paths, ["./faiss"])

    def test_update_rag_system_initializes_embeddings_when_missing(self):
        rag_system = RAGSystem()
        docs = [DummyLangChainDoc("Academic text for indexing.", "sample.md", title="Sample Paper")]
        fake_embeddings = object()
        fake_vectorstore = object()

        def assign_embeddings(_embedding_model_path=None):
            rag_system.embeddings = fake_embeddings

        with mock.patch.object(rag_system, "setup_embeddings", side_effect=assign_embeddings) as mocked_setup_embeddings:
            with mock.patch.object(rag_system, "load_md_documents", return_value=docs):
                with mock.patch.object(
                    rag_system,
                    "setup_vector_store_semantic_arxiv",
                    return_value=fake_vectorstore,
                ) as mocked_setup_vectorstore:
                    with mock.patch.object(
                        rag_system,
                        "setup_fallback_retriever",
                        return_value="retriever",
                    ) as mocked_setup_retriever:
                        success = rag_system.update_rag_system(chunk_strategy="semantic_arxiv")

        self.assertTrue(success)
        mocked_setup_embeddings.assert_called_once_with()
        mocked_setup_vectorstore.assert_called_once_with(docs)
        mocked_setup_retriever.assert_called_once_with(k=rag_system_module.FINAL_TOP_K)
        self.assertIs(rag_system.embeddings, fake_embeddings)
        self.assertIs(rag_system.vectorstore, fake_vectorstore)
        self.assertEqual(rag_system.retriever, "retriever")


class RemoteOCRClientTests(unittest.TestCase):
    def _mock_response(self, payload):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    def test_extract_from_data_url_retries_when_finish_reason_is_length(self):
        client = RemoteOCRClient(
            base_url="http://127.0.0.1:18002/v1",
            model="paddle-ocr-vl",
            api_key="EMPTY",
            max_tokens=512,
            retry_max_tokens=1024,
        )
        first_payload = {
            "choices": [
                {
                    "message": {"content": "partial text"},
                    "finish_reason": "length",
                }
            ]
        }
        second_payload = {
            "choices": [
                {
                    "message": {"content": "complete text"},
                    "finish_reason": "stop",
                }
            ]
        }
        with mock.patch.object(
            ocr_client_module.requests,
            "post",
            side_effect=[self._mock_response(first_payload), self._mock_response(second_payload)],
        ) as mocked_post:
            result = client.extract_from_data_url("data:image/png;base64,AAA")

        self.assertEqual(result["text"], "complete text")
        self.assertEqual(mocked_post.call_count, 2)
        self.assertEqual(mocked_post.call_args_list[0].kwargs["json"]["max_tokens"], 512)
        self.assertEqual(mocked_post.call_args_list[1].kwargs["json"]["max_tokens"], 1024)
        image_item = mocked_post.call_args_list[0].kwargs["json"]["messages"][1]["content"][1]
        self.assertEqual(image_item["image_url"]["url"], "data:image/png;base64,AAA")


class PDFProcessorOCRCleaningTests(unittest.TestCase):
    def _workspace_tempdir(self, prefix: str) -> str:
        path = os.path.join(os.getcwd(), f".tmp_{prefix}_{uuid.uuid4().hex}")
        os.makedirs(path, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _create_multi_page_pdf(self, path: str, page_count: int) -> None:
        if fitz is None:
            raise unittest.SkipTest("fitz is not available in the current test environment")
        doc = fitz.open()
        for index in range(page_count):
            page = doc.new_page()
            page.insert_text((72, 72), f"Placeholder page {index + 1}")
        doc.save(path)
        doc.close()

    def test_pdf_processor_cleans_remote_ocr_output(self):
        if PDFProcessor is None:
            self.skipTest("PDFProcessor dependencies are not available in the current test environment")

        pdf_dir = self._workspace_tempdir("ocr_pdf_input")
        md_dir = self._workspace_tempdir("ocr_md_output")
        pdf_path = os.path.join(pdf_dir, "OCR Sample.pdf")
        self._create_multi_page_pdf(pdf_path, 3)

        processor = PDFProcessor(
            output_dir=md_dir,
            lang="en",
            dpi=72,
            ocr_client=FakeOCRClient(
                [
                    (
                        "Conference 2025\n"
                        "A C-LSTM Neural Network for Text Classification<|LOC_245|>\n"
                        "Chunting Zhou\\(1\\)\n"
                        "Abstract\n"
                        "Neural network models have been demon-\n"
                        "strated to be capable.\n"
                        "1"
                    ),
                    "Conference 2025\nIntroduction\nThis is page two body text.\n2",
                    "Conference 2025\nConclusion\nThe model works well.\n3",
                ]
            ),
        )

        result = processor.process_pdf(pdf_path)

        self.assertTrue(os.path.exists(result["md_path"]))
        with open(result["md_path"], "r", encoding="utf-8") as file:
            md_content = file.read()

        self.assertNotIn("<|LOC_", md_content)
        self.assertNotIn("Conference 2025", md_content)
        self.assertNotIn("demon-\nstrated", md_content)
        self.assertIn("demonstrated to be capable.", md_content)
        self.assertIn("Chunting Zhou(1)", md_content)

    def test_trim_reference_tail_removes_references_in_last_30_percent_pages(self):
        if PDFProcessor is None:
            self.skipTest("PDFProcessor dependencies are not available in the current test environment")

        processor = PDFProcessor(output_dir="./md", ocr_client=FakeOCRClient([]))
        cleaned_pages = [
            "Intro page 1",
            "Body page 2",
            "Body page 3",
            "Body page 4",
            "Body page 5",
            "Body page 6",
            "Body page 7",
            "Body page 8",
            "Valid final discussion paragraph.\n\nReferences\n\n[1] First ref.\n[2] Second ref.",
            "[3] Third ref on next page.",
        ]

        trimmed = processor._trim_reference_tail(cleaned_pages)

        self.assertEqual(len(trimmed), 9)
        self.assertEqual(trimmed[-1], "Valid final discussion paragraph.")
        self.assertNotIn("References", "\n".join(trimmed))
        self.assertNotIn("[1] First ref.", "\n".join(trimmed))

    def test_trim_reference_tail_supports_case_insensitive_bibliography(self):
        if PDFProcessor is None:
            self.skipTest("PDFProcessor dependencies are not available in the current test environment")

        processor = PDFProcessor(output_dir="./md", ocr_client=FakeOCRClient([]))
        cleaned_pages = [
            "Body page 1",
            "Body page 2",
            "Closing paragraph.\n\nbIbLiOgRaPhY\n\n[1] Ref entry.",
        ]

        trimmed = processor._trim_reference_tail(cleaned_pages)

        self.assertEqual(trimmed, ["Body page 1", "Body page 2", "Closing paragraph."])

    def test_trim_reference_tail_ignores_heading_outside_tail_window(self):
        if PDFProcessor is None:
            self.skipTest("PDFProcessor dependencies are not available in the current test environment")

        processor = PDFProcessor(output_dir="./md", ocr_client=FakeOCRClient([]))
        cleaned_pages = [
            "Intro page 1",
            "Early section.\n\nReferences\n\n[1] This should stay because it is too early.",
            "Body page 3",
            "Body page 4",
            "Body page 5",
            "Body page 6",
            "Body page 7",
            "Body page 8",
            "Body page 9",
            "Body page 10",
        ]

        trimmed = processor._trim_reference_tail(cleaned_pages)

        self.assertEqual(trimmed, cleaned_pages)

    def test_trim_reference_tail_requires_numbered_reference_entries(self):
        if PDFProcessor is None:
            self.skipTest("PDFProcessor dependencies are not available in the current test environment")

        processor = PDFProcessor(output_dir="./md", ocr_client=FakeOCRClient([]))
        cleaned_pages = [
            "Body page 1",
            "Closing paragraph.\n\nReferences\n\nThis section mentions related work but has no numbered entries.",
            "Appendix content without numbered references.",
        ]

        trimmed = processor._trim_reference_tail(cleaned_pages)

        self.assertEqual(trimmed, cleaned_pages)

    def test_process_pdf_folder_strips_reference_tail_and_keeps_metadata(self):
        if PDFProcessor is None:
            self.skipTest("PDFProcessor dependencies are not available in the current test environment")

        pdf_dir = self._workspace_tempdir("ocr_pdf_reference_input")
        md_dir = self._workspace_tempdir("ocr_md_reference_output")
        pdf_path = os.path.join(pdf_dir, "Reference Sample.pdf")
        self._create_multi_page_pdf(pdf_path, 4)

        processor = PDFProcessor(
            output_dir=md_dir,
            lang="en",
            dpi=72,
            ocr_client=FakeOCRClient(
                [
                    "Title\nIntroduction\nBody page one.",
                    "Method\nBody page two.",
                    "Conclusion\nFinal conclusion paragraph.\n\nREFERENCES\n\n[1] First reference.",
                    "[2] Second reference continues on the next page.",
                ]
            ),
        )

        processed_files = processor.process_pdf_folder(pdf_dir)

        self.assertEqual(len(processed_files), 1)
        md_path = os.path.join(md_dir, "Reference Sample.md")
        sidecar_path = os.path.join(md_dir, "Reference Sample.metadata.json")
        self.assertTrue(os.path.exists(sidecar_path))
        with open(md_path, "r", encoding="utf-8") as file:
            md_content = file.read()

        self.assertIn("Final conclusion paragraph.", md_content)
        self.assertNotIn("REFERENCES", md_content)
        self.assertNotIn("[1] First reference.", md_content)
        self.assertNotIn("## Page 4", md_content)


class StandaloneCrawlerIngestionTests(unittest.TestCase):
    def _workspace_tempdir(self, prefix: str) -> str:
        path = os.path.join(os.getcwd(), f".tmp_{prefix}_{uuid.uuid4().hex}")
        os.makedirs(path, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_execute_ingestion_job_dedupes_and_rebuilds_once(self):
        class StubCrawler(ArxivCrawlerIntegrated):
            def download_paper(self, paper_link: str, filepath: str) -> bool:
                with open(filepath, "wb") as file:
                    file.write(b"%PDF-1.4 dummy")
                return True

        class FakePDFProcessorForIngest:
            def __init__(self, output_dir: str):
                self.output_dir = output_dir
                self.processed: list[str] = []

            def process_pdf(self, pdf_path: str):
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                md_path = os.path.join(self.output_dir, f"{base_name}.md")
                with open(md_path, "w", encoding="utf-8") as file:
                    file.write(f"OCR content for {base_name}")
                sidecar_path = os.path.join(self.output_dir, f"{base_name}.metadata.json")
                with open(sidecar_path, "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "title": base_name,
                            "url": f"https://example.com/{base_name}.pdf",
                            "pdf_link": f"https://example.com/{base_name}.pdf",
                            "source_file": f"{base_name}.pdf",
                            "origin": "local_kb",
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.processed.append(pdf_path)
                return {"md_path": md_path}

        class FakeRAGSystem:
            def __init__(self):
                self.embeddings = object()
                self.update_calls = 0
                self.chunk_strategy = ""

            def setup_embeddings(self):
                self.embeddings = object()

            def update_rag_system(self, chunk_strategy="semantic_arxiv"):
                self.update_calls += 1
                self.chunk_strategy = chunk_strategy
                return True

        output_dir = self._workspace_tempdir("crawler_output")
        md_dir = self._workspace_tempdir("crawler_md")

        with open(os.path.join(output_dir, "Existing Download.metadata.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "title": "Existing Download",
                    "url": "https://arxiv.org/pdf/existing-download.pdf",
                    "pdf_link": "https://arxiv.org/pdf/existing-download.pdf",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        with open(os.path.join(md_dir, "existing-md.metadata.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "title": "Existing MD",
                    "url": "https://arxiv.org/pdf/existing-md.pdf",
                    "pdf_link": "https://arxiv.org/pdf/existing-md.pdf",
                    "source_file": "existing-md.pdf",
                    "origin": "local_kb",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        ingestion_job = {
            "all_papers": [],
            "selected_papers": [],
            "ingest_papers": [
                {"title": "Existing Download", "pdf_link": "https://arxiv.org/pdf/existing-download.pdf"},
                {"title": "Existing MD", "pdf_link": "https://arxiv.org/pdf/existing-md.pdf"},
                {"title": "New Paper 1", "pdf_link": "https://arxiv.org/pdf/new-paper-1.pdf"},
                {"title": "New Paper 2", "pdf_link": "https://arxiv.org/pdf/new-paper-2.pdf"},
                {"title": "New Paper 3", "pdf_link": "https://arxiv.org/pdf/new-paper-3.pdf"},
                {"title": "New Paper 4", "pdf_link": "https://arxiv.org/pdf/new-paper-4.pdf"},
                {"title": "New Paper 5", "pdf_link": "https://arxiv.org/pdf/new-paper-5.pdf"},
                {"title": "New Paper 6", "pdf_link": "https://arxiv.org/pdf/new-paper-6.pdf"},
            ],
            "manifest_csv": "paper_result.csv",
            "manifest_txt": "formatted_papers.txt",
        }

        fake_processor = FakePDFProcessorForIngest(md_dir)
        fake_rag_system = FakeRAGSystem()
        crawler = StubCrawler(output_dir)
        result = crawler.execute_ingestion_job(
            ingestion_job,
            md_output_dir=md_dir,
            max_new_papers=5,
            pdf_processor=fake_processor,
            rag_system=fake_rag_system,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["selected_new_paper_count"], 5)
        self.assertEqual(result["skipped_duplicate_count"], 2)
        self.assertEqual(result["download_success_count"], 5)
        self.assertEqual(result["ocr_success_count"], 5)
        self.assertEqual(result["md_written_count"], 5)
        self.assertTrue(result["rebuild_success"])
        self.assertEqual(fake_rag_system.update_calls, 1)
        self.assertEqual(fake_rag_system.chunk_strategy, "semantic_arxiv")
        self.assertEqual(len(fake_processor.processed), 5)

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
          "required_aspects": ["RAG definition", "retrieval workflow"]
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

    def test_wrap_model_call_allows_retrieve_more_once(self):
        request = DummyRequest(
            state={
                "query_plan": {},
                "retrieval_result": {"status": "success"},
                "retrieval_sufficient": False,
                "retrieval_next_action": "retrieve_more",
                "retrieval_retry_count": 0,
                "crawl_required": False,
                "crawl_used": False,
                "messages": [],
            },
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
                "retrieval_sufficient": False,
                "retrieval_next_action": "crawl_more",
                "retrieval_retry_count": 1,
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
            original_query="How to improve academic retrieval?",
            normalized_query_zh="如何优化学术检索？",
            retrieval_query_zh="中文 检索 查询",
            retrieval_query_en="academic retrieval query",
            crawler_query_en="academic retrieval query",
            keywords_zh=["学术检索", "BM25"],
            keywords_en=["academic retrieval", "BM25"],
            required_aspects=["definition of academic retrieval"],
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


class EmbeddingCompatibilityTests(unittest.TestCase):
    def test_vllm_openai_embeddings_implements_langchain_embeddings_interface(self):
        embeddings = VLLMOpenAIEmbeddings(base_url="http://127.0.0.1:18000/v1", model="bge-m3")

        self.assertIsInstance(embeddings, Embeddings)

    def test_vllm_openai_embeddings_callable_delegates_to_embed_query(self):
        embeddings = VLLMOpenAIEmbeddings(base_url="http://127.0.0.1:18000/v1", model="bge-m3")

        with mock.patch.object(embeddings, "embed_query", return_value=[0.1, 0.2]) as mocked_embed_query:
            result = embeddings("diagnostic query")

        self.assertEqual(result, [0.1, 0.2])
        mocked_embed_query.assert_called_once_with("diagnostic query")


if __name__ == "__main__":
    unittest.main()
