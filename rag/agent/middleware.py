from __future__ import annotations

import json
from typing import Any, Callable, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

try:
    from ..query.optimizer import AcademicQueryPlanner
    from ..retrieval.evaluator import evaluate_retrieval
    from ..schemas import AcademicQueryPlan, CrawlPayload, ResearchState, RetrievalPayload
    from .runtime import context, log_progress
except ImportError:
    from query.optimizer import AcademicQueryPlanner
    from retrieval.evaluator import evaluate_retrieval
    from schemas import AcademicQueryPlan, CrawlPayload, ResearchState, RetrievalPayload
    from agent.runtime import context, log_progress


class AcademicResearchMiddleware(AgentMiddleware[ResearchState, Any]):
    """Plan the query before the agent starts and enforce retrieval-first execution."""

    state_schema = ResearchState

    def __init__(self, llm, retrieve_tool_name: str, crawl_tool_name: str) -> None:
        super().__init__()
        self.planner = AcademicQueryPlanner(llm)
        self.retrieve_tool_name = retrieve_tool_name
        self.crawl_tool_name = crawl_tool_name

    def before_agent(self, state: ResearchState, runtime) -> dict[str, Any] | None:
        original_query = self._extract_latest_user_query(state)
        plan = self.planner.build(original_query)
        context.reset()
        context.original_query = original_query
        context.query_plan = plan.model_dump()
        log_progress("已完成查询预处理，生成单查询检索计划。")
        return {
            "query_plan": plan.model_dump(),
            "retrieval_result": None,
            "retrieval_sufficient": None,
            "relevance_score": None,
            "relevance_reason": None,
            "crawl_required": False,
            "crawl_used": False,
            "final_sources": [],
        }

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        query_plan_raw = request.state.get("query_plan") or {}
        query_plan = AcademicQueryPlan(**query_plan_raw) if query_plan_raw else None
        retrieval_result = request.state.get("retrieval_result")
        retrieval_sufficient = request.state.get("retrieval_sufficient")
        crawl_required = request.state.get("crawl_required", False)
        crawl_used = request.state.get("crawl_used", False)

        filtered_tools = self._filter_tools(request.tools, retrieval_result, crawl_required, crawl_used)
        query_plan_text = self._build_query_plan_hint(
            query_plan=query_plan,
            retrieval_sufficient=retrieval_sufficient,
            relevance_reason=request.state.get("relevance_reason"),
        )
        system_message = self._append_system_message(request.system_message, query_plan_text)

        return handler(request.override(system_message=system_message, tools=filtered_tools))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if isinstance(result, Command):
            return result
        if not isinstance(result, ToolMessage):
            return result

        tool_name = request.tool_call.get("name", "")
        if tool_name == self.retrieve_tool_name:
            return self._handle_retrieval_result(request, result)
        if tool_name == self.crawl_tool_name:
            return self._handle_crawl_result(result)
        return result

    def _handle_retrieval_result(self, request: ToolCallRequest, result: ToolMessage) -> Command | ToolMessage:
        try:
            payload = RetrievalPayload(**json.loads(str(result.content)))
            query_plan = AcademicQueryPlan(**(request.state.get("query_plan") or {}))
            evaluation = evaluate_retrieval(query_plan, payload.docs)
            payload = payload.model_copy(update={"docs": evaluation.scored_docs})

            context.retrieval_result = payload.model_dump()
            if evaluation.sufficient:
                context.set_sources([doc.model_dump() for doc in evaluation.scored_docs])

            updated_message = ToolMessage(
                content=payload.model_dump_json(),
                tool_call_id=result.tool_call_id,
                name=result.name,
                status=result.status,
            )
            log_progress(
                "本地检索结果相关性评估完成："
                + ("足够，跳过爬虫。" if evaluation.sufficient else "不足，需要继续爬虫。")
            )
            return Command(
                update={
                    "retrieval_result": payload.model_dump(),
                    "retrieval_sufficient": evaluation.sufficient,
                    "relevance_score": evaluation.top1_coverage,
                    "relevance_reason": evaluation.reason,
                    "crawl_required": not evaluation.sufficient,
                    "final_sources": [doc.model_dump() for doc in evaluation.scored_docs] if evaluation.sufficient else [],
                    "messages": [updated_message],
                }
            )
        except Exception as exc:
            log_progress(f"检索结果 hook 解析失败，保留原始结果继续执行: {exc}")
            return result

    def _handle_crawl_result(self, result: ToolMessage) -> Command | ToolMessage:
        try:
            payload = CrawlPayload(**json.loads(str(result.content)))
            context.crawl_result = payload.model_dump()
            context.set_sources([doc.model_dump() for doc in payload.evidence_docs])
            updated_message = ToolMessage(
                content=payload.model_dump_json(),
                tool_call_id=result.tool_call_id,
                name=result.name,
                status=result.status,
            )
            log_progress("学术爬虫阶段完成，开始基于补充证据生成最终答案。")
            return Command(
                update={
                    "crawl_used": True,
                    "crawl_required": False,
                    "final_sources": [doc.model_dump() for doc in payload.evidence_docs],
                    "messages": [updated_message],
                }
            )
        except Exception as exc:
            log_progress(f"爬虫结果 hook 解析失败，保留原始结果继续执行: {exc}")
            return result

    def _filter_tools(
        self,
        tools: list[Any],
        retrieval_result: dict[str, Any] | None,
        crawl_required: bool,
        crawl_used: bool,
    ) -> list[Any]:
        if not retrieval_result:
            allowed = {self.retrieve_tool_name}
        elif crawl_required and not crawl_used:
            allowed = {self.crawl_tool_name}
        else:
            allowed = set()
        return [tool for tool in tools if getattr(tool, "name", None) in allowed]

    def _append_system_message(self, system_message: SystemMessage | None, extra_text: str) -> SystemMessage:
        if system_message is not None:
            new_system_content = [
                *system_message.content_blocks,
                {"type": "text", "text": f"\n\n{extra_text}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": extra_text}]
        return SystemMessage(content=cast("list[str | dict[str, str]]", new_system_content))

    def _build_query_plan_hint(
        self,
        *,
        query_plan: AcademicQueryPlan | None,
        retrieval_sufficient: bool | None,
        relevance_reason: str | None,
    ) -> str:
        if query_plan is None:
            return "当前未注入查询计划。"

        stage_hint = (
            "下一步必须调用 `retrieve_local_kb`，参数使用 retrieval_query_zh。"
            if retrieval_sufficient is None
            else "检索结果已足够，请直接基于证据回答。"
            if retrieval_sufficient
            else "检索结果不足，下一步必须调用 `crawl_academic_sources`，参数使用 crawler_query_en 与 keywords_en。"
        )
        return (
            "## Academic Query Plan\n"
            f"- normalized_query_zh: {query_plan.normalized_query_zh}\n"
            f"- retrieval_query_zh: {query_plan.retrieval_query_zh}\n"
            f"- retrieval_query_en: {query_plan.retrieval_query_en}\n"
            f"- crawler_query_en: {query_plan.crawler_query_en}\n"
            f"- keywords_zh: {query_plan.keywords_zh}\n"
            f"- keywords_en: {query_plan.keywords_en}\n"
            f"- required_aspects: {query_plan.required_aspects}\n"
            f"- relevance_reason: {relevance_reason or '尚未评估'}\n"
            f"- workflow_constraint: {stage_hint}\n"
            "回答必须基于工具返回的 JSON 证据，不允许凭空补写。"
        )

    def _extract_latest_user_query(self, state: ResearchState) -> str:
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                if isinstance(message.content, str):
                    return message.content.strip()
                if isinstance(message.content, list):
                    parts = []
                    for block in message.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(str(block.get("text", "")))
                    return " ".join(parts).strip()
                return str(message.content).strip()
        return ""
