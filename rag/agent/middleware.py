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
    from ..schemas import (
        AcademicQueryPlan,
        FinalEvidenceBundle,
        ResearchState,
        RetrievalPayload,
        WebSearchPayload,
    )
    from .evidence import (
        annotate_local_documents,
        build_final_evidence_bundle,
        normalized_doc_to_final_evidence_item,
        select_local_evidence,
    )
    from .runtime import context, log_progress
except ImportError:
    from query.optimizer import AcademicQueryPlanner
    from retrieval.evaluator import evaluate_retrieval
    from schemas import (
        AcademicQueryPlan,
        FinalEvidenceBundle,
        ResearchState,
        RetrievalPayload,
        WebSearchPayload,
    )
    from agent.evidence import (
        annotate_local_documents,
        build_final_evidence_bundle,
        normalized_doc_to_final_evidence_item,
        select_local_evidence,
    )
    from agent.runtime import context, log_progress


class AcademicResearchMiddleware(AgentMiddleware[ResearchState, Any]):
    """Plan the query before the agent starts and enforce retrieval-first execution."""

    state_schema = ResearchState
    PROMPT_LOCAL_EVIDENCE_MAX_ITEMS = 3
    PROMPT_WEB_EVIDENCE_MAX_ITEMS = 5
    PROMPT_EVIDENCE_CONTENT_MAX_CHARS = 900
    TOOL_SUMMARY_EVIDENCE_MAX_ITEMS = 4
    TOOL_SUMMARY_CONTENT_MAX_CHARS = 260

    def __init__(self, llm, retrieve_tool_name: str, web_search_tool_name: str) -> None:
        super().__init__()
        self.planner = AcademicQueryPlanner(llm)
        self.retrieve_tool_name = retrieve_tool_name
        self.web_search_tool_name = web_search_tool_name

    def before_agent(self, state: ResearchState, runtime) -> dict[str, Any] | None:
        original_query = self._extract_latest_user_query(state)
        plan = self.planner.build(original_query)
        context.reset()
        context.original_query = original_query
        context.query_plan = plan.model_dump()
        context.set_current_missing_aspects([])
        log_progress("查询预处理完成，已生成学术检索计划。")
        return {
            "query_plan": plan.model_dump(),
            "retrieval_result": None,
            "retrieval_sufficient": None,
            "retrieval_next_action": None,
            "relevance_score": None,
            "relevance_reason": None,
            "relevance_aspect_coverage": None,
            "relevance_support_strength": None,
            "relevance_noise_ratio": None,
            "relevance_missing_aspects": [],
            "relevance_weak_aspects": [],
            "web_search_required": False,
            "web_search_used": False,
            "web_search_result": None,
            "final_evidence": None,
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
        retrieval_next_action = request.state.get("retrieval_next_action")
        web_search_required = request.state.get("web_search_required", False)
        web_search_used = request.state.get("web_search_used", False)

        filtered_tools = self._filter_tools(
            tools=request.tools,
            retrieval_result=retrieval_result,
            retrieval_sufficient=retrieval_sufficient,
            web_search_required=web_search_required,
            web_search_used=web_search_used,
        )
        query_plan_text = self._build_query_plan_hint(
            query_plan=query_plan,
            retrieval_sufficient=retrieval_sufficient,
            retrieval_next_action=retrieval_next_action,
            relevance_reason=request.state.get("relevance_reason"),
            aspect_coverage=request.state.get("relevance_aspect_coverage"),
            support_strength=request.state.get("relevance_support_strength"),
            noise_ratio=request.state.get("relevance_noise_ratio"),
            missing_aspects=request.state.get("relevance_missing_aspects") or [],
            final_evidence_raw=request.state.get("final_evidence"),
        )
        updated_request = self._override_model_request(
            request=request,
            filtered_tools=filtered_tools,
            extra_system_text=query_plan_text,
        )
        return handler(updated_request)

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
        if tool_name == self.web_search_tool_name:
            return self._handle_web_search_result(request, result)
        return result

    def _handle_retrieval_result(self, request: ToolCallRequest, result: ToolMessage) -> Command | ToolMessage:
        try:
            payload = RetrievalPayload(**json.loads(str(result.content)))
            query_plan = AcademicQueryPlan(**(request.state.get("query_plan") or {}))
            evaluation = evaluate_retrieval(query_plan, payload.docs)
            annotated_docs = annotate_local_documents(evaluation.scored_docs, evaluation)
            payload = payload.model_copy(update={"docs": annotated_docs})

            local_evidence = select_local_evidence(annotated_docs)
            final_bundle = build_final_evidence_bundle(
                query=query_plan.original_query,
                local_evidence=local_evidence,
                web_evidence=[],
                uncovered_aspects=[] if evaluation.sufficient else evaluation.missing_aspects,
                note=(
                    "local evidence is sufficient for answering"
                    if evaluation.sufficient
                    else "web search is required for the remaining missing aspects"
                ),
            )

            relevance_reason = evaluation.reason
            if not evaluation.sufficient:
                relevance_reason = f"{relevance_reason}; next_action=search_web"

            context.retrieval_result = payload.model_dump()
            context.set_current_missing_aspects(evaluation.missing_aspects)
            context.set_final_evidence(final_bundle.model_dump())

            updated_message = ToolMessage(
                content=self._build_retrieval_tool_summary(
                    payload=payload,
                    local_evidence=local_evidence,
                    missing_aspects=evaluation.missing_aspects,
                    next_action="answer" if evaluation.sufficient else "search_web",
                ),
                tool_call_id=result.tool_call_id,
                name=result.name,
                status=result.status,
            )
            log_progress(
                "本地检索相关性评估完成："
                f"sufficient={evaluation.sufficient}, next_action={'answer' if evaluation.sufficient else 'search_web'}"
            )
            return Command(
                update={
                    "retrieval_result": payload.model_dump(),
                    "retrieval_sufficient": evaluation.sufficient,
                    "retrieval_next_action": "answer" if evaluation.sufficient else "search_web",
                    "relevance_score": evaluation.support_strength,
                    "relevance_reason": relevance_reason,
                    "relevance_aspect_coverage": evaluation.aspect_coverage,
                    "relevance_support_strength": evaluation.support_strength,
                    "relevance_noise_ratio": evaluation.noise_ratio,
                    "relevance_missing_aspects": evaluation.missing_aspects,
                    "relevance_weak_aspects": evaluation.weak_aspects,
                    "web_search_required": not evaluation.sufficient,
                    "web_search_used": False,
                    "web_search_result": None,
                    "final_evidence": final_bundle.model_dump(),
                    "messages": [updated_message],
                }
            )
        except Exception as exc:
            log_progress(f"检索结果 hook 解析失败，保留原始结果继续执行: {exc}")
            return result

    def _handle_web_search_result(self, request: ToolCallRequest, result: ToolMessage) -> Command | ToolMessage:
        try:
            payload = WebSearchPayload(**json.loads(str(result.content)))
            context.set_web_search_result(payload.model_dump())

            local_evidence = []
            final_evidence_raw = request.state.get("final_evidence") or {}
            if final_evidence_raw:
                try:
                    bundle = FinalEvidenceBundle(**final_evidence_raw)
                    local_evidence = list(bundle.local_evidence)
                except Exception:
                    local_evidence = []

            web_evidence = [
                normalized_doc_to_final_evidence_item(doc, default_origin="tavily_web")
                for doc in payload.evidence_docs
            ]
            final_bundle = build_final_evidence_bundle(
                query=context.original_query,
                local_evidence=local_evidence,
                web_evidence=web_evidence,
                uncovered_aspects=payload.uncovered_missing_aspects,
                note="local evidence and Tavily web evidence have been merged for final answering",
            )
            context.set_final_evidence(final_bundle.model_dump())

            updated_message = ToolMessage(
                content=self._build_web_search_tool_summary(payload=payload, final_bundle=final_bundle),
                tool_call_id=result.tool_call_id,
                name=result.name,
                status=result.status,
            )
            log_progress("Tavily 搜索阶段完成，开始基于统一证据生成最终答案。")
            return Command(
                update={
                    "web_search_used": True,
                    "web_search_required": False,
                    "web_search_result": payload.model_dump(),
                    "retrieval_next_action": "answer",
                    "final_evidence": final_bundle.model_dump(),
                    "messages": [updated_message],
                }
            )
        except Exception as exc:
            log_progress(f"网页搜索结果 hook 解析失败，保留原始结果继续执行: {exc}")
            return result

    def _filter_tools(
        self,
        tools: list[Any],
        retrieval_result: dict[str, Any] | None,
        retrieval_sufficient: bool | None,
        web_search_required: bool,
        web_search_used: bool,
    ) -> list[Any]:
        if not retrieval_result:
            allowed = {self.retrieve_tool_name}
        elif retrieval_sufficient:
            allowed = set()
        elif web_search_required and not web_search_used:
            allowed = {self.web_search_tool_name}
        else:
            allowed = set()
        return [tool for tool in tools if getattr(tool, "name", None) in allowed]

    def _override_model_request(
        self,
        *,
        request: ModelRequest,
        filtered_tools: list[Any],
        extra_system_text: str,
    ) -> ModelRequest:
        overrides: dict[str, Any] = {"tools": filtered_tools}

        request_system_message = getattr(request, "system_message", None)
        if request_system_message is not None or hasattr(request, "system_message"):
            overrides["system_message"] = self._append_system_message(request_system_message, extra_system_text)
            return request.override(**overrides)

        request_system_prompt = getattr(request, "system_prompt", None)
        if request_system_prompt is not None or hasattr(request, "system_prompt"):
            overrides["system_prompt"] = self._append_system_prompt(request_system_prompt, extra_system_text)
            return request.override(**overrides)

        request_messages = list(getattr(request, "messages", []) or [])
        overrides["messages"] = self._prepend_system_message(request_messages, extra_system_text)
        return request.override(**overrides)

    def _append_system_message(self, system_message: SystemMessage | None, extra_text: str) -> SystemMessage:
        if system_message is not None:
            new_system_content = [
                *self._system_content_blocks(system_message),
                {"type": "text", "text": f"\n\n{extra_text}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": extra_text}]
        return SystemMessage(content=cast("list[str | dict[str, str]]", new_system_content))

    def _append_system_prompt(self, system_prompt: str | None, extra_text: str) -> str:
        prompt = str(system_prompt or "").strip()
        if not prompt:
            return extra_text
        return f"{prompt}\n\n{extra_text}"

    def _prepend_system_message(self, messages: list[Any], extra_text: str) -> list[Any]:
        if messages and isinstance(messages[0], SystemMessage):
            merged = self._append_system_message(messages[0], extra_text)
            return [merged, *messages[1:]]
        return [SystemMessage(content=extra_text), *messages]

    def _system_content_blocks(self, system_message: SystemMessage) -> list[Any]:
        content_blocks = getattr(system_message, "content_blocks", None)
        if content_blocks:
            return list(content_blocks)

        content = getattr(system_message, "content", "")
        if isinstance(content, list):
            return list(content)
        if content:
            return [{"type": "text", "text": str(content)}]
        return []

    def _build_query_plan_hint(
        self,
        *,
        query_plan: AcademicQueryPlan | None,
        retrieval_sufficient: bool | None,
        retrieval_next_action: str | None,
        relevance_reason: str | None,
        aspect_coverage: float | None,
        support_strength: float | None,
        noise_ratio: float | None,
        missing_aspects: list[str],
        final_evidence_raw: dict[str, Any] | None,
    ) -> str:
        if query_plan is None:
            return "当前未注入查询计划。"

        if retrieval_sufficient is None:
            stage_hint = "下一步必须调用 `retrieve_local_kb`，参数使用 retrieval_query_zh。"
        elif retrieval_sufficient:
            stage_hint = "本地证据已经足够，禁止继续调用工具，直接使用 final_evidence_bundle 回答。"
        elif retrieval_next_action == "search_web":
            stage_hint = "本地证据不足，下一步必须调用 `search_web_with_tavily`，参数直接使用当前 missing_aspects 列表。"
        else:
            stage_hint = "如果 final_evidence_bundle 已存在，请直接基于它回答。"

        metrics_text = (
            "尚未评估"
            if aspect_coverage is None or support_strength is None or noise_ratio is None
            else (
                f"aspect_coverage={aspect_coverage:.2f}, "
                f"support_strength={support_strength:.2f}, "
                f"noise_ratio={noise_ratio:.2f}"
            )
        )

        final_evidence_text = "尚未生成"
        if final_evidence_raw:
            try:
                final_evidence = FinalEvidenceBundle(**final_evidence_raw)
                final_evidence_text = self._build_prompt_friendly_final_evidence(final_evidence)
            except Exception:
                final_evidence_text = self._build_prompt_friendly_final_evidence(final_evidence_raw)

        return (
            "## Academic Query Plan\n"
            f"- normalized_query_zh: {query_plan.normalized_query_zh}\n"
            f"- retrieval_query_zh: {query_plan.retrieval_query_zh}\n"
            f"- retrieval_query_en: {query_plan.retrieval_query_en}\n"
            f"- crawler_fallback_en: {query_plan.crawler_query_en}\n"
            f"- keywords_zh: {query_plan.keywords_zh}\n"
            f"- keywords_en_fallback: {query_plan.keywords_en}\n"
            f"- required_aspects: {query_plan.required_aspects}\n"
            f"- retrieval_next_action: {retrieval_next_action or 'pending'}\n"
            f"- evaluation_metrics: {metrics_text}\n"
            f"- missing_aspects: {missing_aspects}\n"
            f"- relevance_reason: {relevance_reason or '尚未评估'}\n"
            f"- workflow_constraint: {stage_hint}\n"
            "## Final Evidence Bundle\n"
            f"{final_evidence_text}\n"
            "最终回答必须优先依据 Final Evidence Bundle。"
            " 其中每条证据都已经标准化为 origin/content/aspects/title/url。"
            " 请优先按 aspect 组织回答，明确区分 local_kb 与 tavily_web。"
            " 如果 uncovered_aspects 非空，必须明确指出仍未覆盖的点。"
        )

    def _build_retrieval_tool_summary(
        self,
        *,
        payload: RetrievalPayload,
        local_evidence: list[Any],
        missing_aspects: list[str],
        next_action: str,
    ) -> str:
        summary = {
            "kind": "retrieval_result_summary",
            "status": payload.status,
            "message": payload.message,
            "query": payload.query,
            "doc_count": payload.doc_count,
            "next_action": next_action,
            "missing_aspects": list(missing_aspects[:5]),
            "selected_local_evidence": [
                self._compact_evidence_item(item, content_max_chars=self.TOOL_SUMMARY_CONTENT_MAX_CHARS)
                for item in local_evidence[: self.TOOL_SUMMARY_EVIDENCE_MAX_ITEMS]
            ],
        }
        return json.dumps(summary, ensure_ascii=False)

    def _build_web_search_tool_summary(
        self,
        *,
        payload: WebSearchPayload,
        final_bundle: FinalEvidenceBundle,
    ) -> str:
        summary = {
            "kind": "web_search_result_summary",
            "status": payload.status,
            "message": payload.message,
            "requested_missing_aspects": list(payload.requested_missing_aspects[:5]),
            "covered_missing_aspects": list(payload.covered_missing_aspects[:5]),
            "uncovered_missing_aspects": list(payload.uncovered_missing_aspects[:5]),
            "search_queries": [item.query for item in payload.search_queries[:5]],
            "selected_web_evidence": [
                self._compact_evidence_item(item, content_max_chars=self.TOOL_SUMMARY_CONTENT_MAX_CHARS)
                for item in final_bundle.web_evidence[: self.TOOL_SUMMARY_EVIDENCE_MAX_ITEMS]
            ],
        }
        return json.dumps(summary, ensure_ascii=False)

    def _build_prompt_friendly_final_evidence(self, bundle_like: FinalEvidenceBundle | dict[str, Any]) -> str:
        bundle = bundle_like if isinstance(bundle_like, FinalEvidenceBundle) else FinalEvidenceBundle(**bundle_like)
        prompt_bundle = {
            "query": bundle.query,
            "summary": bundle.summary,
            "local_evidence": [
                self._compact_evidence_item(item, content_max_chars=self.PROMPT_EVIDENCE_CONTENT_MAX_CHARS)
                for item in bundle.local_evidence[: self.PROMPT_LOCAL_EVIDENCE_MAX_ITEMS]
            ],
            "web_evidence": [
                self._compact_evidence_item(item, content_max_chars=self.PROMPT_EVIDENCE_CONTENT_MAX_CHARS)
                for item in bundle.web_evidence[: self.PROMPT_WEB_EVIDENCE_MAX_ITEMS]
            ],
            "uncovered_aspects": list(bundle.uncovered_aspects[:8]),
            "note": (
                "Evidence content below is excerpted and truncated for prompt budget control. "
                "Prefer these items over raw tool payloads."
            ),
        }
        return json.dumps(prompt_bundle, ensure_ascii=False, indent=2)

    def _compact_evidence_item(self, item: Any, *, content_max_chars: int) -> dict[str, Any]:
        evidence = item if isinstance(item, dict) else item.model_dump()
        content = self._truncate_text(str(evidence.get("content") or "").strip(), content_max_chars)
        return {
            "origin": str(evidence.get("origin") or "").strip(),
            "title": str(evidence.get("title") or "").strip(),
            "url": str(evidence.get("url") or "").strip(),
            "aspects": list(evidence.get("aspects") or [])[:6],
            "score": evidence.get("score"),
            "content_excerpt": content,
        }

    def _truncate_text(self, text: str, max_chars: int) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip() + "...(truncated)"

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
