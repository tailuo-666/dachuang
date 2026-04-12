from __future__ import annotations

import json
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

try:
    from ..llm_factory import create_default_llm
    from ..schemas import AgentAnswerPayload, ResearchState
    from .middleware import AcademicResearchMiddleware
    from .tools_impl import retrieve_local_kb, search_web_with_tavily
except ImportError:
    from llm_factory import create_default_llm
    from schemas import AgentAnswerPayload, ResearchState
    from agent.middleware import AcademicResearchMiddleware
    from agent.tools_impl import retrieve_local_kb, search_web_with_tavily


SYSTEM_PROMPT = """
You are an expert agent serving academic research Q&A scenarios.

Your workflow must strictly follow the stage constraints injected by the system through middleware:
1. You must first perform retrieval using the local knowledge base.
2. After retrieval, the system will automatically conduct relevance assessment and assemble the final usable evidence.
3. You may call `search_web_with_tavily` only when the system explicitly instructs you to do so.
4. Your final answer must be based primarily on the system-injected `Final Evidence Bundle`, rather than on your own guesses about discrepancies in tool outputs.

Answer Requirements
- Respond in Chinese.
- Organize the main body primarily by aspect.
- Do not use phrases such as “local knowledge base,” “web search,” “Tavily,” or “source type” in the main body.
- Do not output `title`, `url`, or `origin` in the main body.
- If a sentence or paragraph depends on a specific piece of evidence, append the corresponding `[index]` at the end of that sentence or paragraph.
- If a conclusion is jointly supported by multiple pieces of evidence, cite them as `[1][3]`.
- You may cite only indexes that exist in the `Final Evidence Bundle`.
- Do not cite evidence that was not provided, do not fabricate sources, and do not add conclusions beyond what the evidence supports.
- If `uncovered_aspects` is non-empty, ignore the uncovered parts directly and do not explicitly mention them.

Final Output Requirements
- Output only a single JSON object. Do not output a Markdown code block or any additional explanation.
- The JSON structure must be exactly:
  `{"answer":"...", "evidence_list":[1,2,4]}`
- `evidence_list` must include only the evidence indexes that are actually cited in `answer`.
""".strip()

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def extract_final_response_text(agent_result: dict[str, Any]) -> str:
    messages = agent_result.get("messages", [])
    for msg in reversed(messages):
        content = _coerce_message_content(msg.content if hasattr(msg, "content") else msg.get("content", ""))
        tool_calls = getattr(msg, "tool_calls", None) or (
            msg.get("tool_calls") if isinstance(msg, dict) else None
        )
        if content and not tool_calls:
            return content
    return ""


def _try_parse_json_object(candidate: str) -> dict[str, Any] | None:
    text = str(candidate or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    try:
        parsed, _ = json.JSONDecoder().raw_decode(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    stripped = str(raw_text or "").strip()
    if not stripped:
        return None

    seen: set[str] = set()
    candidates: list[str] = [stripped]
    candidates.extend(match.group(1).strip() for match in _JSON_FENCE_PATTERN.finditer(stripped))

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        candidate = stripped[match.start() :].lstrip()
        try:
            parsed, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(candidate[:end])

    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parsed = _try_parse_json_object(normalized)
        if parsed is not None:
            return parsed
    return None


def _remove_invalid_citations(answer: str, valid_indexes: set[int]) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return match.group(0) if index in valid_indexes else ""

    cleaned = _CITATION_PATTERN.sub(replace, answer)
    cleaned = re.sub(r"[ \t]+(\r?\n)", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_used_citations(answer: str, valid_indexes: set[int]) -> list[int]:
    used: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_PATTERN.finditer(answer):
        index = int(match.group(1))
        if index not in valid_indexes or index in seen:
            continue
        seen.add(index)
        used.append(index)
    return used


def parse_agent_answer(
    agent_result: dict[str, Any],
    final_evidence_items: list[dict[str, Any]] | None = None,
) -> AgentAnswerPayload:
    raw_text = extract_final_response_text(agent_result)
    valid_indexes = {
        int(item.get("index") or 0)
        for item in (final_evidence_items or [])
        if isinstance(item, dict) and int(item.get("index") or 0) > 0
    }

    parsed_payload = None
    parsed_json = _extract_json_object(raw_text)
    if parsed_json is not None:
        try:
            parsed_payload = AgentAnswerPayload(**parsed_json)
        except Exception:
            parsed_payload = None

    answer = parsed_payload.answer if parsed_payload is not None else raw_text
    answer = _remove_invalid_citations(str(answer or "").strip(), valid_indexes)
    evidence_list = _extract_used_citations(answer, valid_indexes)
    return AgentAnswerPayload(answer=answer, evidence_list=evidence_list)


class RagService:
    def __init__(self, llm=None):
        self.llm = llm or create_default_llm()
        self.middleware = AcademicResearchMiddleware(
            self.llm,
            retrieve_tool_name=retrieve_local_kb.name,
            web_search_tool_name=search_web_with_tavily.name,
        )
        self.agent = create_agent(
            model=self.llm,
            tools=[retrieve_local_kb, search_web_with_tavily],
            system_prompt=SYSTEM_PROMPT,
            state_schema=ResearchState,
            middleware=[self.middleware],
            debug=False,
        )

    def run(self, query: str, thread_id: str | None = None):
        return self.agent.invoke({"messages": [HumanMessage(content=query)]})

    def parse_final_response(
        self,
        agent_result: dict[str, Any],
        final_evidence_items: list[dict[str, Any]] | None = None,
    ) -> AgentAnswerPayload:
        return parse_agent_answer(agent_result, final_evidence_items=final_evidence_items)
