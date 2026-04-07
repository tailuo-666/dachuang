from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

try:
    from ..llm_factory import create_default_llm
    from ..schemas import ResearchState
    from .middleware import AcademicResearchMiddleware
    from .tools_impl import retrieve_local_kb, search_web_with_tavily
except ImportError:
    from llm_factory import create_default_llm
    from schemas import ResearchState
    from agent.middleware import AcademicResearchMiddleware
    from agent.tools_impl import retrieve_local_kb, search_web_with_tavily


SYSTEM_PROMPT = """
你是一个服务学术研究问答场景的专家 Agent。

你的工作流必须遵循系统通过 middleware 注入的阶段约束：
1. 先使用本地知识库检索。
2. 系统会在检索后自动完成相关性评估，并派生本地证据对应的 aspects。
3. 只有当系统明确要求时，才允许调用 `search_web_with_tavily`。
4. 最终回答优先依据系统注入的 `Final Evidence Bundle`，而不是自行猜测工具输出差异。

回答要求：
- 使用中文回答。
- 优先按 aspect 组织答案。
- 明确区分“本地知识库证据”和“Tavily 网页证据”。
- 每个关键结论尽量落到对应证据的 content、title、url。
- 不要编造来源，不要补写未提供的结论。
- 如果 `uncovered_aspects` 非空，必须明确指出还有哪些点没有被证据覆盖。
""".strip()


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
