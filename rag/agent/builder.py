from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

try:
    from ..llm_factory import create_default_llm
    from ..schemas import ResearchState
    from .middleware import AcademicResearchMiddleware
    from .tools_impl import crawl_academic_sources, retrieve_local_kb
except ImportError:
    from llm_factory import create_default_llm
    from schemas import ResearchState
    from agent.middleware import AcademicResearchMiddleware
    from agent.tools_impl import crawl_academic_sources, retrieve_local_kb


SYSTEM_PROMPT = """
你是一个专门服务学术研究场景的专家型 Agent。

你的任务只有一条单查询主线：
1. 先使用本地知识库检索。
2. 检索后由系统 hook 自动判断相关性是否充足。
3. 只有当系统明确判定信息不足时，才允许调用学术爬虫。
4. 最终只基于工具返回的证据回答用户问题。

回答要求：
- 用中文输出。
- 明确区分“本地检索证据”和“爬虫补充证据”。
- 不要编造文献内容，不要臆测论文结论。
- 如果证据仍不足，要清楚说明不足点。
- 回答时优先引用工具 JSON 中的标题、摘要、来源信息。
""".strip()


class RagService:
    def __init__(self, llm=None):
        self.llm = llm or create_default_llm()
        self.middleware = AcademicResearchMiddleware(
            self.llm,
            retrieve_tool_name=retrieve_local_kb.name,
            crawl_tool_name=crawl_academic_sources.name,
        )
        self.agent = create_agent(
            model=self.llm,
            tools=[retrieve_local_kb, crawl_academic_sources],
            system_prompt=SYSTEM_PROMPT,
            state_schema=ResearchState,
            middleware=[self.middleware],
            debug=False,
        )

    def run(self, query: str, thread_id: str | None = None):
        return self.agent.invoke({"messages": [HumanMessage(content=query)]})
