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
4. 调用 `crawl_academic_sources` 时，参数必须直接使用系统给出的 `missing_aspects`。
5. 最终只基于工具返回的 JSON 证据回答用户问题。

回答要求：
- 用中文输出。
- 明确区分“本地检索证据”和“爬虫补充证据”。
- 对爬虫证据，优先按照 `aspect_evidence` 的结构组织：缺失点 -> chunk -> 论文标题/出处。
- 不要编造文献内容，不要臆测论文结论。
- 如果证据仍然不足，要明确指出哪些 missing_aspects 还没有被覆盖。
- 回答时优先引用工具 JSON 中的标题、摘要、来源与 missing_aspects 覆盖情况。
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
