
from tool.tools import (
    query_transform,
    retriever,
    value_evaluator,
    web_deep_research,
    save_sub_task_result,
    report_generator,
    render_text_description
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 设置 LLM 
llm = ChatOpenAI(
    model="qwen-plus",
    api_key="sk-e4b7b6386950428bb71c658d47da47ef",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
) 

SYSTEM_PROMPT = """
# Role:你是一个自主决策的深度研究 Agent。你的目标是接收用户的原始查询，拆解它，逐个攻克子问题，最后生成报告。

# Tools Capabilities
1. `query_transform`: 必须首先调用。将复杂问题拆解为子问题列表。
2. `retriever`: 基础检索。
3. `value_evaluator`: 必须在检索后调用。评估检索结果是否足以回答子问题。返回 "YES" 或 "NO"。
4. `web_deep_research`: 只有在 `value_evaluator` 返回 "NO" 时才能调用。
5. `save_sub_task_result`: **关键步骤**。每当你解决完一个子问题（无论通过检索还是爬虫），必须调用此工具保存结果。
6. `report_generator`: 只有在**所有**子问题都通过 `save_sub_task_result` 保存后，才能调用此工具生成最终答案。

# Execution Stategy (Thinking Loop)
你必须维护自己的“待办事项清单”。收到用户查询后：

1. **Plan**: 调用 `query_transform` 得到子问题列表 [Q1, Q2, Q3...]。
2. **Loop (针对每个子问题 Q)**:
   a. **Act**: 调用 `retriever(Q)`。
   b. **Check**: 调用 `value_evaluator(Q, docs)`。
   c. **Decide**:
      - 如果返回 "YES": 基于文档生成答案 -> 调用 `save_sub_task_result(Q, answer)`。
      - 如果返回 "NO": 调用 `web_deep_research(Q)` -> 基于爬虫结果生成答案 -> 调用 `save_sub_task_result(Q, answer)`。
3. **Finish**: 检查是否列表中的所有问题都已执行过 `save_sub_task_result`。如果是，调用 `report_generator()` 结束任务。

# Constraints
- **不要**在一次对话中试图凭记忆回答所有问题，必须依赖 `save_sub_task_result` 工具记录。
- **严禁**在评估通过（YES）的情况下调用爬虫。
- **严禁**在未完成所有子问题时调用 `report_generator`。
- 遇到子问题循环时，请一步步来，不要试图在一个 Function Call 里做完所有事。保持耐心。
"""

# 使用工具描述来构建prompt
tool_descriptions = render_text_description(tools)
system_message = SYSTEM_PROMPT + "\n\n你可用的工具包括:\n{tool_descriptions}".format(tool_descriptions=tool_descriptions)

# 更新prompt以包含工具描述
#prompt_with_tools = ChatPromptTemplate.from_messages([ 
    #("system", system_message), 
    #("user", "{input}"), 
   # MessagesPlaceholder(variable_name="agent_scratchpad"), # 必须保留，存放 Agent 的思考和工具调用历史 
#]) 

from langchain.agents import create_agent

class RagService:
    def __init__(self): 
        self.agent = create_agent( 
            model=llm, 
            tools = [ 
                query_transform, 
                retriever, 
                value_evaluator, 
                web_deep_research, 
                save_sub_task_result, 
                report_generator 
            ] , 
            prompt=system_message, 
            verbose=True, 
        )

    def run(self, query: str, thread_id: str = None):
        """支持多会话（通过 thread_id 区分状态）"""
        #config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        return self.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            #config=config
        )


if __name__ == "__main__": 
    user_query = "如何优化 RAG 系统中的检索质量？" 
    
    agent = RagService()
    result = agent.run(user_query)

    
    print("\n\n>>>>>>>>>> AGENT 最终输出 <<<<<<<<<<") 
