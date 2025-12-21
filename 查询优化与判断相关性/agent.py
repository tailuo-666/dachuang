

# ========================================
# 第一部分：导入和基础配置
# ========================================

import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from typing_extensions import NotRequired
import json

# LangChain 1.0 核心导入
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ModelCallResult
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, BaseTool
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents import AgentState
# from langchain.runtime import Runtime
from typing import Callable

# LLM模型配置（保持与原文件兼容）
from langchain_openai import ChatOpenAI

print("[OK] LangChain 1.0库导入完成")

# 初始化LLM模型
llm = ChatOpenAI(
    model="qwen-plus",
    api_key="sk-e4b7b6386950428bb71c658d47da47ef",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

print("[OK] LLM模型初始化完成")

# ========================================
# 第二部分：Agent状态定义
# ========================================

class CorrectiveRAGState(AgentState):
    """扩展的Agent状态，用于跟踪Corrective RAG流程"""
    # 继承messages字段
    last_rag_query: NotRequired[Optional[str]]       # 当前RAG查询
    documents_relevant: NotRequired[Optional[bool]]  # 文档相关性评估结果
    grade_reasoning: NotRequired[Optional[str]]      # 评估理由
    used_web_fallback: NotRequired[bool]             # 是否使用了爬虫回退
    subquestion_results: NotRequired[List[Dict]]     # 子问题处理结果
    current_subquestion: NotRequired[Optional[str]]  # 当前处理的子问题
    decomposition_needed: NotRequired[bool]          # 是否需要分解问题
    subquestions: NotRequired[List[str]]            # 分解后的子问题列表

print("[OK] CorrectiveRAGState定义完成")

# ========================================
# 第三部分：Agent工具定义
# ========================================

@tool
def search_documents(query: str) -> str:
    """
    文档检索工具
    
    Args:
        query: 检索查询字符串
        
    Returns:
        str: 检索到的文档内容
    """
    print("[Searching] 正在检索文档...")
    
    # 模拟检索过程（实际应用中可替换为真实的向量检索）
    mock_documents = {
        "向量数据库": "向量数据库是一种专门用于存储和检索高维向量数据的数据库系统...",
        "RAG": "RAG（Retrieval-Augmented Generation）是一种结合了信息检索和生成式AI的技术...",
        "LangChain": "LangChain是一个用于开发基于大语言模型的应用程序的框架..."
    }
    
    # 简单的关键词匹配检索
    for key, content in mock_documents.items():
        if key in query:
            print(f"[Success] 检索成功：找到相关文档（{len(content)}字符）")
            return content
    
    # 如果没有找到相关文档，返回空结果
    print("[Not Found] 未找到相关文档")
    return ""

@tool  
def crawl_web_data(query: str) -> str:
    """
    网络爬虫工具
    
    Args:
        query: 爬虫查询字符串
        
    Returns:
        str: 爬虫获取的文档内容
    """
    print("[Crawling] 正在爬虫获取信息...")
    
    # 模拟爬虫过程（实际应用中可替换为真实的爬虫逻辑）
    mock_crawler_results = {
        "向量数据库": "通过网络爬虫获取的最新信息：向量数据库技术正在快速发展，许多新兴的向量数据库如Pinecone、Weaviate等提供了强大的相似性搜索功能...",
        "RAG": "最新研究显示，RAG技术在2024年取得了显著进展，多模态RAG、Self-RAG等新方法不断涌现...",
        "LangChain": "LangChain生态系统持续扩展，新版本提供了更多中间件选项和更好的性能优化..."
    }
    
    # 简单的关键词匹配
    for key, content in mock_crawler_results.items():
        if key in query:
            print(f"[Success] 爬虫成功：获取到补充信息（{len(content)}字符）")
            return content
    
    # 默认返回模拟结果
    default_result = f"爬虫获取的关于'{query}'的最新信息：通过网络爬虫获得的相关内容，包含该主题的最新发展动态和实用信息..."
    print(f"[Success] 爬虫成功：获取到默认补充信息（{len(default_result)}字符）")
    return default_result   

print("[OK] Agent工具定义完成")
print(f"[Tools] 可用工具：[{search_documents.name}, {crawl_web_data.name}]")

# ========================================
# 第四部分：查询分解组件（保持原有逻辑）
# ========================================

# 定义触发分解的关键词列表
DECOMPOSITION_KEYWORDS = {
    "逻辑关系": ["和", "与", "或", "对比", "差异", "区别", "分别", "谁更", "既", "又", "不仅", "还", "以及", "同时", "相比之下", "相对"],
    "复杂意图": ["为什么", "如何解决", "步骤", "流程", "分析", "总结", "推荐", "论证", "原因", "对策", "优化方案", "怎样", "如何实现", "方法", "策略", "机制", "原理"],
    "多属性/多维度": ["优缺点", "优势劣势", "性能", "价格", "功能", "适用场景", "部署难度", "学习曲线", "成本", "效率", "可靠性", "可扩展性", "兼容性", "用户体验", "维护"]
}

def extract_entities(query: str) -> List[str]:
    """极简版核心实体提取方法"""
    entities = re.split(r'[，,、\s]+', query)
    entities = [entity.strip() for entity in entities if len(entity.strip()) >= 2]
    return entities

def rule_based_prejudgment(query: str) -> str:
    """规则预判断函数"""
    print(f"🔍 规则预判断: {query}")
    
    # 检查是否包含触发分解的关键词
    has_trigger_keyword = False
    for category, keywords in DECOMPOSITION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query:
                has_trigger_keyword = True
                print(f"  [Trigger] 触发关键词: '{keyword}' (类别: {category})")
                break
        if has_trigger_keyword:
            break
    
    # 提取核心实体并计算数量
    entities = extract_entities(query)
    entity_count = len(entities)
    print(f"  [Entities] 核心实体: {entities} (数量: {entity_count})")
    
    # 规则判断
    if not has_trigger_keyword and entity_count <= 1:
        print("  ✅ 明确'否': 无触发词 + 核心实体≤1个")
        return "no"
    elif has_trigger_keyword and entity_count >= 2:
        print("  🔧 明确'是': 有触发词 + 核心实体≥2个")
        return "yes"
    else:
        print("  🤔 模糊case: 介于两者之间")
        return "maybe"

def llm_light_classification(query: str) -> str:
    """LLM轻量分类函数"""
    print(f"[LLM Classification] LLM轻量分类: {query}")
    
    try:
        classification_template = """用户查询：{query}
    任务：判断该查询是否需要拆分成2个及以上独立子问题，才能完整、准确回答。
    要求：1. 仅输出"是"或"否"；2. 无任何额外解释；3. 隐性多意图也需判定为"是"（如"如何优化RAG"需拆成"优化方向+具体方法"）。"""
        
        classification_prompt = ChatPromptTemplate.from_template(classification_template)
        response = llm.invoke(classification_prompt.format(query=query))
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # 标准化输出结果
        if "是" in result:
            print("  ✅ LLM分类结果: 是")
            return "yes"
        elif "否" in result:
            print("  ✅ LLM分类结果: 否")
            return "no"
        else:
            print(f"  ⚠️ 无法识别LLM输出: {result}，默认返回'no'")
            return "no"
    except Exception as e:
        print(f"  [Warning] LLM分类出错: {e}，默认返回'no'")
        return "no"

def generate_subquestions(query: str) -> List[str]:
    """子问题生成函数"""
    print(f"[Sub-question] 子问题生成: {query}")
    
    try:
        subquestion_template = """用户原始查询：{query}
    任务：将其拆分为2-5个独立子问题，需满足：
    1. 每个子问题只对应1个信息点，无重叠；
    2. 保留原始查询的核心上下文（如时间、主体、场景），不丢失关键约束；
    3. 子问题直接可用于检索（无需额外补充信息）；
    4. 不生成冗余子问题（如"对比AB"不拆"什么是A""什么是B"）。
    输出格式：按1.、2.、3.…编号列出子问题，无其他内容。"""
        
        subquestion_prompt = ChatPromptTemplate.from_template(subquestion_template)
        response = llm.invoke(subquestion_prompt.format(query=query))
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # 解析子问题列表
        sub_questions = []
        lines = result.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.\s*', line):
                question = re.sub(r'^\d+\.\s*', '', line).strip()
                if question:
                    sub_questions.append(question)
        
        print(f"  [Success] 生成了 {len(sub_questions)} 个子问题:")
        for i, q in enumerate(sub_questions, 1):
            print(f"    {i}. {q}")
            
        return sub_questions
    except Exception as e:
        print(f"  [Warning] 子问题生成出错: {e}")
        return []

def should_decompose_query(query: str) -> tuple[bool, List[str]]:
    """判断查询是否需要分解的主函数"""
    print("=" * 50)
    print("[Start] 开始判断查询是否需要分解")
    print(f"[Query] 原始查询: {query}")
    print("=" * 50)
    
    # 第一步：规则预判断
    rule_result = rule_based_prejudgment(query)
    
    if rule_result == "no":
        print("🔚 规则预判断结果: 不需要分解")
        return False, [query]
    elif rule_result == "yes":
        print("🔜 规则预判断结果: 需要分解")
        sub_questions = generate_subquestions(query)
        return True, sub_questions
    else:  # rule_result == "maybe"
        print("🤔 规则预判断结果: 模糊case，进入LLM轻量分类")
        # 第二步：LLM轻量分类
        llm_result = llm_light_classification(query)
        
        if llm_result == "no":
            print("🔚 LLM轻量分类结果: 不需要分解")
            return False, [query]
        else:  # llm_result == "yes"
            print("🔜 LLM轻量分类结果: 需要分解")
            sub_questions = generate_subquestions(query)
            return True, sub_questions

print("[OK] 问题分解组件定义完成")

# ========================================
# 第五部分：文档评估
# ========================================

class SimpleDocumentGrader:
    """简化的文档相关性评估器 - 仅判断文档数量是否为0"""
    
    def evaluate_documents(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """
        评估检索结果
        
        Args:
            query: 查询字符串
            documents: 检索到的文档列表
            
        Returns:
            Dict: 包含is_relevant和reasoning
        """
        doc_count = len(documents) if documents else 0
        
        if doc_count == 0:
            return {
                "is_relevant": False,
                "reasoning": f"检索到0个文档，需要使用爬虫获取更多信息",
                "doc_count": 0,
                "action": "use_crawler"  # 明确指示下一步行动
            }
        else:
            return {
                "is_relevant": True, 
                "reasoning": f"检索到{doc_count}个文档，内容可直接用于回答",
                "doc_count": doc_count,
                "action": "use_retrieved_docs"  # 明确指示下一步行动
            }

class DocumentGradingMiddleware(AgentMiddleware[CorrectiveRAGState]):
    """文档分级中间件 - 简化的文档数量评估"""
    
    state_schema = CorrectiveRAGState
    
    def __init__(self, rag_tool_name: str = "search_documents"):
        self.rag_tool_name = rag_tool_name
        self.doc_grader = SimpleDocumentGrader()
        self._pending_grading: dict | None = None
    
    def before_model(self, state: CorrectiveRAGState) -> dict[str, Any] | None:
        """应用挂起的评估结果到状态"""
        # 应用任何挂起的评估结果
        if self._pending_grading is not None:
            updates = self._pending_grading
            self._pending_grading = None  # 清除
            return updates
        
        return None
    
    def after_model(self, state: CorrectiveRAGState) -> dict[str, Any] | None:
        """记录模型调用后的状态"""
        return None
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """拦截工具调用以评估RAG输出"""
        tool_name = request.tool_call.get("name", "")
        tool_args = request.tool_call.get("args", {})
        
        # 首先执行工具
        result = handler(request)
        
        # 只有在RAG工具时才评估
        if tool_name == self.rag_tool_name and isinstance(result, ToolMessage):
            query = tool_args.get("query", "")
            doc_content = result.content
            
            # 简化评估：判断文档数量是否为0
            try:
                # 假设result.content是字符串，我们按某种方式分割得到文档列表
                # 这里用简单的换行分割来模拟多个文档
                documents = doc_content.split('\n\n') if doc_content else []
                # 过滤掉空文档
                documents = [doc.strip() for doc in documents if doc.strip()]
                
                evaluation = self.doc_grader.evaluate_documents(query, documents)
                
                print(f"\n[Evaluation] 文档评估:")
                print(f"   查询: {query}")
                print(f"   文档数量: {evaluation['doc_count']}")
                print(f"   相关性: {evaluation['is_relevant']}")
                print(f"   理由: {evaluation['reasoning']}")
                
                # 存储评估结果 - 将在下一个before_model中应用
                self._pending_grading = {
                    "last_rag_query": query,
                    "documents_relevant": evaluation['is_relevant'],
                    "grade_reasoning": evaluation['reasoning'],
                }
                
            except Exception as e:
                print(f"[Warning] 评估失败: {e}")
        
        return result

class ToolFilteringMiddleware(AgentMiddleware[CorrectiveRAGState]):
    """工具过滤中间件 - Wrap-Style实现"""
    
    state_schema = CorrectiveRAGState
    
    def __init__(self, rag_tool_name: str, web_tool_name: str):
        self.rag_tool_name = rag_tool_name
        self.web_tool_name = web_tool_name
        self._web_search_called: bool = False
    
    def before_model(self, state: CorrectiveRAGState) -> dict[str, Any] | None:
        """应用网络搜索跟踪到状态"""
        if self._web_search_called:
            self._web_search_called = False
            return {"used_web_fallback": True}
        return None
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """基于状态过滤工具"""
        
        documents_relevant = request.state.get("documents_relevant")
        used_web_fallback = request.state.get("used_web_fallback", False)
        
        should_enable_web = (documents_relevant is False and not used_web_fallback)
        
        if should_enable_web:
            filtered_tools = request.tools
            print(f"   [Unlocked] 启用所有工具: {[t.name for t in filtered_tools]}")
        else:
            filtered_tools = [t for t in request.tools if t.name == self.rag_tool_name]
            print(f"   [Locked] 过滤到: {[t.name for t in filtered_tools]}")
        
        modified_request = request.override(tools=filtered_tools)
        # 关键：传递modified_request而不是原始request给handler！
        return handler(modified_request)
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """跟踪网络搜索工具何时被调用"""
        tool_name = request.tool_call.get("name", "")
        result = handler(request)
        
        if tool_name == self.web_tool_name:
            self._web_search_called = True
            print(f"   [Tracking] 跟踪：网络搜索工具被调用")
        
        return result

print("[OK] 文档评估中间件定义完成")

# ========================================
# 第六部分：Agent初始化和配置（修正版）
# ========================================

# 实例化中间件（使用简化的评估器）
grading_middleware = DocumentGradingMiddleware(
    rag_tool_name="search_documents"
)

tool_filtering_middleware = ToolFilteringMiddleware(
    rag_tool_name="search_documents",
    web_tool_name="crawl_web_data"
)

# 优化后的系统提示词 - 专门处理子问题流程，同时强调原始查询的重要性
SYSTEM_PROMPT = """你是一个专业的查询处理助手，具备以下能力：
1. 使用search_documents工具从本地知识库检索相关信息
2. 使用crawl_web_data工具从互联网获取最新信息
3. 评估检索结果质量并决定是否需要使用爬虫工具
4. 基于获取的信息生成准确、全面的答案

你可以使用的工具包括：
- search_documents: 从本地知识库检索文档（优先使用）
- crawl_web_data: 从互联网获取最新信息（当本地检索无结果时使用）

处理流程：
1. 首先获取原始用户查询，它在state的original_query字段中
2. 获取当前需要处理的子问题列表，它在state的subquestions字段中
3. 对每个子问题按以下步骤处理：
   a. 使用search_documents工具检索相关信息
   b. 评估检索结果：
      - 如果检索到0个文档（content为空），则自动使用crawl_web_data工具获取补充信息
      - 如果有文档内容，则直接使用检索结果
   c. 基于获取的信息生成该子问题的答案
   d. 将答案格式化为问答对
4. 在处理完所有子问题后，基于原始用户查询整合所有子问题的答案，生成最终的综合性回答

重要规则：
- 始终记住原始用户查询，确保最终答案直接回应用户的原始提问
- 在处理每个子问题时，考虑其与原始查询的关联性
- 不要判断问题是否需要分解（已经在外部分解完成）
- 不要修改子问题的表述
- 对每个子问题都要经过完整的：检索 → 评估 → 必要时爬虫 → 生成答案 → 格式化问答对 流程
- 最后必须基于原始查询整合所有子问题的答案生成完整回答，确保上下文连贯性和逻辑一致性

原始用户查询在state的original_query字段中，当前子问题列表在state的subquestions字段中，请按顺序处理每个子问题。"""

print("[OK] Agent配置完成")

# 创建Agent
tools = [search_documents, crawl_web_data]
print(f"[Tools] 注册工具: {[t.name for t in tools]}")

# 创建Agent实例
agent = create_agent(
    model=llm,
    tools=tools,
    state_schema=CorrectiveRAGState,
    system_prompt=SYSTEM_PROMPT,
    middleware=[grading_middleware, tool_filtering_middleware]
)
print("[OK] Agent创建完成")
print("[Config] Agent配置：")
print(f"   工具: [search_documents, crawl_web_data]")
print(f"   中间件: [grading, tool_filtering]")
print(f"   状态: CorrectiveRAGState")

# ========================================
# 第七部分：Agent执行函数
# ========================================

def execute_corrective_rag_agent(query: str, sub_questions: List[str]) -> Dict[str, Any]:
    """
    执行Corrective RAG Agent的主函数
    
    Args:
        query: 用户查询问题（仅用于记录）
        sub_questions: 已经分解好的子问题列表
        
    Returns:
        Dict: 包含处理结果的字典
    """
    print("🚀 开始执行Corrective RAG Agent")
    print(f"❓ 用户查询: {query}")
    print(f"📋 接收到的子问题列表: {len(sub_questions)} 个")
    
    # 不再判断是否需要分解，直接处理子问题列表
    print(f"📋 将处理 {len(sub_questions)} 个子问题")
    
    # 初始化Agent状态，添加original_query字段
    initial_state = {
        "messages": [HumanMessage(content=f"请依次处理以下子问题：{sub_questions}")],
        "subquestions": sub_questions,
        "subquestion_results": [],
        "current_subquestion": None,
        "documents_relevant": None,
        "used_web_fallback": False,
        "original_query": query  # 添加原始查询到状态中
    }
    
    # 执行Agent
    try:
        result = agent.invoke(initial_state)
        
        print("✅ Agent执行完成")
        print(f"📊 最终状态:")
        print(f"   - 处理了 {len(sub_questions)} 个子问题")
        print(f"   - 使用爬虫回退: {result.get('used_web_fallback', False)}")
        print(f"   - 最终消息数: {len(result.get('messages', []))}")
        
        return {
            "success": True,
            "query": query,
            "subquestions": sub_questions,
            "final_result": result,
            "messages": result.get("messages", []),
            "used_web_fallback": result.get("used_web_fallback", False),
        }
        
    except Exception as e:
        print(f"❌ Agent执行出错: {e}")
        return {
            "success": False,
            "query": query,
            "subquestions": sub_questions,
            "error": str(e),
        }

def test_decomposition_functionality():
    """测试查询分解功能"""
    print("=" * 60)
    print("🧪 测试查询分解功能")
    print("=" * 60)
    
    test_queries = [
        "什么是向量数据库？",
        "向量数据库和RAG技术的区别是什么？",
        "如何优化LangChain的性能和部署？",
        "向量数据库有哪些优缺点？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 测试用例 {i}: {query}")
        print("-" * 40)
        need_decompose, sub_questions = should_decompose_query(query)
        print(f"   分解结果: {'需要分解' if need_decompose else '不需要分解'}")
        if need_decompose:
            print(f"   子问题数量: {len(sub_questions)}")
            for j, sq in enumerate(sub_questions, 1):
                print(f"     {j}. {sq}")

def test_agent_execution():
    """测试Agent执行功能"""
    print("=" * 60)
    print("🤖 测试Agent执行功能")
    print("=" * 60)
    
    # 测试简单查询（不需要分解）
    print("\n📋 测试1: 简单查询")
    simple_query = "什么是向量数据库？"
    need_decompose, sub_questions = should_decompose_query(simple_query)
    result1 = execute_corrective_rag_agent(simple_query, sub_questions)
    
    # 测试复杂查询（需要分解）
    print("\n📋 测试2: 复杂查询")
    complex_query = "向量数据库和RAG技术有什么区别？"
    need_decompose2, sub_questions2 = should_decompose_query(complex_query)
    result2 = execute_corrective_rag_agent(complex_query, sub_questions2)
    
    return [result1, result2]

def main():
    """主函数 - 演示完整的Corrective RAG Agent功能"""
    print("🚀 启动Corrective RAG Agent演示")
    print("=" * 60)
    
    try:
        # 1. 测试查询分解功能
        test_decomposition_functionality()
        
        print("\n" + "=" * 60)
        
        # 2. 测试Agent执行功能
        test_results = test_agent_execution()
        
        print("\n" + "=" * 60)
        print("✅ 演示完成！")
        print("📊 测试结果摘要:")
        for i, result in enumerate(test_results, 1):
            status = "✅ 成功" if result.get("success") else "❌ 失败"
            print(f"   测试 {i}: {status}")
            if result.get("success"):
                print(f"     - 查询: {result.get('query')}")
                print(f"     - 子问题数: {len(result.get('subquestions', []))}")
                print(f"     - 使用爬虫: {'是' if result.get('used_web_fallback') else '否'}")
        
    except Exception as e:
        print(f"❌ 演示过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()