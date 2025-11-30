# ========================================
# RAG查询转换与评估演示程序
#
# 本脚本演示了RAG系统中的核心组件：
# 1. 文档相关性评估 - 使用LLM评估检索文档与查询的相关性
# 2. 查询分解 - 将复杂问题分解为多个子问题
# 3. RAG处理链 - 构建完整的检索-生成流程
#
# 系统要求:
# - LangChain框架
# - OpenAI API访问
# - 适当的向量数据库和检索器
# ========================================

# ========================================
# 第一部分：文档相关性评估组件
#
# 功能描述:
# 构建一个评估链，用于判断检索到的文档与用户查询的相关性。
# ========================================

# 导入必要的库
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

print("✅ 基础库导入完成")

# 定义数据模型
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

print("✅ 数据模型定义完成")
print(f"📋 模型字段: {list(GradeDocuments.__fields__.keys())}")

# 初始化LLM模型用于相关性评估
# TODO：改为我们使用的模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
     model="qwen-plus",
    api_key="sk-e4b7b6386950428bb71c658d47da47ef",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


print("✅ 相关性评估链构建完成")
print("🔗 链结构: 提示模板 -> 结构化输出模型")

# ========================================
# 第二部分：查询分解组件
#
# 功能描述:
# 将复杂用户问题分解为多个可独立回答的子问题，提高检索效率。
# ========================================

# ========================================
# 2.1 基础查询分解功能
# ========================================

# 导入查询分解相关库
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import List
import re

print("✅ 查询分解库导入完成")

# 定义触发分解的关键词列表
DECOMPOSITION_KEYWORDS = {
    "逻辑关系": ["和", "与", "或", "对比", "差异", "区别", "分别", "谁更", "既", "又", "不仅", "还"],
    "复杂意图": ["为什么", "如何解决", "步骤", "流程", "分析", "总结", "推荐", "论证", "原因", "对策", "优化方案","怎么办","怎么搞"],
    "多属性/多维度": ["优缺点", "优势劣势", "性能", "价格", "功能", "适用场景", "部署难度", "学习曲线"]
}

print("✅ 分解关键词列表定义完成")

# ========================================
# 2.2 规则预判断功能
# ========================================

def extract_entities(query: str) -> List[str]:
    """
    极简版核心实体提取方法
    在实际应用中可以使用spaCy分词或调用LLM提取
    """
    # 这里使用简单的分词方法作为示例
    # 实际应用中可以替换为更复杂的实体提取方法
    entities = re.split(r'[，,、？\s]+', query)
    # 过滤掉空字符串和长度小于2的字符串
    entities = [entity.strip() for entity in entities if len(entity.strip()) >= 2]
    return entities

def rule_based_prejudgment(query: str) -> str:
    """
    规则预判断函数
    返回值: "yes"(需要分解), "no"(不需要分解), "maybe"(模糊case)
    """
    print(f"🔍 规则预判断: {query}")
    
    # 检查是否包含触发分解的关键词
    has_trigger_keyword = False
    for category, keywords in DECOMPOSITION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query:
                has_trigger_keyword = True
                print(f"  🎯 触发关键词: '{keyword}' (类别: {category})")
                break
        if has_trigger_keyword:
            break
    
    # 提取核心实体并计算数量
    entities = extract_entities(query)
    entity_count = len(entities)
    print(f"  📦 核心实体: {entities} (数量: {entity_count})")
    
    # 规则判断
    if not has_trigger_keyword and entity_count <= 1:
        print("  ✅ 明确'否': 无触发词 and 核心实体≤1个")
        return "no"
    elif has_trigger_keyword and entity_count >= 2:
        print("  🔧 明确'是': 有触发词 and 核心实体≥2个")
        return "yes"
    else:
        print("  🤔 模糊case: 介于两者之间")
        return "maybe"

print("✅ 规则预判断功能定义完成")

# ========================================
# 2.3 LLM轻量分类功能
# ========================================

# 定义LLM轻量分类提示模板
classification_template = """用户查询：{query}
任务：判断该查询是否需要拆分成2个及以上独立子问题，才能完整、准确回答。
要求：1. 仅输出"是"或"否"；2. 无任何额外解释；3. 隐性多意图也需判定为"是"（如"如何优化RAG"需拆成"优化方向+具体方法"）。"""

classification_prompt = ChatPromptTemplate.from_template(classification_template)

def llm_light_classification(query: str) -> str:
    """
    LLM轻量分类函数
    对规则无法判定的查询，用极简Prompt快速分类
    """
    print(f"🧠 LLM轻量分类: {query}")
    
    try:
        # 使用已配置的LLM模型
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
            # 默认处理
            print(f"  ⚠️ 无法识别LLM输出: {result}，默认返回'no'")
            return "no"
    except Exception as e:
        print(f"  ⚠️ LLM分类出错: {e}，默认返回'no'")
        return "no"

print("✅ LLM轻量分类功能定义完成")

# ========================================
# 2.4 子问题生成功能
# ========================================

# 定义子问题生成提示模板
subquestion_template = """用户原始查询：{query}
任务：将其拆分为2-5个独立子问题，需满足：
1. 每个子问题只对应1个信息点，无重叠；
2. 保留原始查询的核心上下文（如时间、主体、场景），不丢失关键约束；
3. 子问题直接可用于检索（无需额外补充信息）；
4. 不生成冗余子问题（如"对比AB"不拆"什么是A""什么是B"）。
输出格式：按1.、2.、3.…编号列出子问题，无其他内容。"""

subquestion_prompt = ChatPromptTemplate.from_template(subquestion_template)

def generate_subquestions(query: str) -> List[str]:
    """
    子问题生成函数
    """
    print(f"🧩 子问题生成: {query}")
    
    try:
        # 使用已配置的LLM模型
        response = llm.invoke(subquestion_prompt.format(query=query))
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # 解析子问题列表
        sub_questions = []
        lines = result.split('\n')
        for line in lines:
            line = line.strip()
            # 匹配编号格式的子问题 (如 "1. 什么是LLM?")
            if re.match(r'^\d+\.\s*', line):
                # 移除编号前缀
                question = re.sub(r'^\d+\.\s*', '', line).strip()
                if question:
                    sub_questions.append(question)
        
        print(f"  ✅ 生成了 {len(sub_questions)} 个子问题:")
        for i, q in enumerate(sub_questions, 1):
            print(f"    {i}. {q}")
            
        return sub_questions
    except Exception as e:
        print(f"  ⚠️ 子问题生成出错: {e}")
        return []

print("✅ 子问题生成功能定义完成")

# ========================================
# 2.5 查询是否需要分解的判断函数
# ========================================

def should_decompose_query(query: str) -> tuple[bool, List[str]]:
    """
    判断查询是否需要分解的主函数
    返回值: (是否需要分解, 子问题列表)
    """
    print("=" * 50)
    print("🔍 开始判断查询是否需要分解")
    print(f"� 原始查询: {query}")
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

print("✅ 查询分解判断主函数定义完成")

# ========================================
# 2.6 测试查询分解功能
# ========================================

# 测试查询分解功能
test_questions = [
    "什么是向量数据库",
    "大模型幻觉的应对方案",
    "RAG检索准确率低怎么办",
    "对比LangChain和LlamaIndex的优缺点，推荐新手用哪个？",
    "2024年A公司和B公司的营收分别是多少？谁更高？"
]

print("\n" + "=" * 60)
print("🧪 测试查询分解功能")
print("=" * 60)

for i, test_query in enumerate(test_questions, 1):
    print(f"\n--- 测试用例 {i} ---")
    need_decompose, sub_questions = should_decompose_query(test_query)
    print(f"📋 最终结果: {'需要分解' if need_decompose else '不需要分解'}")
    if need_decompose and sub_questions:
        print("📄 分解后的子问题:")
        for j, sq in enumerate(sub_questions, 1):
            print(f"  {j}. {sq}")
    print("-" * 30)



    