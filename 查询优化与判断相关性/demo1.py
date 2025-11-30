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
from typing import List, Dict, Any
import re
from datetime import datetime

print("✅ 查询分解库导入完成")

# 定义触发分解的关键词列表
DECOMPOSITION_KEYWORDS = {
    "逻辑关系": ["和", "与", "或", "对比", "差异", "区别", "分别", "谁更", "既", "又", "不仅", "还", "以及", "同时", "相比之下", "相对"],
    "复杂意图": ["为什么", "如何解决", "步骤", "流程", "分析", "总结", "推荐", "论证", "原因", "对策", "优化方案", "怎样", "如何实现", "方法", "策略", "机制", "原理"],
    "多属性/多维度": ["优缺点", "优势劣势", "性能", "价格", "功能", "适用场景", "部署难度", "学习曲线", "成本", "效率", "可靠性", "可扩展性", "兼容性", "用户体验", "维护"]
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
    entities = re.split(r'[，,、\s]+', query)
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
        print("  ✅ 明确'否': 无触发词 + 核心实体≤1个")
        return "no"
    elif has_trigger_keyword and entity_count >= 2:
        print("  🔧 明确'是': 有触发词 + 核心实体≥2个")
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

# ========================================
# 第三部分：文档相关性评估流程
#
# 功能描述:
# 对检索得到的文档进行相关性评估，判断其与子问题的匹配程度。
# ========================================


# 定义数据模型
class GradeDocuments(BaseModel):
    """多维度文档相关性评估"""
    binary_score: str = Field(description="基础相关性：'yes' 或 'no'")
    relevance_score: int = Field(description="相关性评分：1-5分")
    key_topics: List[str] = Field(description="匹配的关键主题")
    missing_aspects: List[str] = Field(description="缺失的重要方面")
    confidence: float = Field(description="评估置信度：0.0-1.0")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 定义多维度文档相关性评估函数
def evaluate_document_relevance(question: str, document: str) -> GradeDocuments:
    """
    多维度评估文档与查询的相关性
    
    Args:
        question: 原始查询问题
        document: 待评估的文档内容
        
    Returns:
        GradeDocuments: 包含多维度评估结果的文档
    """
    print(f"  📊 评估文档相关性...")
    
    # 构建评估提示模板
    evaluation_system_prompt = """你是一个专业的学术文档评估专家。请严格基于文档中的学术术语进行关键主题匹配，避免同义替换导致的偏差（如计算机网络中 “路由协议” 与 “选路协议” 需明确区分），
    请从多个维度评估文档与查询问题的相关性：
    


    评估维度：
    1. 基础相关性：文档是否直接回答了问题
    2. 相关性评分：1-5分（1分：完全不相关，5分：高度相关且信息丰富）
    3. 关键主题匹配：列出文档中与问题相关的主要概念
    4. 缺失方面：指出问题的重要方面但文档未覆盖的
    5. 置信度：对评估结果的自信程度

    评分标准：
    - 5分：文档完整回答问题，包含丰富的细节和深度分析
    - 4分：文档基本回答问题，包含相关信息
    - 3分：文档部分相关，但信息不够充分
    - 2分：文档轻微相关，信息有限
    - 1分：文档与问题基本无关"""

    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", evaluation_system_prompt),
        ("human", "查询问题：{question}\n\n待评估文档：\n{document}")
    ])
    
    # 创建文档评估链
    document_evaluator = evaluation_prompt | structured_llm_grader
    
    try:
        result = document_evaluator.invoke({
            "question": question,
            "document": document
        })
        
        print(f"  ✅ 评估完成 - 相关性评分: {result.relevance_score}/5")
        print(f"  🎯 关键主题: {', '.join(result.key_topics)}")
        print(f"  ❌ 缺失方面: {', '.join(result.missing_aspects) if result.missing_aspects else '无'}")
        print(f"  💪 置信度: {result.confidence:.2f}")
        
        return result
    except Exception as e:
        print(f"  ⚠️  评估出错: {e}")
        # 返回默认值
        return GradeDocuments(
            binary_score="no",
            relevance_score=2,
            key_topics=[],
            missing_aspects=["评估功能异常"],
            confidence=0.5
        )

# 查询重写函数（用于改进检索效果）
def rewrite_query_for_retrieval(original_query: str, missing_aspects: List[str]) -> str:
    """
    基于缺失方面重写查询，提高检索效果
    
    Args:
        original_query: 原始查询
        missing_aspects: 文档评估中发现的缺失方面
        
    Returns:
        str: 重写后的查询
    """
    print(f"  🔄 重写查询以提高检索效果...")
    
    if not missing_aspects:
        return original_query
    
    # 构建查询重写提示
    rewrite_prompt = f"""基于以下信息重写查询，提高检索效果：

    原始查询：{original_query}
    当前检索结果缺失的重要方面：{', '.join(missing_aspects)}

    请生成一个更精确的查询，旨在检索到包含上述缺失信息的文档。查询应该：
    1. 明确包含缺失的关键概念
    2. 保持学术性和准确性
    3. 不超过30个词
    4. 用中文表达

    重写后的查询："""
    
    try:
        response = llm.invoke(rewrite_prompt)
        rewritten_query = response.content.strip()
        print(f"  ✅ 查询重写完成")
        print(f"  📝 原始查询: {original_query}")
        print(f"  🔄 重写后: {rewritten_query}")
        return rewritten_query
    except Exception as e:
        print(f"  ⚠️  查询重写出错: {e}")
        return original_query

# 检索与评估循环
def retrieval_evaluation_loop(question: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    检索与评估循环：最多执行3次检索-评估迭代
    
    Args:
        question: 待处理的问题
        max_iterations: 最大迭代次数
        
    Returns:
        Dict: 包含迭代结果的字典
    """
    print(f"\n🔄 开始检索与评估循环...")
    print(f"🎯 目标问题: {question}")
    print(f"🔄 最大迭代次数: {max_iterations}")
    
    loop_log = {
        "question": question,
        "iterations": [],
        "success": False,
        "final_answer": None,
        "total_iterations": 0,
        "crawler_fallback": False
    }
    
    current_question = question
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- 迭代 {iteration}/{max_iterations} ---")
        loop_log["total_iterations"] = iteration
        
        # 1. 模拟文档检索 TODO: 替换为实际文档检索函数，另外，可以先使用top-k检索，用向量相似筛选，也可以不用，取决于我们有没有那么大的知识库
        print(f"  🔍 模拟文档检索...")
        retrieved_doc = mock_document_retrieval(current_question, iteration)
        # print(f"📄 检索到文档长度: {len(retrieved_doc)} 字符")
        
        # 2. 评估文档相关性
        evaluation = evaluate_document_relevance(current_question, retrieved_doc)
        
        # 3. 记录迭代信息
        iteration_info = {
            "iteration": iteration,
            "question_used": current_question,
            "document_preview": retrieved_doc[:100] + "...",
            "document_full": retrieved_doc,  # 保存完整文档用于后续选择
            "evaluation": evaluation.dict(),
            "timestamp": datetime.now().isoformat()
        }
        loop_log["iterations"].append(iteration_info)
        
        # 4. 根据评分决定下一步操作
        if evaluation.relevance_score >= 4:
            print(f"✅ 文档质量优秀 (评分: {evaluation.relevance_score}/5)")
            print(f"🎯 接受当前文档，生成答案...")
            
            # 生成答案 TODO: 替换为实际答案生成函数
            answer = generate_answer_with_context(question, retrieved_doc, evaluation)
            loop_log["success"] = True
            loop_log["final_answer"] = answer
            break
            
        elif evaluation.relevance_score >= 3:
            print(f"⚠️  文档质量一般 (评分: {evaluation.relevance_score}/5)")
            print(f"🔄 改写查询进行重检索...")
            
            # 改写查询
            current_question = rewrite_query_for_retrieval(
                question, 
                evaluation.missing_aspects
            )
            
            if iteration == max_iterations:
                print(f"⏰ 达到最大迭代次数，选择历史最佳结果")
                # 选择历史中评分最高的文档
                best_iteration = max(loop_log["iterations"], key=lambda x: x["evaluation"]["relevance_score"])
                best_doc = best_iteration["document_full"]  # 使用完整文档
                best_evaluation = GradeDocuments(**best_iteration["evaluation"])
                print(f"🏆 选择第{best_iteration['iteration']}次迭代的文档(评分: {best_evaluation.relevance_score}/5)")
                # 生成答案 TODO: 替换为实际答案生成函数
                answer = generate_answer_with_context(question, best_doc, best_evaluation)
                loop_log["success"] = True
                loop_log["final_answer"] = answer
                
        else:  # 评分 <= 2
            print(f"❌ 文档质量不佳 (评分: {evaluation.relevance_score}/5)")
            
            if iteration == max_iterations:
                print(f"⏰ 达到最大迭代次数，触发爬虫替代方案获取额外信息")
                # 执行爬虫获取新资料
                # 爬虫这里可以加补充关键词限定范围，在log中可以查看每一次迭代文档的缺失部分。
                crawler_result = crawler_fallback_solution(question, retrieved_doc)
                
                # 将爬虫结果作为额外的迭代记录
                crawler_evaluation = evaluate_document_relevance(question, crawler_result)
                crawler_info = {
                    "iteration": iteration + 1,  # 标记为额外迭代
                    "question_used": question,
                    "document_preview": crawler_result[:100] + "...",
                    "document_full": crawler_result,
                    "evaluation": crawler_evaluation.dict(),
                    "timestamp": datetime.now().isoformat(),
                    "source": "crawler"
                }
                loop_log["iterations"].append(crawler_info)
                loop_log["crawler_fallback"] = True
                loop_log["crawler_result"] = crawler_result
                
                # 在包括爬虫结果的所有文档中选择评分最高的
                best_iteration = max(loop_log["iterations"], key=lambda x: x["evaluation"]["relevance_score"])
                best_doc = best_iteration["document_full"]
                best_evaluation = GradeDocuments(**best_iteration["evaluation"])
                print(f"🏆 选择{('第' + str(best_iteration['iteration']) + '次迭代') if best_iteration.get('source') != 'crawler' else '爬虫'}的文档(评分: {best_evaluation.relevance_score}/5)")
                
                # 基于最佳文档生成答案
                answer = generate_answer_with_context(question, best_doc, best_evaluation)
                loop_log["success"] = True
                loop_log["final_answer"] = answer + "\n\n⚠️ 注意：当前答案结合了检索和爬虫信息，建议参考爬虫替代方案获取更多资料。"
            else:
                print(f"🔄 继续下一次迭代...")
                # 改写查询继续
                current_question = rewrite_query_for_retrieval(
                    question, 
                    evaluation.missing_aspects
                )
    
    return loop_log


f"""
这个环节和之前一样
for(子问题)
    直接调用 retrieval_evaluation_loop即可
整合时，需要手动补上拼接问答对的操作，下面有整合问答对的函数和最终的提示词，仅供参考
"""


# 问答生成提示词模板
# 定义回答生成提示模板
template = """以下是一组与目标问题相关的问答对（学术素材）：
{context}

你的任务是基于上述问答对，合成一个全面、学术严谨的答案。请严格遵循以下整合要求和输出标准，确保答案的准确性、完整性和逻辑性：

## 核心整合要求（必须全部遵守）
1. 去重筛选：识别并删除问答对中重复的表述、冗余解释或重复数据点，仅保留对回答问题有价值的独特信息，不遗漏关键细节；
2. 逻辑组织：按清晰的学术逻辑顺序排列内容（例如“定义→原理→方法→结果→结论”“基础概念→深度分析→实际应用”或“问题背景→核心方案→局限性”），采用层级结构（如“要点+子要点”）提升可读性，避免碎片化；
3. 学术严谨性：
   - 准确引用问答对中的核心观点、数据和专业术语，不扭曲、夸大或简化原意，确保表述与原始素材一致；
   - 技术术语的定义和使用保持统一（若问答对中有统一标准，遵循其表述；若无统一标准，明确标注差异，例如“文中‘路由收敛’存在两种定义：1.XXX（来自问答对2）；2.XXX（来自问答对4）”）；
   - 避免主观假设或无依据推断，所有结论必须有问答对中的信息直接支撑，不添加个人观点；
4. 完整性与连贯性：
   - 覆盖回答问题所需的所有核心维度（如问题涉及“是什么、为什么、怎么做”，需逐一回应；涉及技术细节的，需包含关键参数、步骤或适用条件）；
   - 使用过渡词/短语（如“首先”“此外”“相反”“综上”“进一步而言”）连接不同信息点，确保逻辑流畅，上下文衔接自然；
5. 冲突处理：
   - 若问答对中存在矛盾信息或对立观点：
     1. 优先保留有数据支撑、逻辑严谨或符合主流学术共识的信息（若问答对中隐含相关依据）；
     2. 若无法判定优先级（如双方观点均缺乏充分证据），客观呈现所有冲突视角，标注“争议点”+简要说明（例如“关于该问题存在两种对立观点：观点1……（来自问答对3）；观点2……（来自问答对5）。现有信息中两种观点均未提供充分论证，需进一步验证”）；
     3. 严禁无依据丢弃冲突信息或强制统一结论。

## 输出标准
- 格式：采用正式学术写作风格（避免口语化、网络用语），使用层级结构（如“1. 核心要点→（1）子要点”或“加粗要点+详细解释”），确保清晰易读；
- 完整性检查：合成后需确认未遗漏问答对中的关键信息（与目标问题无关的内容可剔除）；
- 补充说明：若问答对缺乏回答问题必需的信息（如核心原理、关键数据、关键步骤或适用场景），在结尾明确标注：“注：现有问答对未覆盖「具体缺失方面（例如‘算法的数学推导’‘不同条件下的实验结果’或‘实际应用案例’）」，因此答案可能不够全面，需补充相关素材进一步完善”；
- 禁止外部信息：不得引入问答对之外的知识、数据或观点，严格基于给定语境合成，不泄露未提及的学术内容。

请按照上述要求，合成以下问题的答案：
{question}

整合后的完整答案：
"""




# 定义格式化问答对的函数
def format_qa_pair(question, answer):
    """格式化问答对为字符串形式"""
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

#=====================================================================================================================================


