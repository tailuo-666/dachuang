from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
     model="qwen-plus",
    api_key="sk-e4b7b6386950428bb71c658d47da47ef",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



# 查询重写函数（用于改进检索效果）
def rewrite_query_for_retrieval(original_query: str, missing_aspects: list[str]) -> str:
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

def demonstrate_query_rewrite_example():
    """
    演示查询重写过程的示例函数
    """
    print("\n=== 查询重写示例 ===")
    
    # 示例场景：用户询问关于LLM自主智能体的问题
    original_query = "LLM自主智能体的工作原理是什么？"
    missing_aspects = ["具体组件构成", "决策流程", "应用场景"]
    
    print(f"原始查询: {original_query}")
    print(f"缺失方面: {', '.join(missing_aspects)}")
    
    # 模拟重写过程
    rewritten_query = f"{original_query} 包含{', '.join(missing_aspects)}"
    print(f"重写后查询: {rewritten_query}")
    
    print("\n这样重写的目的是：")
    print("1. 明确要求包含缺失的关键信息（组件构成、决策流程、应用场景）")
    print("2. 提高检索系统的召回率，获取更全面的信息")
    print("3. 避免检索到只介绍基本概念而缺乏深度的文档")

demonstrate_query_rewrite_example()