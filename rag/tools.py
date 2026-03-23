import json

# LangChain 核心组件
from langchain_core.tools import tool

import sys
import os
# 导入工具模块
from checkDecomposition import QueryDecomposerAgent
from value import RelevanceGraderTool
# 添加arxiv爬虫相关导入
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))
from arxiv_crawler_integrated import ArxivCrawlerIntegrated

# 添加工具目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
tool_dir = os.path.join(current_dir, "tool")
sys.path.insert(0, tool_dir)

# 模块级依赖（由 api_server.py 启动时注入）
_rag_system = None
_pdf_processor = None
_llm = None
_progress_callback = None

def init_tools(rag_system, pdf_processor, llm):
    """启动时调用，注入全局依赖"""
    global _rag_system, _pdf_processor, _llm
    _rag_system = rag_system
    _pdf_processor = pdf_processor
    _llm = llm
    print("Tools initialized with dependencies")

def set_progress_callback(callback):
    """设置进度回调函数"""
    global _progress_callback
    _progress_callback = callback

def _log(msg):
    """统一的日志输出函数，同时打印到控制台和调用回调"""
    print(msg)
    if _progress_callback:
        _progress_callback(msg)




# --- 1. 定义一个用于管理上下文的类 (用来存子问题问答对) --- 
class ResearchContext: 
    def __init__(self): 
        self.qa_pairs = [] # 存储 [{"question": "...", "answer": "..."}] 
        self.original_query = "" 
        self.sub_questions = [] # 存储子问题列表
        self.papers = [] # 存储检索到的论文列表

    def reset(self): 
        self.qa_pairs = [] 
        self.original_query = "" 
        self.sub_questions = [] # 重置子问题列表
        self.papers = [] # 重置检索到的论文列表

# 初始化全局上下文 
context = ResearchContext() 

# --- 2. 定义 Tools  --- 

@tool 
def query_transform(original_query: str) -> str: 
    """ 
    第一步调用。接收用户原始查询，将其分解为多个独立的子问题列表。 
    """ 
    print(f"\n 正在分解查询: {original_query}") 
    context.reset() 
    context.original_query = original_query 
    
    # 模拟 queryTransform.py 的逻辑 
    # 实际项目中，这里调用 queryTransform.transform(original_query) 
    #return [ 
    #    f"子问题1: {original_query} 的核心概念是什么", 
    #    f"子问题2: {original_query} 的优缺点分析", 
    #   f"子问题3: {original_query} 的未来发展" 
    #] 

    #调用check_decomposition.py
    # 创建分解器实例并调用 route_query 方法
    decomposer = QueryDecomposerAgent(llm_client=_llm)
    # 使用注入的 _llm 实例
    needs_decomposition, reason, sub_questions = decomposer.route_query(original_query)
    
    context.sub_questions = sub_questions # 存储子问题列表

    # 构造带提示信息的返回字符串
    result_info = {
        "status": "success",
        "message": f"查询优化完成，分解成以下{len(sub_questions)}个子问题",
        "sub_questions": sub_questions,
        "next_step": "循环处理每一个子问题"
    }
    return json.dumps(result_info, ensure_ascii=False)

@tool
def retriever(sub_question: str) -> str:
    """
    检索工具。用于在本地向量数据库中搜索相关文档。
    """
    _log(f"正在检索: {sub_question}")
    if _rag_system is None or _rag_system.retriever is None:
        return json.dumps({"status": "error", "message": "RAG系统未初始化", "docs": []}, ensure_ascii=False)
    try:
        docs = _rag_system.retriever.invoke(sub_question)
        context.papers.extend(docs)   # 存入 context 供后续提取 sources
        formatted = [{"content": d.page_content[:500], "source": d.metadata.get("source", "未知")} for d in docs]
        return json.dumps({
            "status": "success",
            "message": f"检索到{len(docs)}个相关文档",
            "docs": formatted,
            "next_step": "调用value_evaluator评估文档相关性"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "docs": []}, ensure_ascii=False) 


@tool 
def value_evaluator(sub_question: str, docs: str) -> str: 
    """ 
    评估工具。评估检索到的文档是否足以回答子问题。 
    返回 'PASS' (无需爬虫) 或 'FAIL' (需要爬虫)。 
    """ 
    print(f"\n 正在评估文档相关性...") 
    
    # 创建相关性评估工具实例
    grader = RelevanceGraderTool()
    
    # 将docs字符串转换为列表（如果需要的话）
    if isinstance(docs, str):
        doc_list = [docs]
    else:
        doc_list = docs if isinstance(docs, list) else [str(docs)]
    
    # 调用value.py中的评估逻辑，使用BERT策略
    result = grader._run(query=sub_question, documents=doc_list, strategy="bert")
    
    # 根据评估结果返回 'PASS' 或 'FAIL'
    # 如果评估结果显示应该使用上下文，则返回 'PASS'，否则返回 'FAIL'
    if result.get("action") == "use_context":
        print(f" 文档评估结果: PASS - 相关性评分: {result.get('score', 0)}")
        return json.dumps({
            "status": "PASS",
            "score": result.get('score', 0),
            "message": "文档相关性评分较高，无需调用爬虫工具。",
            "action": "continue_without_crawl"
        }, ensure_ascii=False)
    else:
        print(f" 文档评估结果: FAIL - 相关性评分: {result.get('score', 0)}")
        return json.dumps({
            "status": "FAIL", 
            "score": result.get('score', 0),
            "message": "文档相关性评分较低，需要调用爬虫工具获取更多信息。",
            "action": "proceed_with_crawl"
        }, ensure_ascii=False) 

@tool
def web_deep_research(sub_question: str) -> str:
    """
    爬虫工具。仅在 value_evaluator 返回 FAIL 时调用。
    从互联网获取文档，提取内容。
    """
    _log(f"正在执行深度爬虫: {sub_question}")

    # 创建爬虫实例
    crawler = ArxivCrawlerIntegrated("./paper_results")

    # 分步执行
    # 1. 爬取论文
    papers = crawler.crawl_papers(sub_question, max_pages=3)
    context.papers = papers # 存储检索到的论文列表

    # 2. 保存到CSV
    crawler.save_to_csv(papers, "ml_papers.csv")

    # 3. 格式化论文
    formatted = crawler.generate_paper_list("ml_papers.csv")
    crawler.save_formatted_papers(formatted, "formatted_ml_papers.txt")

    # 4. 下载论文
    success_count = crawler.download_papers(max_downloads=3)

    # 关键：处理 PDF 并更新向量库
    if success_count > 0 and _pdf_processor and _rag_system:
        _log("正在处理下载的PDF并更新向量库...")
        try:
            processed = _pdf_processor.process_pdf_folder("./paper_results")
            if processed:
                _rag_system.update_rag_system()
                new_docs = _rag_system.retriever.invoke(sub_question)
                context.papers.extend(new_docs)
                formatted_docs = [{"content": d.page_content[:500], "source": d.metadata.get("source", "未知")} for d in new_docs[:3]]
                return json.dumps({
                    "status": "success",
                    "message": f"爬虫完成，下载{success_count}篇论文并更新知识库",
                    "docs": formatted_docs,
                    "next_step": "基于新文档回答子问题"
                }, ensure_ascii=False)
        except Exception as e:
            _log(f"PDF处理失败: {e}")

    return json.dumps({
        "status": "partial_success",
        "message": f"爬虫完成，找到{len(papers)}篇论文，但PDF处理未执行",
        "papers_found": len(papers)
    }, ensure_ascii=False) 

@tool 
def save_sub_task_result(sub_question: str, answer: str) -> str: 
    """ 
    【关键工具】保存工具。 
    当你为一个子问题生成了满意的答案后，必须调用此工具进行保存。 
    """ 
    print(f"\n 保存子问题结果: {sub_question[:10]}...") 
    context.qa_pairs.append({"question": sub_question, "answer": answer}) 
  # return f"成功保存。当前已完成 {len(context.qa_pairs)} 个子问题。还剩下 {len(context.sub_questions) - len(context.qa_pairs)} 个子问题待处理。下一个子问题是：{context.sub_questions[len(context.qa_pairs)]}" 
    remaining_count = len(context.sub_questions) - len(context.qa_pairs)
    next_question = context.sub_questions[len(context.qa_pairs)] if len(context.qa_pairs) < len(context.sub_questions) else "所有子问题已完成，可以总结了。"
    return json.dumps({
        "status": "success",
        "completed_count": len(context.qa_pairs),
        "remaining_count": remaining_count,
        "next_question": next_question,
        "message": f"成功保存子问题答案。当前已完成 {len(context.qa_pairs)} 个子问题。"
    }, ensure_ascii=False)
@tool 
def report_generator() -> str: 
    """ 
    总结工具。仅在所有子问题都已保存后调用。 
    它会自动读取后台保存的所有问答对，生成最终报告。 
    """ 
    print(f"\n 正在生成最终报告...") 
    
    # 模拟 report.py 
    if not context.qa_pairs: 
        return json.dumps({
            "status": "error",
            "message": "错误：没有找到任何保存的子问题记录。"
        }, ensure_ascii=False) 
    
    # 构建给总结模型的输入 
    evidence = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in context.qa_pairs]) 
    
    final_prompt = f""" 
    基于原始问题: "{context.original_query}" 
    以及以下子问题研究结果: 
    {evidence} 
    
    生成一份综合回答。 
    """ 

    final_answer = f"{final_prompt}\n\n最终回答：\n(这里是整合后的内容...)"
    return json.dumps({
        "status": "success",
        "report_title": "最终报告",
        "analysis_count": len(context.qa_pairs),
        "original_query": context.original_query,
        "final_answer": final_answer,
        "message": f"基于 {len(context.qa_pairs)} 个维度的详细分析生成的最终报告"
    }, ensure_ascii=False) 
