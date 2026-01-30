import logging
import time
from typing import List, Dict, Union, Literal
from pydantic import BaseModel, Field
import re

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RelevanceGraderInput(BaseModel):
    """文档相关性评估工具的输入参数Schema"""
    
    query: str = Field(
        ..., 
        description="用户的原始查询问题，例如：'什么是向量数据库？'"
    )
    documents: List[str] = Field(
        ..., 
        description="检索到的文档内容列表，每个元素为一个文档的完整文本"
    )
    strategy: Literal["llm", "bert"] = Field(
        "llm", 
        description="评估策略：'llm' 使用大语言模型评估，'bert' 使用BERT模型计算余弦相似度"
    )


class RelevanceGraderTool(BaseTool):
    """
    文档相关性评估工具
    
    该工具用于评估检索到的文档与用户查询问题的相关性，帮助AI Agent决定
    是使用现有文档上下文还是调用爬虫获取更多信息。
    
    功能特点：
    - 支持两种评估策略：LLM评估和BERT语义相似度评估
    - 提供同步和异步调用方式
    - 自动处理空文档和异常情况
    - 返回结构化的评估结果
    
    使用场景：
    - RAG系统中判断检索结果质量
    - 智能问答系统中决定是否需要补充信息
    - 文档检索后的质量过滤
    """
    
    name: str = "document_relevance_grader"
    description: str = """评估检索文档与问题的相关性，帮助决定是否需要获取更多信息。

该工具接收用户查询和检索到的文档列表，使用指定的策略评估文档相关性：
- 如果文档相关且包含足够信息，返回'use_context'，建议使用现有文档
- 如果文档不相关或信息不足，返回'call_crawler'，建议调用爬虫获取更多信息

支持两种评估策略：
1. 'llm': 使用大语言模型进行深度语义理解评估（精度高，速度较慢）
2. 'bert': 使用BERT模型计算余弦相似度（速度快，适合实时场景）

返回结果包含：
- action: 'use_context' 或 'call_crawler'
- score: 0-1之间的相关性评分
- reason: 决策原因说明
- doc_count: 评估的文档数量
"""
    args_schema: type[BaseModel] = RelevanceGraderInput
    
    # 内部组件
    llm: ChatOpenAI = None
    embedding_model: HuggingFaceEmbeddings = None
    similarity_threshold: float = 0.75

    def __init__(self, **kwargs):
        """
        初始化文档相关性评估工具
        
        Args:
            **kwargs: 传递给父类的额外参数
        """
        super().__init__(**kwargs)
        # 初始化 LLM (用于LLM评估策略)
        # self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 初始化 Embedding (用于BERT评估策略，延迟加载以节省资源)
        # self.embedding_model = HuggingFaceEmbeddings(model_name="m3e-base")

    def _run(self, query: str, documents: List[str], strategy: str = "bert") -> Dict:
        """
        同步执行文档相关性评估
        
        Args:
            query: 用户的原始查询问题
            documents: 检索到的文档内容列表
            strategy: 评估策略，'llm' 或 'bert'
            
        Returns:
            Dict: 评估结果字典，包含以下字段：
                - action: 'use_context' 或 'call_crawler'
                - score: 相关性评分（0-1）
                - reason: 决策原因
                - latency: 评估耗时（秒）
                - doc_count: 评估的文档数量
                
        Example:
            >>> grader = RelevanceGraderTool()
            >>> result = grader._run(
            ...     query="什么是向量数据库？",
            ...     documents=["向量数据库是一种专门用于存储和检索向量数据的数据库..."],
            ...     strategy="bert"
            ... )
            >>> print(result['action'])
            'use_context'
        """
        start_time = time.time()
        logger.info(f"开始评估，策略: {strategy}, 文档数量: {len(documents)}")

        # 1. 检查空数据 (Pre-check)
        if not documents:
            logger.warning("检索结果为空，直接触发爬虫。")
            return {"action": "call_crawler", "reason": "empty_results", "score": 0.0}

        # 2. 执行评估
        try:
            if strategy == "llm":
                result = self._evaluate_with_llm(query, documents)
            elif strategy == "bert":
                result = self._evaluate_with_bert(query, documents)
            else:
                raise ValueError(f"不支持的策略: {strategy}")
            
            elapsed = time.time() - start_time
            logger.info(f"评估完成，耗时: {elapsed:.4f}s, 结果: {result['action']}")
            
            return {
                **result,
                "latency": elapsed,
                "doc_count": len(documents)
            }
            
        except Exception as e:
            logger.error(f"评估过程发生异常: {str(e)}")
            # 发生异常时的兜底策略：通常为了保险起见，可以触发爬虫或返回错误
            return {"action": "call_crawler", "reason": f"error: {str(e)}", "score": 0.0}

    async def _arun(self, query: str, documents: List[str], strategy: str = "bert") -> Dict:
        """
        异步执行文档相关性评估
        
        Args:
            query: 用户的原始查询问题
            documents: 检索到的文档内容列表
            strategy: 评估策略，'llm' 或 'bert'
            
        Returns:
            Dict: 评估结果字典，格式与 _run 方法相同
            
        Note:
            当前实现为同步方法的包装，未来可以优化为真正的异步实现
        """
        return self._run(query, documents, strategy)

    def _evaluate_with_llm(self, query: str, docs: List[str]) -> Dict:
        """
        使用大语言模型评估文档相关性
        
        该方法通过构建详细的提示词，让LLM理解查询和文档内容，
        然后判断文档是否包含回答查询所需的关键信息。
        
        Args:
            query: 用户的原始查询问题
            docs: 检索到的文档列表
            
        Returns:
            Dict: 包含 action、score、reason 的评估结果字典
            
        Note:
            需要先初始化 self.llm 才能使用此方法
        """
        # 拼接文档内容，为了节省Token，可以截断
        context_text = "\n\n".join([f"Doc {i+1}: {doc[:500]}..." for i, doc in enumerate(docs)])
        
        prompt = ChatPromptTemplate.from_template(
            """你是一个严格的文档相关性评分员。
            
            用户问题: {query}
            
            检索到的文档集:
            {context}
            
            请评估上述文档集是否包含回答用户问题所需的关键信息。
            - 只要有一篇文档包含核心答案，即视为"yes"。
            - 如果所有文档都偏题或无法回答，视为"no"。
            
            请仅以JSON格式返回结果，格式如下:
            {{
                "score": <0到1之间的浮点数评分>,
                "reason": "<简短理由>",
                "decision": "<'yes' or 'no'>"
            }}
            """
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        response = chain.invoke({"query": query, "context": context_text})
        
        if response.get("decision") == "yes":
            return {"action": "use_context", "score": response.get("score"), "reason": response.get("reason")}
        else:
            return {"action": "call_crawler", "score": response.get("score"), "reason": response.get("reason")}

    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本，提取关键信息
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 如果文本过长，可以考虑只保留关键部分
        # 这里暂时不做截断，可根据需要调整
        
        return text

    def _evaluate_with_bert(self, query: str, docs: List[str]) -> Dict:
        """
        使用BERT模型和余弦相似度评估文档相关性
        
        该方法通过以下步骤评估相关性：
        1. 将查询和文档转换为向量表示
        2. 计算查询向量与每个文档向量的余弦相似度
        3. 取最大相似度作为最终评分
        4. 与阈值比较，决定是否使用现有文档
        
        Args:
            query: 用户的原始查询问题
            docs: 检索到的文档列表
            
        Returns:
            Dict: 包含 action、score、reason 的评估结果字典
            
        Note:
            - 使用 sentence-transformers/all-MiniLM-L6-v2 模型
            - 相似度阈值默认为 0.75
            - 首次调用时会加载模型，后续调用会复用
        """
        # 延迟加载Embedding模型
        if not self.embedding_model:
            logger.info("正在加载Embedding模型...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        if docs is None or len(docs) == 0:
            return {"action": "call_crawler", "score": 0.0, "reason": "no_docs"}

        # 预处理查询和文档
        processed_query = self._preprocess_text(query)
        processed_docs = [self._preprocess_text(doc) for doc in docs]
        
        # 计算 Query 向量
        query_emb = self.embedding_model.embed_query(processed_query)
        
        # 计算 Docs 向量
        doc_embs = self.embedding_model.embed_documents(processed_docs)
        
        # 计算相似度矩阵
        scores = cosine_similarity([query_emb], doc_embs)[0]
        max_score = float(np.max(scores))
        
        logger.info(f"BERT最大相似度得分: {max_score}")
        
        # 改进的阈值判断逻辑
        # 如果相似度很高，直接使用上下文
        if max_score >= self.similarity_threshold:
            return {
                "action": "use_context", 
                "score": max_score, 
                "reason": "semantic_similarity_pass"
            }
        # 如果相似度较低但高于0.6，可以考虑进一步判断
        elif max_score >= 0.6:
            # 检查是否有一定的语义相关性
            avg_score = float(np.mean(scores))
            # 如果平均分也较高，说明文档整体与查询有一定相关性
            if avg_score >= 0.5:
                return {
                    "action": "use_context", 
                    "score": max_score, 
                    "reason": "moderate_similarity_with_context"
                }
            else:
                return {
                    "action": "call_crawler", 
                    "score": max_score, 
                    "reason": "low_similarity"
                }
        else:
            return {
                "action": "call_crawler", 
                "score": max_score, 
                "reason": "low_similarity"
            }


# 使用示例
if __name__ == "__main__":
    # 测试代码
    pass