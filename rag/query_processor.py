# query_processor.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser



class GradeDocuments(BaseModel):
    """文档相关性评估模型"""
    binary_score: str = Field(
        description="文档是否与问题相关，'yes' 或 'no'"
    )


class QueryProcessor:
    def __init__(self, llm):
        self.llm = llm
        self.setup_components()

    def setup_components(self):
        """初始化所有处理组件"""
        # 1. 文档相关性评估器
        self.setup_document_grader()

        # 2. 查询分解器
        self.setup_query_decomposer()

        # 3. 回答生成器
        self.setup_answer_generator()

    def setup_document_grader(self):
        """设置文档相关性评估链"""
        system_prompt = """你是一个评估检索文档与用户问题相关性的评分器。
        如果文档包含与问题相关的关键词或语义含义，请将其评为相关。
        给出二元分数 'yes' 或 'no' 来指示文档是否与问题相关。"""

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "检索到的文档：\n\n{document}\n\n用户问题：{question}"),
        ])

        # 使用本地模型进行结构化输出（需要适配）
        self.retrieval_grader = grade_prompt | self.llm | StrOutputParser()

    def setup_query_decomposer(self):
        """设置查询分解链"""
        decomposition_template = """你是一个有用的助手，可以生成与输入问题相关的多个子问题。
        目标是将输入分解为可以独立回答的一组子问题/子查询。
        生成与以下内容相关的多个搜索查询：{question}
        输出（3个查询）："""

        prompt_decomposition = ChatPromptTemplate.from_template(decomposition_template)

        self.query_decomposer = (
                prompt_decomposition
                | self.llm
                | StrOutputParser()
                | (lambda x: [q.strip() for q in x.split("\n") if q.strip()][:3])
        )

    def setup_answer_generator(self):
        """设置回答生成链"""
        self.answer_prompt = ChatPromptTemplate.from_template("""
基于以下上下文信息回答问题。如果上下文不足以回答问题，请说明哪些信息缺失。

上下文：
{context}

问题：{question}

请提供准确、详细的回答：""")

        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

    def decompose_query(self, question):
        """分解复杂查询为子问题"""
        try:
            return self.query_decomposer.invoke({"question": question})
        except Exception as e:
            print(f"查询分解失败: {e}")
            return [question]  # 失败时返回原始问题

    def grade_document(self, document, question):
        """评估文档相关性"""
        try:
            response = self.retrieval_grader.invoke({
                "document": document,
                "question": question
            })
            # 解析响应判断相关性
            return "yes" in response.lower()
        except Exception as e:
            print(f"文档评估失败: {e}")
            return True  # 失败时默认相关

    def generate_answer(self, context, question):
        """生成最终回答"""
        return self.answer_chain.invoke({
            "context": context,
            "question": question
        })