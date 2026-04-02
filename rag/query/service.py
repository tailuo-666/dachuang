from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .optimizer import AcademicQueryPlanner
from ..schemas import AcademicQueryPlan


class AcademicQueryService:
    """Single-query planning and answer synthesis for academic RAG."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.planner = AcademicQueryPlanner(llm)
        self.answer_prompt = ChatPromptTemplate.from_template(
            """
基于以下上下文信息回答问题。如果上下文不足以回答问题，请明确指出信息不足之处。
上下文：
{context}

问题：{question}

请以学术研究助手的口吻，用中文给出准确、谨慎、结构化的回答：
            """.strip()
        )
        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

    def build_query_plan(self, question: str) -> AcademicQueryPlan:
        return self.planner.build(question)

    def generate_answer(self, context: str, question: str) -> str:
        return self.answer_chain.invoke({"context": context, "question": question})
