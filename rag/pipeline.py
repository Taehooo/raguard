"""
기본 RAG 파이프라인
Retriever + Generator를 연결하는 최상위 인터페이스
"""

from __future__ import annotations

from config import TOP_K
from rag.retriever import DenseRetriever
from rag.generator import Generator


class RAGPipeline:
    def __init__(
        self,
        retriever: DenseRetriever,
        generator: Generator,
        top_k: int = TOP_K,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> dict:
        """
        기본 RAG 실행

        Returns
        -------
        {
            "query"    : 원본 질문,
            "contexts" : 검색된 chunk 리스트 (top_k개),
            "answer"   : LLM 생성 응답,
        }
        """
        contexts = self.retriever.retrieve(query, top_k=self.top_k)
        answer   = self.generator.generate(query, contexts)

        return {
            "query":    query,
            "contexts": contexts,
            "answer":   answer,
        }
