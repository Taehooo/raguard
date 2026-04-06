"""
FAISS 기반 dense retriever
임베딩 모델: BAAI/bge-m3 (한국어 포함 다국어 지원)
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Optional

from config import RETRIEVER_MODEL


class DenseRetriever:
    def __init__(self, model_name: str = RETRIEVER_MODEL):
        print(f"[Retriever] 임베딩 모델 로드 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: list[dict] = []

    # ── 인덱스 구축 ────────────────────────────────────────────────────────

    def build_index(self, documents: list[dict]) -> None:
        """
        문서 리스트로 FAISS 인덱스 구축

        Parameters
        ----------
        documents : [{"id": ..., "content": ..., ...}, ...]
        """
        self.documents = list(documents)
        texts = [d["content"] for d in self.documents]

        print(f"[Retriever] {len(texts)}개 문서 임베딩 중...")
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner Product (코사인 유사도)
        self.index.add(embeddings)
        print(f"[Retriever] 인덱스 구축 완료 (dim={dim})")

    def add_documents(self, documents: list[dict]) -> None:
        """기존 인덱스에 문서 추가 (공격 문서 주입에 사용)"""
        if self.index is None:
            self.build_index(documents)
            return

        new_texts = [d["content"] for d in documents]
        embeddings = self._encode(new_texts)
        self.index.add(embeddings)
        self.documents.extend(documents)
        print(f"[Retriever] {len(documents)}개 문서 추가 (총 {len(self.documents)}개)")

    # ── 검색 ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        쿼리에 대해 top_k개의 유사 문서 반환

        Returns
        -------
        list of {"score": float, **document_fields}
        """
        if self.index is None:
            raise RuntimeError("인덱스가 구축되지 않았습니다. build_index()를 먼저 호출하세요.")

        q_emb = self._encode([query])
        scores, indices = self.index.search(q_emb, min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = dict(self.documents[idx])
            doc["score"] = float(score)
            results.append(doc)

        return results

    def get_embedding(self, text: str) -> np.ndarray:
        """단일 텍스트의 임베딩 벡터 반환"""
        return self._encode([text])[0]

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """두 텍스트의 코사인 유사도 반환"""
        emb_a = self._encode([text_a])
        emb_b = self._encode([text_b])
        return float(np.dot(emb_a[0], emb_b[0]))

    # ── 내부 유틸 ──────────────────────────────────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        """L2 정규화된 임베딩 반환 (코사인 유사도를 inner product로 계산하기 위함)"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return np.array(embeddings, dtype=np.float32)
