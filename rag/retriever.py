"""
FAISS 기반 dense retriever
임베딩: Ollama bge-m3 (OpenAI 호환 /v1/embeddings 엔드포인트 사용)
"""

from __future__ import annotations

import numpy as np
import faiss
from openai import OpenAI
from typing import Optional

from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


class DenseRetriever:
    def __init__(
        self,
        model: str = OLLAMA_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        print(f"[Retriever] 임베딩 모델: {model} (Ollama)")
        self.model = model
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: list[dict] = []

    # ── 인덱스 구축 ────────────────────────────────────────────────────────

    def build_index(self, documents: list[dict]) -> None:
        self.documents = list(documents)
        texts = [d["content"] for d in self.documents]

        print(f"[Retriever] {len(texts)}개 문서 임베딩 중...")
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"[Retriever] 인덱스 구축 완료 (dim={dim})")

    def add_documents(self, documents: list[dict]) -> None:
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
        if self.index is None:
            raise RuntimeError("인덱스가 구축되지 않았습니다.")

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

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self._encode([text_a])
        emb_b = self._encode([text_b])
        return float(np.dot(emb_a[0], emb_b[0]))

    # ── 내부 유틸 ──────────────────────────────────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Ollama /v1/embeddings 호출 후 L2 정규화"""
        # Ollama 임베딩은 배치 단위로 처리
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        vecs = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )
        # L2 정규화 (코사인 유사도 → inner product로 계산)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return vecs / norms
