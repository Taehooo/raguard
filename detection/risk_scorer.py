"""
RAGuard 탐지 모듈 - 4가지 신호 기반 위험도 점수 계산

신호 구성:
  (1) instructionality_score   : 지시형 패턴 감지 (가중치 0.35)
  (2) refusal_score            : 거부 유도 표현 감지 (가중치 0.30)
  (3) ppl_outlier_score        : 정상 코퍼스 대비 어휘 이상치 (가중치 0.20)
                                 (신경망 불필요 — 휴리스틱 기반)
  (4) relevance_mismatch_score : 검색 유사도 높으나 factual 정보 밀도 낮음 (가중치 0.15)

risk = 0.35*inst + 0.30*refusal + 0.20*ppl + 0.15*mismatch
"""

from __future__ import annotations

import re
import math
import statistics
from collections import Counter

from config import (
    WEIGHT_INSTRUCTIONALITY,
    WEIGHT_REFUSAL,
    WEIGHT_PPL_OUTLIER,
    WEIGHT_RELEVANCE_MISMATCH,
    INSTRUCTION_KEYWORDS,
    REFUSAL_KEYWORDS,
    FACTUAL_ANCHOR_PATTERNS,
)


class RiskScorer:
    def __init__(self):
        # 정상 코퍼스 어휘 통계 (calibrate()에서 설정)
        self._corpus_ttr_mean = 0.6     # type-token ratio 평균
        self._corpus_ttr_std  = 0.1
        self._corpus_len_mean = 100.0   # 문서 길이(단어 수) 평균
        self._corpus_len_std  = 50.0
        self._corpus_vocab: set[str] = set()

    # ── 공개 API ────────────────────────────────────────────────────────────

    def calibrate(self, corpus_documents: list[dict]) -> None:
        """정상 코퍼스로 통계 보정 (최초 1회 호출 권장)"""
        ttrs = []
        lens = []
        for doc in corpus_documents:
            tokens = self._tokenize(doc["content"])
            lens.append(len(tokens))
            ttrs.append(len(set(tokens)) / max(len(tokens), 1))
            self._corpus_vocab.update(tokens)

        self._corpus_len_mean = statistics.mean(lens)
        self._corpus_len_std  = statistics.stdev(lens) if len(lens) > 1 else 1.0
        self._corpus_ttr_mean = statistics.mean(ttrs)
        self._corpus_ttr_std  = statistics.stdev(ttrs) if len(ttrs) > 1 else 0.01

        print(
            f"[RiskScorer] 보정 완료 — "
            f"len: {self._corpus_len_mean:.0f}±{self._corpus_len_std:.0f}, "
            f"TTR: {self._corpus_ttr_mean:.3f}±{self._corpus_ttr_std:.3f}, "
            f"vocab: {len(self._corpus_vocab)}개"
        )

    def score(
        self,
        chunk: dict,
        query: str,
        retrieval_score: float = 0.0,
    ) -> dict:
        text = chunk["content"]

        inst     = self._instructionality_score(text)
        refusal  = self._refusal_score(text)
        ppl_out  = self._ppl_outlier_score(text)
        mismatch = self._relevance_mismatch_score(text, retrieval_score)

        risk = (
            WEIGHT_INSTRUCTIONALITY    * inst
            + WEIGHT_REFUSAL           * refusal
            + WEIGHT_PPL_OUTLIER       * ppl_out
            + WEIGHT_RELEVANCE_MISMATCH * mismatch
        )
        risk = min(1.0, max(0.0, risk))

        return {
            "risk":               risk,
            "instructionality":   round(inst,     4),
            "refusal":            round(refusal,  4),
            "ppl_outlier":        round(ppl_out,  4),
            "relevance_mismatch": round(mismatch, 4),
        }

    def score_batch(self, chunks: list[dict], query: str) -> list[dict]:
        scored = []
        for chunk in chunks:
            retrieval_score = chunk.get("score", 0.0)
            detail = self.score(chunk, query, retrieval_score)
            enriched = dict(chunk)
            enriched["risk_detail"] = detail
            enriched["risk"] = detail["risk"]
            scored.append(enriched)
        return scored

    # ── 개별 신호 계산 ────────────────────────────────────────────────────

    def _instructionality_score(self, text: str) -> float:
        """지시형 키워드 밀도"""
        text_lower = text.lower()
        hit = sum(1 for kw in INSTRUCTION_KEYWORDS if kw.lower() in text_lower)
        return min(hit / 5.0, 1.0)

    def _refusal_score(self, text: str) -> float:
        """거부 유도 키워드 밀도"""
        text_lower = text.lower()
        hit = sum(1 for kw in REFUSAL_KEYWORDS if kw.lower() in text_lower)
        return min(hit / 3.0, 1.0)

    def _ppl_outlier_score(self, text: str) -> float:
        """
        휴리스틱 PPL 대용 지표 (신경망 불필요)
        두 가지 신호의 평균:
          - 문서 길이 z-score: 비정상적으로 짧거나 긴 문서
          - OOV 비율: 정상 코퍼스 어휘에 없는 단어 비율
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.5

        # 1) 문서 길이 이상치 (z-score → sigmoid)
        z_len = (len(tokens) - self._corpus_len_mean) / max(self._corpus_len_std, 1)
        len_score = 1 / (1 + math.exp(-abs(z_len) + 1))  # 중심에서 멀수록 1에 가까움

        # 2) OOV 비율 (정상 코퍼스에 없는 단어 비율)
        if self._corpus_vocab:
            oov = sum(1 for t in tokens if t not in self._corpus_vocab)
            oov_ratio = oov / len(tokens)
        else:
            oov_ratio = 0.0

        return min((len_score + oov_ratio) / 2.0, 1.0)

    def _relevance_mismatch_score(
        self, text: str, retrieval_score: float
    ) -> float:
        """검색 유사도는 높지만 factual anchor 부족"""
        if retrieval_score < 0.3:
            return 0.0

        anchor_count = sum(
            1 for pat in FACTUAL_ANCHOR_PATTERNS
            if re.search(pat, text)
        )
        factual_density = min(anchor_count / 3.0, 1.0)
        return min(retrieval_score * (1.0 - factual_density), 1.0)

    # ── 내부 유틸 ──────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """간단한 공백/구두점 기반 토크나이징"""
        return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())
