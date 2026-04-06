"""
RAGuard 탐지 모듈 - 4가지 신호 기반 위험도 점수 계산

신호 구성:
  (1) instructionality_score  : 지시형 패턴 감지 (가중치 0.35)
  (2) refusal_score           : 거부 유도 표현 감지 (가중치 0.30)
  (3) ppl_outlier_score       : 정상 코퍼스 대비 PPL 이상치 (가중치 0.20)
  (4) relevance_mismatch_score: 검색 유사도 높으나 factual 정보 밀도 낮음 (가중치 0.15)

risk = 0.35*inst + 0.30*refusal + 0.20*ppl + 0.15*mismatch
"""

from __future__ import annotations

import re
import math
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    PPL_MODEL,
    WEIGHT_INSTRUCTIONALITY,
    WEIGHT_REFUSAL,
    WEIGHT_PPL_OUTLIER,
    WEIGHT_RELEVANCE_MISMATCH,
    INSTRUCTION_KEYWORDS,
    REFUSAL_KEYWORDS,
    FACTUAL_ANCHOR_PATTERNS,
    PPL_CORPUS_MEAN,
    PPL_CORPUS_STD,
)


class RiskScorer:
    def __init__(self, ppl_model_name: str = PPL_MODEL):
        print(f"[RiskScorer] PPL 모델 로드 중: {ppl_model_name}")
        self._ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
        self._ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_name)
        self._ppl_model.eval()

        # 정상 코퍼스 PPL 통계 (calibrate()로 업데이트)
        self._corpus_ppl_mean = PPL_CORPUS_MEAN
        self._corpus_ppl_std  = PPL_CORPUS_STD

    # ── 공개 API ────────────────────────────────────────────────────────────

    def calibrate(self, corpus_documents: list[dict]) -> None:
        """
        정상 코퍼스로 PPL 통계 보정
        (최초 인덱스 구축 후 1회 호출 권장)
        """
        ppls = [self._compute_ppl(d["content"]) for d in corpus_documents]
        self._corpus_ppl_mean = float(sum(ppls) / len(ppls))
        variance = sum((p - self._corpus_ppl_mean) ** 2 for p in ppls) / len(ppls)
        self._corpus_ppl_std = float(math.sqrt(variance)) + 1e-6
        print(
            f"[RiskScorer] PPL 보정 완료 "
            f"(mean={self._corpus_ppl_mean:.1f}, std={self._corpus_ppl_std:.1f})"
        )

    def score(
        self,
        chunk: dict,
        query: str,
        retrieval_score: float = 0.0,
    ) -> dict:
        """
        단일 chunk의 위험도 점수 계산

        Parameters
        ----------
        chunk            : {"content": str, ...}
        query            : 사용자 질문
        retrieval_score  : retriever 유사도 점수 (0~1)

        Returns
        -------
        {
            "risk"             : float,   # 최종 가중 합산 (0~1)
            "instructionality" : float,
            "refusal"          : float,
            "ppl_outlier"      : float,
            "relevance_mismatch": float,
        }
        """
        text = chunk["content"]

        inst    = self._instructionality_score(text)
        refusal = self._refusal_score(text)
        ppl_out = self._ppl_outlier_score(text)
        mismatch = self._relevance_mismatch_score(text, retrieval_score)

        risk = (
            WEIGHT_INSTRUCTIONALITY   * inst
            + WEIGHT_REFUSAL          * refusal
            + WEIGHT_PPL_OUTLIER      * ppl_out
            + WEIGHT_RELEVANCE_MISMATCH * mismatch
        )
        risk = min(1.0, max(0.0, risk))

        return {
            "risk":               risk,
            "instructionality":   inst,
            "refusal":            refusal,
            "ppl_outlier":        ppl_out,
            "relevance_mismatch": mismatch,
        }

    def score_batch(
        self,
        chunks: list[dict],
        query: str,
    ) -> list[dict]:
        """
        chunk 목록 전체에 점수 계산, 결과를 chunk에 병합하여 반환

        Returns
        -------
        각 chunk에 "risk_detail" 키가 추가된 리스트
        """
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
        """
        텍스트가 '지시형' 문서인지 측정
        규칙 기반: 지시형 키워드 밀도
        """
        text_lower = text.lower()
        hit = sum(1 for kw in INSTRUCTION_KEYWORDS if kw.lower() in text_lower)
        # 최대 5개 히트를 1.0으로 정규화
        return min(hit / 5.0, 1.0)

    def _refusal_score(self, text: str) -> float:
        """
        텍스트가 LLM 거부를 유도하는지 측정
        규칙 기반: 거부 유도 키워드 밀도
        """
        text_lower = text.lower()
        hit = sum(1 for kw in REFUSAL_KEYWORDS if kw.lower() in text_lower)
        return min(hit / 3.0, 1.0)

    def _ppl_outlier_score(self, text: str) -> float:
        """
        정상 코퍼스 대비 PPL 이상치 점수
        z-score를 시그모이드로 변환
        """
        ppl = self._compute_ppl(text)
        z = (ppl - self._corpus_ppl_mean) / self._corpus_ppl_std
        # z > 0: 정상보다 PPL이 높음(부자연스러움) → 위험도 상승
        return float(1 / (1 + math.exp(-z)))

    def _relevance_mismatch_score(
        self, text: str, retrieval_score: float
    ) -> float:
        """
        retriever 유사도는 높은데 factual 정보 밀도가 낮은 chunk 탐지
        factual anchor(날짜, 수치, 인명 등)의 부재를 penalty로 사용
        """
        if retrieval_score < 0.3:
            # 유사도 자체가 낮으면 mismatch 아님
            return 0.0

        anchor_count = sum(
            1 for pat in FACTUAL_ANCHOR_PATTERNS
            if re.search(pat, text)
        )
        # factual anchor가 적을수록 mismatch 가능성 높음
        factual_density = min(anchor_count / 3.0, 1.0)
        mismatch = retrieval_score * (1.0 - factual_density)
        return min(mismatch, 1.0)

    # ── PPL 계산 ──────────────────────────────────────────────────────────

    def _compute_ppl(self, text: str, max_length: int = 512) -> float:
        """GPT-2 기반 perplexity 계산"""
        encodings = self._ppl_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings["input_ids"]

        if input_ids.shape[1] < 2:
            return self._corpus_ppl_mean  # 너무 짧은 텍스트

        with torch.no_grad():
            outputs = self._ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss

        return float(math.exp(loss.item()))
