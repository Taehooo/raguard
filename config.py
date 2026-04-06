import os

# === 모델 설정 ===
RETRIEVER_MODEL = "BAAI/bge-m3"          # 다국어 임베딩 (한국어 지원)
PPL_MODEL       = "gpt2"                  # PPL 계산용 언어 모델
GENERATOR_MODEL = "gpt-4o-mini"           # OpenAI 생성 모델

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# === RAG 설정 ===
TOP_K = 5     # 최종 LLM에 전달할 chunk 수
TOP_N = 20    # 위험도 평가 전 후보 pool 크기

# === Risk Score 가중치 ===
WEIGHT_INSTRUCTIONALITY  = 0.35
WEIGHT_REFUSAL           = 0.30
WEIGHT_PPL_OUTLIER       = 0.20
WEIGHT_RELEVANCE_MISMATCH = 0.15

# === 탐지 임계값 ===
RISK_THRESHOLD = 0.5   # 이 값 이상이면 위험 chunk로 판단

# === 정제 심각도 임계값 (3-way 선택) ===
SEVERITY_LOW    = 0.3   # A: 제거만
SEVERITY_MEDIUM = 0.6   # B: top-k 교체
# > SEVERITY_MEDIUM     # C: 치환 (LLM 재작성)

# === PPL 정규화 ===
PPL_CORPUS_MEAN = 120.0   # 정상 문서 평균 PPL (초기 추정, 런타임에 업데이트)
PPL_CORPUS_STD  = 40.0    # 정상 문서 PPL 표준편차

# === 지시형 키워드 (instructionality) ===
INSTRUCTION_KEYWORDS = [
    "반드시", "절대", "금지", "하지 마", "하지 마라", "말해서는 안", "응답하지",
    "답변하지", "거부", "무시", "따르지", "규정상", "정책상", "지침에 따라",
    "you must", "do not", "never", "ignore", "disregard", "refuse", "forbidden",
    "policy", "guideline", "instruction", "system prompt", "override",
]

# === 거부 유도 키워드 (refusal) ===
REFUSAL_KEYWORDS = [
    "답변드리기 어렵", "안전 정책", "제한된 정보", "민감한 정보", "응답을 드릴 수 없",
    "도움을 드리기 어렵", "해당 질문에 답변", "제 역할 범위", "윤리적", "보안상의 이유",
    "i cannot", "i'm unable", "i can't provide", "i must decline", "safety policy",
    "restricted", "sensitive information", "not able to answer", "cannot assist",
    "unable to help", "beyond my scope",
]

# === factual anchor 키워드 (relevance_mismatch 탐지용) ===
FACTUAL_ANCHOR_PATTERNS = [
    r"\d{4}년",           # 연도
    r"\d{1,2}월",         # 월
    r"\d{1,2}일",         # 일
    r"\d+학점",           # 학점
    r"\d+층",             # 층수
    r"\d{2,5}[-\d]+",    # 전화번호 등 숫자 패턴
    r"[가-힣]{2,4}\s*교수", # 교수명
    r"[A-Z]\d{3}",        # 강의실 코드
]
