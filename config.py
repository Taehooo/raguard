import os

# === Ollama 설정 ===
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = "bge-m3"      # ollama list에서 확인
OLLAMA_CHAT_MODEL  = "qwen3:4b"    # ollama list에서 확인

# === 모델 설정 (하위 호환) ===
RETRIEVER_MODEL = OLLAMA_EMBED_MODEL
GENERATOR_MODEL = OLLAMA_CHAT_MODEL

# === RAG 설정 ===
TOP_K = 5     # 최종 LLM에 전달할 chunk 수
TOP_N = 20    # 위험도 평가 전 후보 pool 크기

# === Risk Score 가중치 ===
WEIGHT_INSTRUCTIONALITY   = 0.35
WEIGHT_REFUSAL            = 0.30
WEIGHT_PPL_OUTLIER        = 0.20
WEIGHT_RELEVANCE_MISMATCH = 0.15

# === 탐지 임계값 ===
RISK_THRESHOLD = 0.5   # 이 값 이상이면 위험 chunk로 판단

# === 정제 심각도 임계값 (3-way 선택) ===
SEVERITY_LOW    = 0.3   # A: 제거만
SEVERITY_MEDIUM = 0.6   # B: top-k 교체
# > SEVERITY_MEDIUM     # C: 치환 (LLM 재작성)

# === PPL 정규화 (휴리스틱 기반, 신경망 모델 불필요) ===
PPL_CORPUS_MEAN = 0.0   # calibrate()에서 자동 설정
PPL_CORPUS_STD  = 1.0   # calibrate()에서 자동 설정

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
