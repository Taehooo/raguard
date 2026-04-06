"""
Ollama 기반 generator (OpenAI 호환 API 사용)
모델: qwen3:4b
"""

from __future__ import annotations

from openai import OpenAI

from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


SYSTEM_PROMPT = """당신은 대학교 학사 정보 안내 도우미입니다.
주어진 참고 문서를 바탕으로 학생의 질문에 정확하고 친절하게 답변하세요.
참고 문서에 없는 내용은 '해당 정보가 문서에 포함되어 있지 않습니다'라고 명확히 답하세요.
참고 문서 이외의 개인적인 추측이나 외부 정보를 포함하지 마세요."""


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[문서 {i}] {chunk['content']}")
    return "\n\n".join(parts)


class Generator:
    def __init__(
        self,
        model: str = OLLAMA_CHAT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",   # Ollama는 API 키 불필요, 임의값 입력
        )
        self.model = model

    def generate(
        self,
        query: str,
        contexts: list[dict],
        temperature: float = 0.0,
    ) -> str:
        context_block = _build_context_block(contexts)

        user_message = (
            f"다음 참고 문서를 바탕으로 질문에 답하세요.\n\n"
            f"=== 참고 문서 ===\n{context_block}\n\n"
            f"=== 질문 ===\n{query}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
            max_tokens=512,
        )

        return response.choices[0].message.content.strip()
