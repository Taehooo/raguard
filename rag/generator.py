"""
OpenAI API 기반 generator
모델: gpt-4o-mini (기본값, config에서 변경 가능)
"""

from __future__ import annotations

from openai import OpenAI

from config import OPENAI_API_KEY, GENERATOR_MODEL


SYSTEM_PROMPT = """당신은 대학교 학사 정보 안내 도우미입니다.
주어진 참고 문서를 바탕으로 학생의 질문에 정확하고 친절하게 답변하세요.
참고 문서에 없는 내용은 '해당 정보가 문서에 포함되어 있지 않습니다'라고 명확히 답하세요.
참고 문서 이외의 개인적인 추측이나 외부 정보를 포함하지 마세요."""


def _build_context_block(chunks: list[dict]) -> str:
    """검색된 chunk 목록을 컨텍스트 블록으로 변환"""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[문서 {i}] {chunk['content']}")
    return "\n\n".join(parts)


class Generator:
    def __init__(
        self,
        model: str = GENERATOR_MODEL,
        api_key: str = OPENAI_API_KEY,
    ):
        if not api_key:
            raise ValueError(
                "OpenAI API 키가 설정되지 않았습니다. "
                "환경 변수 OPENAI_API_KEY를 설정하거나 config.py를 수정하세요."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        query: str,
        contexts: list[dict],
        temperature: float = 0.0,
    ) -> str:
        """
        RAG 프롬프트로 LLM 응답 생성

        Parameters
        ----------
        query    : 사용자 질문
        contexts : retriever가 반환한 chunk 리스트
        temperature : 생성 온도 (0.0 = 결정론적)

        Returns
        -------
        생성된 응답 문자열
        """
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
