from __future__ import annotations

import os

from app.llm.base import LLMClient


class OpenAILLMClient(LLMClient):
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._api_key = os.getenv("OPENAI_API_KEY", "")

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        try:
            import openai  # lazy import — only required when actually used
        except ImportError as exc:
            raise RuntimeError("openai package not installed; add it to requirements.txt") from exc

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
