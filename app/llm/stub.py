from __future__ import annotations

from app.llm.base import LLMClient


class StubLLMClient(LLMClient):
    """Deterministic stub used for local dev and tests without API credentials."""

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        return f"[stub] reasoning for: {user_prompt[:80]}"
