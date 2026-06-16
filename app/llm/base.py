from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError
