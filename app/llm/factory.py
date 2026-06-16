from __future__ import annotations

import os

from app.llm.base import LLMClient
from app.llm.stub import StubLLMClient
from app.llm.openai_client import OpenAILLMClient


def build_llm_client() -> LLMClient:
    if os.getenv("OPENAI_API_KEY", "").strip():
        return OpenAILLMClient(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    return StubLLMClient()
