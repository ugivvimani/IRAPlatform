from __future__ import annotations

import logging
import os

from app.llm.base import LLMClient

logger = logging.getLogger(__name__)

# Supported providers and the env var that must be non-empty to use them.
_PROVIDER_KEYS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
}


def build_llm_client() -> LLMClient:
    """
    Build the LLM client based on the ``LLM_PROVIDER`` environment variable.

    Supported values:
        openrouter   — OpenRouter.ai  (default)
        openai       — OpenAI direct
        azure_openai — Azure OpenAI Service  (future swap target)
        stub         — deterministic stub for tests / offline dev

    Falls back to ``stub`` when the required API key for the chosen provider
    is missing so the app still starts without credentials.
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

    if provider == "stub":
        logger.info("llm_provider=stub (forced)")
        from app.llm.stub import StubLLMClient
        return StubLLMClient()

    required_key = _PROVIDER_KEYS.get(provider)
    if required_key and not os.getenv(required_key, "").strip():
        logger.warning(
            "llm_provider=%s but %s is not set — falling back to stub",
            provider,
            required_key,
        )
        from app.llm.stub import StubLLMClient
        return StubLLMClient()

    if provider == "openrouter":
        from app.llm.openrouter_client import OpenRouterLLMClient
        model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
        logger.info("llm_provider=openrouter model=%s", model)
        return OpenRouterLLMClient(model=model)

    if provider == "openai":
        from app.llm.openai_client import OpenAILLMClient
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        logger.info("llm_provider=openai model=%s", model)
        return OpenAILLMClient(model=model)

    if provider == "azure_openai":
        from app.llm.azure_openai_client import AzureOpenAILLMClient
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        logger.info("llm_provider=azure_openai deployment=%s", deployment)
        return AzureOpenAILLMClient()

    logger.warning("Unknown LLM_PROVIDER=%s — falling back to stub", provider)
    from app.llm.stub import StubLLMClient
    return StubLLMClient()
