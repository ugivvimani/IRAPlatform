from __future__ import annotations

import logging
import os

from app.llm.base import LLMClient

logger = logging.getLogger(__name__)


class AzureOpenAILLMClient(LLMClient):
    """
    Client for Azure OpenAI Service (Chat Completions).

    Required environment variables:
        AZURE_OPENAI_ENDPOINT   — e.g. https://<resource>.openai.azure.com
        AZURE_OPENAI_API_KEY    — Azure API key (or use managed identity)
        AZURE_OPENAI_DEPLOYMENT — deployment / model name (e.g. gpt-4o-mini)
        AZURE_OPENAI_API_VERSION — e.g. 2024-02-01

    To switch from OpenRouter to Azure OpenAI set LLM_PROVIDER=azure_openai
    in the environment and fill in the four variables above. No code changes needed.
    """

    def __init__(
        self,
        deployment: str | None = None,
        api_version: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self._deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self._api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("openai package not installed; run: pip install openai") from exc

        client = openai.AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )

        logger.debug("azure_openai_request deployment=%s max_tokens=%d", self._deployment, max_tokens)
        response = client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug("azure_openai_response chars=%d", len(content))
        return content
