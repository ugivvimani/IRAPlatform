from __future__ import annotations

import logging
import os

from app.llm.base import LLMClient

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLMClient(LLMClient):
    """
    OpenAI-compatible client pointed at OpenRouter.

    OpenRouter accepts the same request shape as the OpenAI Chat Completions
    API, so we reuse the `openai` SDK and just override `base_url` + `api_key`.
    Model names follow the OpenRouter convention: ``provider/model-name``
    (e.g. ``openai/gpt-4o-mini``, ``anthropic/claude-3-haiku``).

    Swap to Azure OpenAI in future by changing LLM_PROVIDER to ``azure_openai``
    in the environment — no code changes needed.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        site_url: str = "https://github.com/ugivvimani/IRAPlatform",
        site_name: str = "IRA Platform",
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._site_url = site_url
        self._site_name = site_name

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("openai package not installed; run: pip install openai") from exc

        client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self._api_key,
            default_headers={
                "HTTP-Referer": self._site_url,
                "X-Title": self._site_name,
            },
        )

        logger.debug("openrouter_request model=%s max_tokens=%d", self.model, max_tokens)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug("openrouter_response chars=%d", len(content))
        return content
