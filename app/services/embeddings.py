"""
Real embedding pipeline for semantic search.
Supports OpenAI embeddings, OpenRouter embeddings, local models (sentence-transformers), and fallback strategies.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class EmbeddingModel(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
    
    @abstractmethod
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenRouterEmbeddings(EmbeddingModel):
    """
    Embeddings via OpenRouter — uses `openai/text-embedding-3-small` by default.
    OpenRouter exposes the same /embeddings endpoint as the OpenAI API, so we
    reuse the openai SDK with a custom base_url and the OPENROUTER_API_KEY.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        site_url: str = "https://github.com/ugivvimani/IRAPlatform",
        site_name: str = "IRA Platform",
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        self._headers = {"HTTP-Referer": site_url, "X-Title": site_name}
        self._client = None
        self._async_client = None
        if self._api_key:
            try:
                import openai
                self._client = openai.OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=self._api_key,
                    default_headers=self._headers,
                )
                self._async_client = openai.AsyncOpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=self._api_key,
                    default_headers=self._headers,
                )
            except ImportError:
                logger.warning("openai package not installed; OpenRouter embeddings unavailable")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._async_client:
            return self.embed_sync(texts)
        try:
            response = await self._async_client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.error("OpenRouter async embedding error: %s", exc)
            return [[0.0] * 1536 for _ in texts]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not self._client:
            logger.warning("OpenRouter client not initialised; returning zero vectors")
            return [[0.0] * 1536 for _ in texts]
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.error("OpenRouter embedding error: %s", exc)
            return [[0.0] * 1536 for _ in texts]


class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI Embeddings API (text-embedding-3-small or text-embedding-3-large)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.async_client = None
        
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed; falling back to local embeddings")
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.async_client:
            return self.embed_sync(texts)
        try:
            response = await self.async_client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.error("OpenAI embedding error: %s; falling back to zero vectors", exc)
            return [[0.0] * 1536 for _ in texts]
    
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            logger.warning("OpenAI client not initialized; returning zero vectors")
            return [[0.0] * 1536 for _ in texts]
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.error("OpenAI embedding error: %s; falling back to zero vectors", exc)
            return [[0.0] * 1536 for _ in texts]


class LocalEmbeddings(EmbeddingModel):
    """Local embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info("Loaded local embedding model: %s", model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed; embeddings will use fallback")
        except Exception as exc:
            logger.error("Failed to load embedding model %s: %s", model_name, exc)
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_sync, texts)
    
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not self.model:
            logger.warning("Model not loaded; returning zero vectors")
            return [[0.0] * 384 for _ in texts]
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as exc:
            logger.error("Local embedding error: %s", exc)
            return [[0.0] * 384 for _ in texts]


class HybridEmbeddings(EmbeddingModel):
    """Hybrid embedding: tries OpenRouter → OpenAI → local, in that order."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-small",
        local_model: str = "all-MiniLM-L6-v2",
    ):
        self.openrouter = OpenRouterEmbeddings()
        self.openai = OpenAIEmbeddings(api_key=openai_api_key, model=openai_model)
        self.local_model_name = local_model
        self.local: Optional[LocalEmbeddings] = None

    def _local(self) -> LocalEmbeddings:
        if self.local is None:
            self.local = LocalEmbeddings(model_name=self.local_model_name)
        return self.local
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.openrouter._async_client:
            try:
                return await self.openrouter.embed(texts)
            except Exception:
                pass
        if self.openai.async_client:
            try:
                return await self.openai.embed(texts)
            except Exception:
                pass
        return await self._local().embed(texts)
    
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        if self.openrouter._client:
            try:
                return self.openrouter.embed_sync(texts)
            except Exception:
                pass
        if self.openai.client:
            try:
                return self.openai.embed_sync(texts)
            except Exception:
                pass
        return self._local().embed_sync(texts)


class EmbeddingFactory:
    """Factory for creating embedding models based on configuration."""
    
    @staticmethod
    def create(embedding_type: str | None = None) -> EmbeddingModel:
        """
        Create embedding model based on EMBEDDING_TYPE env var or explicit argument.

        Priority when type is ``openrouter`` (default):
          1. OpenRouter (OPENROUTER_API_KEY) — 1536-dim real embeddings
          2. OpenAI (OPENAI_API_KEY) — 1536-dim real embeddings
          3. local sentence-transformers — 384-dim real embeddings (no key needed)

        Set EMBEDDING_TYPE=local to skip cloud providers entirely.
        """
        embedding_type = embedding_type or os.getenv("EMBEDDING_TYPE", "openrouter")
        openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if embedding_type == "openrouter":
            if openrouter_key:
                logger.info("embedding_provider=openrouter model=text-embedding-3-small")
                return OpenRouterEmbeddings()
            if openai_key:
                logger.info("embedding_provider=openai (openrouter key absent)")
                return OpenAIEmbeddings(model=openai_embedding_model)
            logger.warning("No cloud embedding key found; falling back to local sentence-transformers")
            return LocalEmbeddings(model_name=local_embedding_model)

        if embedding_type == "openai":
            if openai_key:
                return OpenAIEmbeddings(model=openai_embedding_model)
            if openrouter_key:
                logger.info("embedding_provider=openrouter (openai key absent, using openrouter)")
                return OpenRouterEmbeddings()
            logger.warning("No cloud embedding key found; falling back to local sentence-transformers")
            return LocalEmbeddings(model_name=local_embedding_model)

        if embedding_type == "local":
            return LocalEmbeddings(model_name=local_embedding_model)

        # hybrid: try in order
        return HybridEmbeddings(openai_model=openai_embedding_model, local_model=local_embedding_model)
