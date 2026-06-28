"""
Real embedding pipeline for semantic search.
Supports OpenAI embeddings, local models (sentence-transformers), and fallback strategies.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        raise NotImplementedError
    
    @abstractmethod
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding (for non-async contexts)."""
        raise NotImplementedError


class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI Embeddings API (text-embedding-3-small or text-embedding-3-large)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embeddings client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model name (text-embedding-3-small or text-embedding-3-large)
        """
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
        """Async embedding using OpenAI API."""
        if not self.async_client:
            logger.warning("OpenAI client not initialized; using sync embedding")
            return self.embed_sync(texts)
        
        try:
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}; falling back to zero vectors")
            return [[0.0] * 1536 for _ in texts]  # Fallback: zero vectors
    
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding using OpenAI API."""
        if not self.client:
            logger.warning("OpenAI client not initialized; returning zero vectors (local fallback disabled)")
            return [[0.0] * 1536 for _ in texts]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}; falling back to zero vectors")
            return [[0.0] * 1536 for _ in texts]


class LocalEmbeddings(EmbeddingModel):
    """Local embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding model.
        
        Args:
            model_name: HuggingFace model name (defaults to all-MiniLM-L6-v2 for speed/quality balance)
        """
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded local embedding model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed; embeddings will use fallback")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Async embedding using local model."""
        # For CPU-bound operations, run in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_sync, texts)
    
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding using local model."""
        if not self.model:
            logger.warning("Model not loaded; returning zero vectors")
            # Return zero vectors of reasonable size (384 for all-MiniLM-L6-v2)
            return [[0.0] * 384 for _ in texts]
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            return [[0.0] * 384 for _ in texts]


class HybridEmbeddings(EmbeddingModel):
    """Hybrid embedding: tries OpenAI first, falls back to local."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-small",
        local_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize hybrid embedding strategy."""
        self.openai = OpenAIEmbeddings(api_key=openai_api_key, model=openai_model)
        self.local_model_name = local_model
        self.local: Optional[LocalEmbeddings] = None
        # Flag to prefer one or the other
        self.prefer_openai = bool(openai_api_key or os.getenv("OPENAI_API_KEY"))

    def _local_embeddings(self) -> LocalEmbeddings:
        if self.local is None:
            self.local = LocalEmbeddings(model_name=self.local_model_name)
        return self.local
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Try OpenAI first, fall back to local."""
        if self.prefer_openai and self.openai.async_client:
            try:
                return await self.openai.embed(texts)
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}; falling back to local")
        
        return await self._local_embeddings().embed(texts)
    
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Try OpenAI first, fall back to local (synchronous)."""
        if self.prefer_openai and self.openai.client:
            try:
                return self.openai.embed_sync(texts)
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}; falling back to local")
        
        return self._local_embeddings().embed_sync(texts)


class EmbeddingFactory:
    """Factory for creating embedding models based on configuration."""
    
    @staticmethod
    def create(embedding_type: str = "openai") -> EmbeddingModel:
        """
        Create embedding model based on type.
        
        Args:
            embedding_type: "openai", "local", or "hybrid" (default: openai)
        
        Returns:
            Configured embedding model
        """
        embedding_type = embedding_type or os.getenv("EMBEDDING_TYPE", "openai")
        openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        if embedding_type == "openai":
            return OpenAIEmbeddings(model=openai_embedding_model)
        elif embedding_type == "local":
            return LocalEmbeddings(model_name=local_embedding_model)
        else:  # hybrid (default)
            return HybridEmbeddings(openai_model=openai_embedding_model, local_model=local_embedding_model)
