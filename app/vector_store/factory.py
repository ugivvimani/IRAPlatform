from __future__ import annotations

from typing import Callable

from app.vector_store.base import VectorStoreRepository
from app.vector_store.pinecone_store import PineconeVectorStore


def build_vector_store(embedding_fn: Callable[[str], list[float]] | None = None) -> VectorStoreRepository:
    return PineconeVectorStore(embedding_fn=embedding_fn)
