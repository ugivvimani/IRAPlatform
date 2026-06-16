from __future__ import annotations

from app.vector_store.base import VectorStoreRepository
from app.vector_store.pinecone_store import PineconeVectorStore


def build_vector_store() -> VectorStoreRepository:
    return PineconeVectorStore()
