from __future__ import annotations

import os

from app.vector_store.base import VectorStoreRepository
from app.vector_store.chroma_store import ChromaVectorStore
from app.vector_store.pinecone_store import PineconeVectorStore


def build_vector_store() -> VectorStoreRepository:
    backend = os.getenv("VECTOR_BACKEND", "chroma").strip().lower()
    if backend == "pinecone":
        return PineconeVectorStore()
    return ChromaVectorStore()
