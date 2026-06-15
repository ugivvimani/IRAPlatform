from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.vector_store.base import VectorDocument, VectorStoreRepository


class PineconeVectorStore(VectorStoreRepository):
    """Production adapter contract. Replace placeholder logic with Pinecone SDK calls."""

    def __init__(self) -> None:
        self._storage: dict[str, list[VectorDocument]] = defaultdict(list)

    def upsert(self, namespace: str, docs: list[VectorDocument]) -> None:
        existing = {doc.doc_id: doc for doc in self._storage[namespace]}
        for doc in docs:
            existing[doc.doc_id] = doc
        self._storage[namespace] = list(existing.values())

    def query(self, namespace: str, text: str, top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[VectorDocument]:
        candidates = self._storage.get(namespace, [])
        filtered: list[VectorDocument] = []
        for item in candidates:
            if metadata_filter and not all(item.metadata.get(k) == v for k, v in metadata_filter.items()):
                continue
            filtered.append(item)
        ranked = sorted(
            filtered,
            key=lambda d: sum(1 for token in text.lower().split() if token in d.text.lower()),
            reverse=True,
        )
        return ranked[:top_k]
