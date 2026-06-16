from __future__ import annotations

import os
from typing import Any

from app.vector_store.base import VectorDocument, VectorStoreRepository


class PineconeVectorStore(VectorStoreRepository):
    """
    Pinecone vector store adapter.
    Uses the Pinecone SDK when PINECONE_API_KEY is set; falls back to
    an in-memory store for local dev without credentials.
    """

    def __init__(self) -> None:
        self._api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self._index_name = os.getenv("PINECONE_INDEX", "ira-platform-memory")
        self._namespace_prefix = os.getenv("PINECONE_NAMESPACE", "default")
        self._index: Any = None
        self._memory: dict[str, list[VectorDocument]] = {}

        if self._api_key:
            self._index = self._connect()

    def _connect(self) -> Any:
        try:
            from pinecone import Pinecone  # type: ignore[import]
            return Pinecone(api_key=self._api_key).Index(self._index_name)
        except Exception:
            return None

    def _ns(self, namespace: str) -> str:
        return f"{self._namespace_prefix}/{namespace}"

    def upsert(self, namespace: str, docs: list[VectorDocument]) -> None:
        if self._index is not None:
            self._sdk_upsert(namespace, docs)
        else:
            self._mem_upsert(namespace, docs)

    def query(
        self,
        namespace: str,
        text: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorDocument]:
        if self._index is not None:
            return self._sdk_query(namespace, text, top_k, metadata_filter)
        return self._mem_query(namespace, text, top_k, metadata_filter)

    # ── SDK paths ──────────────────────────────────────────────────────────────

    def _sdk_upsert(self, namespace: str, docs: list[VectorDocument]) -> None:
        try:
            vectors = [
                {"id": doc.doc_id, "values": self._embed(doc.text), "metadata": doc.metadata}
                for doc in docs
            ]
            self._index.upsert(vectors=vectors, namespace=self._ns(namespace))
        except Exception:
            self._mem_upsert(namespace, docs)

    def _sdk_query(
        self,
        namespace: str,
        text: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[VectorDocument]:
        try:
            result = self._index.query(
                vector=self._embed(text),
                top_k=top_k,
                namespace=self._ns(namespace),
                filter=metadata_filter,
                include_metadata=True,
            )
            return [
                VectorDocument(
                    doc_id=match["id"],
                    text=match.get("metadata", {}).get("summary", ""),
                    metadata=match.get("metadata", {}),
                )
                for match in result.get("matches", [])
            ]
        except Exception:
            return self._mem_query(namespace, text, top_k, metadata_filter)

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Deterministic placeholder embedding — replace with real encoder in production."""
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(((h >> i) & 0xFF) / 255.0) for i in range(1536)]

    # ── In-memory fallback ─────────────────────────────────────────────────────

    def _mem_upsert(self, namespace: str, docs: list[VectorDocument]) -> None:
        existing = {d.doc_id: d for d in self._memory.get(namespace, [])}
        for doc in docs:
            existing[doc.doc_id] = doc
        self._memory[namespace] = list(existing.values())

    def _mem_query(
        self,
        namespace: str,
        text: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[VectorDocument]:
        candidates = self._memory.get(namespace, [])
        if metadata_filter:
            candidates = [d for d in candidates if all(d.metadata.get(k) == v for k, v in metadata_filter.items())]
        ranked = sorted(
            candidates,
            key=lambda d: sum(1 for t in text.lower().split() if t in d.text.lower()),
            reverse=True,
        )
        return ranked[:top_k]
