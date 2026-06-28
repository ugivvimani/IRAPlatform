from __future__ import annotations

import logging
import os
from typing import Any

from app.vector_store.base import VectorDocument, VectorStoreRepository

logger = logging.getLogger(__name__)


class PineconeVectorStore(VectorStoreRepository):
    """
    Pinecone vector store adapter.
    Uses the Pinecone SDK when PINECONE_API_KEY is set; falls back to
    an in-memory store for local dev without credentials.

    Accepts an optional ``embedding_fn`` callable so the caller (main.py)
    can inject the real OpenAI embedding model.  When not provided it falls
    back to the deterministic MD5 hash (safe for unit tests / offline dev
    but NOT suitable for meaningful semantic search).
    """

    def __init__(self, embedding_fn=None) -> None:
        self._api_key = os.getenv("PINECONE_API_KEY", "").strip()
        self._index_name = os.getenv("PINECONE_INDEX", "ira-platform-memory")
        self._namespace_prefix = os.getenv("PINECONE_NAMESPACE", "default")
        self._index: Any = None
        self._memory: dict[str, list[VectorDocument]] = {}
        # Caller-injected embedding function; signature: (text: str) -> list[float]
        self._embedding_fn = embedding_fn

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
                {
                    "id": doc.doc_id,
                    "values": self._embed(doc.text),
                    # Store text in metadata so it can be retrieved on query
                    "metadata": {**doc.metadata, "_text": doc.text},
                }
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
                    # Prefer explicit _text field; fall back to legacy summary key
                    text=match.get("metadata", {}).get("_text")
                        or match.get("metadata", {}).get("summary", ""),
                    metadata={k: v for k, v in match.get("metadata", {}).items() if k != "_text"},
                )
                for match in result.get("matches", [])
            ]
        except Exception:
            return self._mem_query(namespace, text, top_k, metadata_filter)

    def _embed(self, text: str) -> list[float]:
        """Return real embeddings when a fn is injected; MD5 hash as offline fallback."""
        if self._embedding_fn is not None:
            try:
                # embedding_fn signature: (texts: list[str]) -> list[list[float]]
                # We pass a single-element list and take the first result.
                result = self._embedding_fn([text])
                if result and isinstance(result[0], list):
                    return result[0]
            except Exception as exc:
                logger.warning("embedding_fn failed, using hash fallback: %s", exc)
        # Deterministic fallback — only for tests / offline dev
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
