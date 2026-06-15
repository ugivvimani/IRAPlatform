from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VectorDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStoreRepository(ABC):
    @abstractmethod
    def upsert(self, namespace: str, docs: list[VectorDocument]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, namespace: str, text: str, top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[VectorDocument]:
        raise NotImplementedError
