from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


_LEGAL_SUFFIXES = (
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "ltd",
    "ltd.",
    "limited",
    "llc",
    "plc",
    "ag",
    "sa",
    "nv",
    "holdings",
    "group",
)


def _normalize_name(name: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", name.lower())
    tokens = [t for t in text.split() if t and t not in _LEGAL_SUFFIXES]
    return " ".join(tokens).strip()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


@dataclass(frozen=True, slots=True)
class EntityResolutionResult:
    query_name: str
    matched_name: str
    canonical_entity_id: str
    similarity: float
    requires_review: bool


class EntityResolver:
    def __init__(self, similarity_threshold: float = 0.75) -> None:
        self.similarity_threshold = similarity_threshold

    def resolve(self, query_name: str, matched_name: str | None = None) -> EntityResolutionResult:
        candidate = matched_name or query_name
        query_norm = _normalize_name(query_name)
        candidate_norm = _normalize_name(candidate)
        similarity = _similarity(query_norm, candidate_norm)
        canonical_entity_id = query_norm.replace(" ", "_") if query_norm else query_name.lower().replace(" ", "_")
        return EntityResolutionResult(
            query_name=query_name,
            matched_name=candidate,
            canonical_entity_id=canonical_entity_id,
            similarity=similarity,
            requires_review=similarity < self.similarity_threshold,
        )
