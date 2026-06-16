from __future__ import annotations

from collections import defaultdict

from app.contracts import EvidenceItem, RiskDimension


class RetrievalAgent:
    """Real-time retrieval placeholder with normalized evidence output."""

    def retrieve(self, query_company: str, seeded_evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        if seeded_evidence:
            return seeded_evidence

        # Bootstrap fallback while connectors are being integrated.
        return [
            EvidenceItem(
                evidence_id=f"{query_company}-sanctions-official",
                dimension=RiskDimension.SANCTIONS,
                signal="sanctions_status",
                value="not_sanctioned",
                source_name="OFAC",
                source_tier="official",
                timestamp="2026-06-15T00:00:00Z",
                entity_match_confidence=0.92,
                source_confidence=0.95,
                provenance_url="https://ofac.treasury.gov/",
                metadata={"jurisdiction": "US"},
            ),
            EvidenceItem(
                evidence_id=f"{query_company}-news-sanctions-claim",
                dimension=RiskDimension.SANCTIONS,
                signal="sanctions_status",
                value="reported_sanctioned",
                source_name="Reuters",
                source_tier="tier1_news",
                timestamp="2026-06-14T00:00:00Z",
                entity_match_confidence=0.87,
                source_confidence=0.78,
                provenance_url="https://www.reuters.com/",
                metadata={"jurisdiction": "EU"},
            ),
        ]

    @staticmethod
    def detect_conflicts(evidence: list[EvidenceItem]) -> bool:
        grouped: dict[tuple[RiskDimension, str], set[str]] = defaultdict(set)
        for item in evidence:
            grouped[(item.dimension, item.signal)].add(item.value)
        return any(len(values) > 1 for values in grouped.values())
