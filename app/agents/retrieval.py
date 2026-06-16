from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from app.contracts import EvidenceItem, RiskDimension
from app.policy import source_priority


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    connector: str
    signal: str
    value: str
    source_name: str
    source_tier: str
    provenance_url: str
    dimension: RiskDimension
    timestamp: datetime
    entity_match_confidence: float
    source_confidence: float
    metadata: dict[str, str]


class RetrievalConnector:
    connector_name: str

    def fetch(self, query_company: str) -> list[RetrievalObservation]:
        raise NotImplementedError


class SanctionsConnector(RetrievalConnector):
    connector_name = "sanctions_ofac"

    def fetch(self, query_company: str) -> list[RetrievalObservation]:
        return [
            RetrievalObservation(
                connector=self.connector_name,
                signal="sanctions_status",
                value="not_sanctioned",
                source_name="OFAC",
                source_tier="official",
                provenance_url="https://ofac.treasury.gov/",
                dimension=RiskDimension.SANCTIONS,
                timestamp=datetime.now(timezone.utc),
                entity_match_confidence=0.93,
                source_confidence=0.95,
                metadata={"company": query_company, "jurisdiction": "US"},
            )
        ]


class RegulatoryConnector(RetrievalConnector):
    connector_name = "regulatory_filings"

    def fetch(self, query_company: str) -> list[RetrievalObservation]:
        return [
            RetrievalObservation(
                connector=self.connector_name,
                signal="regulatory_enforcement",
                value="no_recent_enforcement",
                source_name="SEC",
                source_tier="regulator",
                provenance_url="https://www.sec.gov/",
                dimension=RiskDimension.REGULATORY,
                timestamp=datetime.now(timezone.utc),
                entity_match_confidence=0.9,
                source_confidence=0.88,
                metadata={"company": query_company, "window_days": "365"},
            )
        ]


class NewsConnector(RetrievalConnector):
    connector_name = "news_feed"

    def fetch(self, query_company: str) -> list[RetrievalObservation]:
        return [
            RetrievalObservation(
                connector=self.connector_name,
                signal="sanctions_status",
                value="reported_sanctioned",
                source_name="Reuters",
                source_tier="tier1_news",
                provenance_url="https://www.reuters.com/",
                dimension=RiskDimension.SANCTIONS,
                timestamp=datetime.now(timezone.utc),
                entity_match_confidence=0.86,
                source_confidence=0.77,
                metadata={"company": query_company, "jurisdiction": "EU"},
            ),
            RetrievalObservation(
                connector=self.connector_name,
                signal="reputation_signal",
                value="negative_press",
                source_name="Reuters",
                source_tier="tier1_news",
                provenance_url="https://www.reuters.com/",
                dimension=RiskDimension.REPUTATIONAL,
                timestamp=datetime.now(timezone.utc),
                entity_match_confidence=0.86,
                source_confidence=0.74,
                metadata={"company": query_company},
            ),
        ]


class EsgConnector(RetrievalConnector):
    connector_name = "esg_incident_feed"

    def fetch(self, query_company: str) -> list[RetrievalObservation]:
        return [
            RetrievalObservation(
                connector=self.connector_name,
                signal="esg_incident",
                value="none_material",
                source_name="ESGDB",
                source_tier="secondary",
                provenance_url="https://example.org/esgdb",
                dimension=RiskDimension.ESG,
                timestamp=datetime.now(timezone.utc),
                entity_match_confidence=0.82,
                source_confidence=0.7,
                metadata={"company": query_company},
            )
        ]


class RetrievalAgent:
    """Retrieves and normalizes multi-source observations with fallback behavior."""

    def __init__(self) -> None:
        self.connectors: list[RetrievalConnector] = [
            SanctionsConnector(),
            RegulatoryConnector(),
            NewsConnector(),
            EsgConnector(),
        ]

    def _normalize(self, query_company: str, observations: list[RetrievalObservation]) -> list[EvidenceItem]:
        normalized: list[EvidenceItem] = []
        for idx, item in enumerate(observations, start=1):
            normalized.append(
                EvidenceItem(
                    evidence_id=f"{query_company}-{item.connector}-{idx}",
                    dimension=item.dimension,
                    signal=item.signal,
                    value=item.value,
                    source_name=item.source_name,
                    source_tier=item.source_tier,
                    timestamp=item.timestamp,
                    entity_match_confidence=item.entity_match_confidence,
                    source_confidence=item.source_confidence,
                    provenance_url=item.provenance_url,
                    metadata=item.metadata,
                )
            )
        return sorted(
            normalized,
            key=lambda e: (
                source_priority(e.source_tier),
                e.timestamp,
                e.entity_match_confidence,
            ),
            reverse=True,
        )

    def retrieve(self, query_company: str, seeded_evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        if seeded_evidence:
            return seeded_evidence

        observations: list[RetrievalObservation] = []
        failed_connectors: list[str] = []
        for connector in self.connectors:
            try:
                observations.extend(connector.fetch(query_company))
            except Exception:
                failed_connectors.append(connector.connector_name)

        evidence = self._normalize(query_company, observations)
        if not evidence:
            # Conservative fallback to avoid false confidence on total retrieval failure.
            return [
                EvidenceItem(
                    evidence_id=f"{query_company}-retrieval-fallback",
                    dimension=RiskDimension.OPERATIONAL,
                    signal="retrieval_health",
                    value="insufficient_live_data",
                    source_name="system",
                    source_tier="secondary",
                    timestamp=datetime.now(timezone.utc),
                    entity_match_confidence=1.0,
                    source_confidence=0.2,
                    provenance_url="internal://retrieval/fallback",
                    metadata={"failed_connectors": ",".join(failed_connectors)},
                )
            ]
        return evidence

    @staticmethod
    def detect_conflicts(evidence: list[EvidenceItem]) -> bool:
        grouped: dict[tuple[RiskDimension, str], set[str]] = defaultdict(set)
        for item in evidence:
            grouped[(item.dimension, item.signal)].add(item.value)
        return any(len(values) > 1 for values in grouped.values())
