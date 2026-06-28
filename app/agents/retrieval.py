from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.contracts import EvidenceItem, RiskDimension, SourceTier
from app.entity_resolution import EntityResolver
from app.policy import source_priority
from app.services.connectors import MultiSourceConnector

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        live_connector: MultiSourceConnector | None = None,
        enable_live_connectors: bool | None = None,
    ) -> None:
        self.entity_resolver = EntityResolver(similarity_threshold=0.75)
        if enable_live_connectors is None:
            enable_live_connectors = os.getenv("ENABLE_LIVE_CONNECTORS", "false").strip().lower() == "true"
        self.enable_live_connectors = enable_live_connectors
        self.live_connector = live_connector
        self.connectors: list[RetrievalConnector] = [
            SanctionsConnector(),
            RegulatoryConnector(),
            NewsConnector(),
            EsgConnector(),
        ]

    @staticmethod
    def _run_async(coro: Any):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - defensive propagation path
                error["value"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "value" in error:
            raise error["value"]
        return result.get("value", [])

    def _apply_entity_resolution(self, query_company: str, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        normalized: list[EvidenceItem] = []
        ambiguous_matches: list[str] = []
        for item in evidence:
            matched_name = str(item.metadata.get("matched_name") or item.metadata.get("company") or query_company)
            resolution = self.entity_resolver.resolve(query_name=query_company, matched_name=matched_name)
            if resolution.requires_review:
                ambiguous_matches.append(f"{item.source_name}:{matched_name}")
            normalized.append(
                item.model_copy(
                    update={
                        "entity_match_confidence": min(item.entity_match_confidence, resolution.similarity),
                        "metadata": {
                            **item.metadata,
                            "entity_resolution_query_name": resolution.query_name,
                            "entity_resolution_matched_name": resolution.matched_name,
                            "entity_resolution_similarity": f"{resolution.similarity:.4f}",
                            "entity_resolution_requires_review": resolution.requires_review,
                            "canonical_entity_id": resolution.canonical_entity_id,
                        },
                    }
                )
            )

        if ambiguous_matches:
            normalized.append(
                EvidenceItem(
                    evidence_id=f"{query_company}-entity-resolution-review",
                    dimension=RiskDimension.OPERATIONAL,
                    signal="entity_resolution_ambiguous",
                    value="requires_review",
                    source_name="system",
                    source_tier="secondary",
                    timestamp=datetime.now(timezone.utc),
                    entity_match_confidence=0.0,
                    source_confidence=0.4,
                    provenance_url="internal://entity-resolution/review",
                    metadata={"ambiguous_matches": ",".join(ambiguous_matches)},
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

    def _normalize(self, query_company: str, observations: list[RetrievalObservation]) -> list[EvidenceItem]:
        normalized: list[EvidenceItem] = []
        ambiguous_matches: list[str] = []
        for idx, item in enumerate(observations, start=1):
            matched_name = str(item.metadata.get("matched_name") or item.metadata.get("company") or query_company)
            resolution = self.entity_resolver.resolve(query_name=query_company, matched_name=matched_name)
            if resolution.requires_review:
                ambiguous_matches.append(f"{item.connector}:{matched_name}")
            normalized.append(
                EvidenceItem(
                    evidence_id=f"{query_company}-{item.connector}-{idx}",
                    dimension=item.dimension,
                    signal=item.signal,
                    value=item.value,
                    source_name=item.source_name,
                    source_tier=item.source_tier,
                    timestamp=item.timestamp,
                    entity_match_confidence=min(item.entity_match_confidence, resolution.similarity),
                    source_confidence=item.source_confidence,
                    provenance_url=item.provenance_url,
                    metadata={
                        **item.metadata,
                        "entity_resolution_query_name": resolution.query_name,
                        "entity_resolution_matched_name": resolution.matched_name,
                        "entity_resolution_similarity": f"{resolution.similarity:.4f}",
                        "entity_resolution_requires_review": resolution.requires_review,
                        "canonical_entity_id": resolution.canonical_entity_id,
                    },
                )
            )
        if ambiguous_matches:
            normalized.append(
                EvidenceItem(
                    evidence_id=f"{query_company}-entity-resolution-review",
                    dimension=RiskDimension.OPERATIONAL,
                    signal="entity_resolution_ambiguous",
                    value="requires_review",
                    source_name="system",
                    source_tier="secondary",
                    timestamp=datetime.now(timezone.utc),
                    entity_match_confidence=0.0,
                    source_confidence=0.4,
                    provenance_url="internal://entity-resolution/review",
                    metadata={"ambiguous_matches": ",".join(ambiguous_matches)},
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

    # Signals that indicate a connector found nothing for the entity
    # (not an error — the source responded but had no match)
    _NO_DATA_SIGNALS: frozenset[str] = frozenset({
        "not_sanctioned",       # OpenSanctions: no match
        "not_sec_registered",   # SECFinancials: not in registry
        "0",                    # SECConnector: zero filings
    })

    @staticmethod
    def _is_no_data_response(evidence: list[EvidenceItem]) -> bool:
        """
        Return True when all live evidence items are no-data signals
        (i.e. every connector responded but found nothing substantive).
        Excludes system-generated items (entity_resolution, retrieval_health).
        """
        real_items = [e for e in evidence if e.source_name.lower() != "system"]
        if not real_items:
            return False
        return all(e.value in RetrievalAgent._NO_DATA_SIGNALS for e in real_items)

    def retrieve(self, query_company: str, seeded_evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        if seeded_evidence:
            return seeded_evidence

        if self.enable_live_connectors and self.live_connector is not None:
            try:
                logger.info("retrieval_live_sources_start entity=%s", query_company)
                live_evidence = self._run_async(self.live_connector.fetch_all(query_company))
                if live_evidence:
                    resolved = self._apply_entity_resolution(query_company, live_evidence)
                    logger.info("retrieval_live_sources_done entity=%s evidence_count=%d", query_company, len(resolved))
                    # Detect when every connector returned a no-data response
                    if self._is_no_data_response(resolved):
                        logger.info("retrieval_entity_not_found entity=%s", query_company)
                        resolved.append(EvidenceItem(
                            evidence_id=f"{query_company}-entity-not-found",
                            dimension=RiskDimension.OPERATIONAL,
                            signal="entity_not_found",
                            value="no_data_across_all_sources",
                            source_name="system",
                            source_tier=SourceTier.SECONDARY,
                            timestamp=datetime.now(timezone.utc),
                            entity_match_confidence=1.0,
                            source_confidence=0.9,
                            provenance_url="internal://retrieval/not-found",
                            metadata={"query": query_company},
                        ))
                    return resolved
                logger.warning("retrieval_live_sources_empty entity=%s", query_company)
            except Exception as exc:
                logger.warning("retrieval_live_sources_failed entity=%s error=%s", query_company, exc)

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
