from __future__ import annotations

from datetime import datetime, timezone

from app.contracts import CalibrationRecord, EvidenceItem, MemoryFact, UserQuery
from app.vector_store.base import VectorDocument, VectorStoreRepository


class MemoryManagerAgent:
    def __init__(self, vector_store: VectorStoreRepository) -> None:
        self.vector_store = vector_store
        self.working_memory: dict[str, object] = {}

    def initialize_working_memory(self, query: UserQuery, evidence: list[EvidenceItem]) -> None:
        self.working_memory = {
            "query": query.model_dump(),
            "evidence_count": len(evidence),
            "initialized_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_historical_context(self, entity_id: str, top_k: int = 5) -> list[MemoryFact]:
        docs = self.vector_store.query(
            namespace="historical_facts",
            text=entity_id,
            top_k=top_k,
            metadata_filter={"entity_id": entity_id},
        )
        facts: list[MemoryFact] = []
        for doc in docs:
            facts.append(
                MemoryFact(
                    fact_id=doc.doc_id,
                    entity_id=entity_id,
                    summary=doc.text,
                    dimension=doc.metadata.get("dimension", "reputational"),
                    severity=float(doc.metadata.get("severity", 0.5)),
                    source_reference=doc.metadata.get("source_reference", "memory"),
                    timestamp=datetime.fromisoformat(doc.metadata.get("timestamp", "2026-01-01T00:00:00+00:00")),
                    metadata=doc.metadata,
                )
            )
        return facts

    def persist_facts(self, facts: list[MemoryFact]) -> None:
        docs = [
            VectorDocument(
                doc_id=fact.fact_id,
                text=fact.summary,
                metadata={
                    "entity_id": fact.entity_id,
                    "dimension": fact.dimension.value,
                    "severity": fact.severity,
                    "source_reference": fact.source_reference,
                    "timestamp": fact.timestamp.isoformat(),
                },
            )
            for fact in facts
        ]
        self.vector_store.upsert(namespace="historical_facts", docs=docs)

    def persist_calibration(self, record: CalibrationRecord) -> None:
        calibration_doc_id = f"{record.entity_id}:{record.source_name}:{record.signal_type}"
        existing = self.vector_store.query(
            namespace="calibration",
            text=calibration_doc_id,
            top_k=1,
            metadata_filter={"entity_id": record.entity_id, "source_name": record.source_name, "signal_type": record.signal_type},
        )
        if existing:
            previous = existing[0].metadata
            record.true_positive += int(previous.get("true_positive", 0))
            record.false_positive += int(previous.get("false_positive", 0))
            record.true_negative += int(previous.get("true_negative", 0))
            record.false_negative += int(previous.get("false_negative", 0))
            record.effective_sample_size += float(previous.get("effective_sample_size", 0.0))

        total_outcomes = record.true_positive + record.true_negative + record.false_positive + record.false_negative
        record.total_outcomes = total_outcomes
        if total_outcomes > 0 and record.effective_sample_size > 0:
            normalized_weight = record.effective_sample_size / total_outcomes
            weighted_success = (record.true_positive + record.true_negative) * normalized_weight
            weighted_failures = (record.false_positive + record.false_negative) * normalized_weight
            record.reliability_score = (1.0 + weighted_success) / (2.0 + weighted_success + weighted_failures)
            record.uncertainty_score = 1.0 / (1.0 + weighted_success + weighted_failures)
        else:
            record.reliability_score = 0.5
            record.uncertainty_score = 1.0

        self.vector_store.upsert(
            namespace="calibration",
            docs=[
                VectorDocument(
                    doc_id=calibration_doc_id,
                    text=f"{record.source_name} {record.signal_type} reliability",
                    metadata=record.model_dump(),
                )
            ],
        )

    def load_source_reliability(self, entity_id: str) -> dict[str, float]:
        docs = self.vector_store.query(
            namespace="calibration",
            text=entity_id,
            top_k=50,
            metadata_filter={"entity_id": entity_id},
        )
        by_source: dict[str, float] = {}
        prior_strength = 3.0
        for doc in docs:
            source_name = str(doc.metadata.get("source_name", "")).strip()
            if not source_name:
                continue
            reliability = float(doc.metadata.get("reliability_score", 0.5))
            sample_size = float(doc.metadata.get("effective_sample_size", 0.0))
            by_source[source_name] = (
                (reliability * sample_size + 0.5 * prior_strength) / (sample_size + prior_strength)
            )
        return by_source
