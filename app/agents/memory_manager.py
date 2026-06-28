from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.contracts import CalibrationRecord, EvidenceItem, MemoryFact, UserQuery
from app.vector_store.base import VectorDocument, VectorStoreRepository

if TYPE_CHECKING:
    from app.contracts import ConflictResolutionResult

logger = logging.getLogger(__name__)

# ── Global source reliability priors ─────────────────────────────────────────
# Used as the starting point when no entity-specific calibration exists.
# Values are derived from published accuracy benchmarks and domain knowledge:
#   OpenSanctions – maintained by professional data team, high precision
#   SEC EDGAR – official filings, near-perfect reliability for public companies
#   NewsAPI – aggregator; quality varies; conservative prior
#   Sustainalytics – commercial ESG provider; moderate accuracy
_GLOBAL_SOURCE_PRIORS: dict[str, float] = {
    "opensanctions":         0.88,
    "sec_edgar_filings":     0.82,
    "sec_edgar_financial":   0.82,
    "newsapi":               0.65,
    "sustainalytics_esg":    0.72,
    "system":                0.90,  # internal system signals
}
_PRIOR_STRENGTH = 3.0  # equivalent sample size used in Bayesian blending


class MemoryManagerAgent:
    def __init__(self, vector_store: VectorStoreRepository, llm_client=None) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.working_memory: dict[str, object] = {}

    def initialize_working_memory(self, query: UserQuery, evidence: list[EvidenceItem]) -> None:
        self.working_memory = {
            "query": query.model_dump(),
            "evidence_count": len(evidence),
            "initialized_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_historical_context(self, entity_id: str, top_k: int = 5, query: str | None = None) -> list[MemoryFact]:
        docs = self.vector_store.query(
            namespace="historical_facts",
            text=query or entity_id,
            top_k=top_k * 2,  # fetch more candidates so LLM filter has room to discard
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

        # LLM relevance filter: discard facts that are not relevant to the current query.
        # Graceful fallback: if no LLM or LLM fails, return all facts (safe default).
        if self.llm_client and facts and query:
            facts = self._llm_filter_facts(facts, entity_id=entity_id, query=query, max_keep=top_k)

        return facts[:top_k]

    def _llm_filter_facts(
        self,
        facts: list[MemoryFact],
        entity_id: str,
        query: str,
        max_keep: int,
    ) -> list[MemoryFact]:
        """
        Ask the LLM to score each fact for relevance to the current assessment query.
        Facts scoring below 0.4 are discarded to prevent semantic drift.
        Returns at most `max_keep` facts ordered by relevance score.
        """
        try:
            fact_lines = "\n".join(
                f"{i}. [{f.dimension}] {f.summary}" for i, f in enumerate(facts, 1)
            )
            user_prompt = (
                f"Entity: {entity_id}\n"
                f"Current assessment query: {query}\n\n"
                f"Historical facts from memory:\n{fact_lines}\n\n"
                "For each numbered fact, output ONE line in this exact format:\n"
                "<number>: <score>\n"
                "where <score> is a float 0.0-1.0 indicating how relevant the fact is "
                "to the current query (1.0 = highly relevant, 0.0 = irrelevant). "
                "Output ONLY the numbered scores, nothing else."
            )
            response = self.llm_client.complete(
                system_prompt=(
                    "You are a relevance filter for a compliance risk assessment system. "
                    "Score historical memory facts for relevance to the current query. "
                    "Be strict: only facts directly relevant to the assessment should score above 0.5."
                ),
                user_prompt=user_prompt,
                max_tokens=200,
            )
            # Parse "1: 0.85\n2: 0.20\n..." format
            scored: list[tuple[float, MemoryFact]] = []
            for line in response.strip().splitlines():
                parts = line.strip().split(":")
                if len(parts) == 2:
                    idx_str, score_str = parts
                    try:
                        idx = int(idx_str.strip()) - 1
                        score = float(score_str.strip())
                        if 0 <= idx < len(facts) and score >= 0.4:
                            scored.append((score, facts[idx]))
                    except ValueError:
                        continue
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                return [f for _, f in scored[:max_keep]]
            # If parsing failed, return originals
            return facts[:max_keep]
        except Exception as exc:
            logger.warning("llm_relevance_filter_failed error=%s (returning all facts)", exc)
            return facts[:max_keep]

    def persist_facts(self, facts: list[MemoryFact]) -> None:
        docs = [
            VectorDocument(
                doc_id=fact.fact_id,
                text=self._summarize_content(fact.summary, context=fact.entity_id),
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

    def _summarize_content(self, text: str, context: str = "") -> str:
        """
        If `text` is rich prose (> 120 chars), ask the LLM to condense it into
        1-2 sentences (~80 words max) for cost-efficient vector storage.
        Falls back to the raw text when the LLM is unavailable or the text is
        already short enough.
        """
        if not text or len(text) <= 120 or not self.llm_client:
            return text
        try:
            response = self.llm_client.complete(
                system_prompt=(
                    "You are a compliance data summarizer. "
                    "Summarize the following evidence snippet in 1-2 concise sentences "
                    "(max 80 words). Preserve key facts: entity name, risk signal, "
                    "severity indicators, and source. Output ONLY the summary text."
                ),
                user_prompt=f"Entity context: {context}\n\nEvidence: {text}",
                max_tokens=100,
            )
            return response.strip() or text
        except Exception as exc:
            logger.warning("evidence_summarize_failed error=%s (using raw text)", exc)
            return text

    def persist_assessment_narrative(
        self,
        entity_id: str,
        risk_rating: str,
        confidence: str,
        summary: str,
        requires_manual_review: bool,
    ) -> None:
        """
        Store the LLM-generated assessment summary in the ``assessment_narratives``
        namespace so future assessments can retrieve semantically similar past cases.
        """
        doc_id = f"{entity_id}:narrative:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        narrative_text = (
            f"Assessment of {entity_id}: {summary} "
            f"[Rating: {risk_rating}, Confidence: {confidence}"
            f"{', requires manual review' if requires_manual_review else ''}]"
        )
        self.vector_store.upsert(
            namespace="assessment_narratives",
            docs=[
                VectorDocument(
                    doc_id=doc_id,
                    text=narrative_text,
                    metadata={
                        "entity_id": entity_id,
                        "risk_rating": risk_rating,
                        "confidence": confidence,
                        "requires_manual_review": requires_manual_review,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )

    def seed_cold_start(self, entity_id: str, source_names: list[str]) -> None:
        """
        For a brand-new entity with no calibration history, seed the vector store
        with global prior documents so that `load_source_reliability()` can blend
        informed priors from the very first assessment.

        Documents are only written if no calibration record already exists for
        this entity + source combination (idempotent / warm-up only).
        """
        for source_name in source_names:
            prior = _GLOBAL_SOURCE_PRIORS.get(source_name.lower(), 0.60)
            doc_id = f"{entity_id}:{source_name}:cold_start_prior"
            existing = self.vector_store.query(
                namespace="calibration",
                text=doc_id,
                top_k=1,
                metadata_filter={"entity_id": entity_id, "source_name": source_name},
            )
            if not existing:
                self.vector_store.upsert(
                    namespace="calibration",
                    docs=[
                        VectorDocument(
                            doc_id=doc_id,
                            text=f"{source_name} cold start prior",
                            metadata={
                                "entity_id":             entity_id,
                                "source_name":           source_name,
                                "signal_type":           "cold_start_prior",
                                "reliability_score":     prior,
                                "uncertainty_score":     1.0 - prior,
                                "true_positive":         0,
                                "false_positive":        0,
                                "true_negative":         0,
                                "false_negative":        0,
                                "total_outcomes":        0,
                                "effective_sample_size": 0.0,
                                "is_cold_start_prior":   True,
                            },
                        )
                    ],
                )

    def persist_conflict_note(
        self,
        entity_id: str,
        conflict_result: "ConflictResolutionResult",
    ) -> None:
        """
        Persist the outcome of a conflict resolution pass to the vector DB.

        Stored in namespace ``conflict_notes`` so future conflict resolution
        can query: "has this entity had similar contradictions before?"
        The retrieved notes feed directly into `temporal_coherence` scoring.
        """
        if not conflict_result.conflict_detected or conflict_result.winner is None:
            return
        winner = conflict_result.winner
        alt_count = len(conflict_result.alternatives)
        note_text = (
            f"Conflict resolved for {entity_id}. "
            f"Winner: {winner.interpretation} (score={winner.composite_score:.2f}, "
            f"confidence={winner.confidence.value}). "
            f"{alt_count} alternative(s) preserved. "
            f"Rationale: {conflict_result.rationale}"
        )
        doc_id = f"{entity_id}:conflict:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        self.vector_store.upsert(
            namespace="conflict_notes",
            docs=[
                VectorDocument(
                    doc_id=doc_id,
                    text=note_text,
                    metadata={
                        "entity_id":            entity_id,
                        "winner_interpretation": winner.interpretation,
                        "winner_score":          winner.composite_score,
                        "winner_confidence":     winner.confidence.value,
                        "alternatives_count":    alt_count,
                        "requires_manual_review": conflict_result.requires_manual_review,
                        "timestamp":             datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )

    def load_conflict_history(self, entity_id: str, top_k: int = 3) -> list[str]:
        """
        Return past conflict resolution summaries for an entity.
        Used by ConflictResolutionAgent to improve temporal_coherence scoring.
        """
        docs = self.vector_store.query(
            namespace="conflict_notes",
            text=entity_id,
            top_k=top_k,
            metadata_filter={"entity_id": entity_id},
        )
        return [doc.text for doc in docs]

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
                    metadata={k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in record.model_dump().items()},
                )
            ],
        )

    def load_source_reliability(self, entity_id: str) -> dict[str, float]:
        """
        Return per-source reliability scores for an entity.

        Blending strategy:
        1. Entity-specific calibration records (from observed TP/FP/TN/FN)
           are blended with global priors via Laplace smoothing
           (prior_strength = 3.0 equivalent sample units).
        2. If no entity-specific records exist at all, fall back to
           `_GLOBAL_SOURCE_PRIORS` keyed by source name — this is the
           cold-start warm-up path.
        3. If a source has a cold_start_prior record (seeded by `seed_cold_start`)
           it already carries the global prior value; observed samples will
           dilute it appropriately over time.
        """
        docs = self.vector_store.query(
            namespace="calibration",
            text=entity_id,
            top_k=50,
            metadata_filter={"entity_id": entity_id},
        )
        by_source: dict[str, float] = {}
        for doc in docs:
            source_name = str(doc.metadata.get("source_name", "")).strip()
            if not source_name:
                continue
            reliability = float(doc.metadata.get("reliability_score", 0.5))
            sample_size = float(doc.metadata.get("effective_sample_size", 0.0))
            # Use the source's known global prior as the Bayesian centre so that
            # cold-start seeds (sample_size=0) return the prior, not 0.5.
            source_prior = _GLOBAL_SOURCE_PRIORS.get(source_name.lower(), 0.60)
            by_source[source_name] = (
                (reliability * sample_size + source_prior * _PRIOR_STRENGTH) / (sample_size + _PRIOR_STRENGTH)
            )

        if not by_source:
            # Cold-start: no entity-specific calibration at all — return global priors
            return dict(_GLOBAL_SOURCE_PRIORS)

        return by_source

