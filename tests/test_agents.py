from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.contracts import (
    AssessRequest,
    CalibrationRecord,
    CriticScoreVector,
    EvidenceItem,
    RiskDimension,
    UserQuery,
)
from app.policy import composite_score, recency_score, source_priority
from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent, RetrievalConnector, RetrievalObservation
from app.orchestrator import OrchestratorAgent
from app.vector_store.pinecone_store import PineconeVectorStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _ev(
    evidence_id: str,
    dimension: RiskDimension,
    signal: str,
    value: str,
    source_name: str,
    source_tier: str,
    confidence: float = 0.9,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        dimension=dimension,
        signal=signal,
        value=value,
        source_name=source_name,
        source_tier=source_tier,
        timestamp=datetime.now(timezone.utc),
        entity_match_confidence=confidence,
        source_confidence=confidence,
        provenance_url=f"https://example.org/{source_name.lower()}",
        metadata={},
    )


@pytest.fixture()
def orchestrator_instance():
    vs = PineconeVectorStore()
    mem = MemoryManagerAgent(vs)
    return OrchestratorAgent(
        retrieval=RetrievalAgent(),
        analysis=AnalysisForecastingAgent(),
        conflict=ConflictResolutionAgent(),
        memory=mem,
        composer=OutputComposerAgent(),
        calibration=CalibrationAgent(),
    )


# ── policy ────────────────────────────────────────────────────────────────────

class TestPolicy:
    def test_source_priority_ordering(self):
        from app.contracts import SourceTier
        assert source_priority(SourceTier.OFFICIAL) > source_priority(SourceTier.TIER1_NEWS)
        assert source_priority(SourceTier.TIER1_NEWS) > source_priority(SourceTier.SECONDARY)

    def test_recency_score_decays(self):
        from datetime import timedelta
        old = datetime.now(timezone.utc) - timedelta(days=60)
        recent = datetime.now(timezone.utc)
        assert recency_score(recent) > recency_score(old)

    def test_composite_score_bounded(self):
        vec = CriticScoreVector(
            authority=1.0, recency=1.0, entity_certainty=1.0,
            corroboration=1.0, temporal_coherence=1.0,
            contradiction_penalty=0.0, evidence_sufficiency_penalty=0.0,
        )
        score = composite_score(vec)
        assert 0.0 <= score <= 1.0

    def test_composite_penalizes_contradiction(self):
        base = CriticScoreVector(
            authority=0.8, recency=0.8, entity_certainty=0.8,
            corroboration=0.8, temporal_coherence=0.8,
            contradiction_penalty=0.0, evidence_sufficiency_penalty=0.0,
        )
        penalized = base.model_copy(update={"contradiction_penalty": 1.0})
        assert composite_score(penalized) < composite_score(base)


# ── retrieval ─────────────────────────────────────────────────────────────────

class TestRetrieval:
    def test_seeded_evidence_returned_as_is(self):
        agent = RetrievalAgent()
        seeded = [_ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official")]
        result = agent.retrieve("Test Co", seeded)
        assert result == seeded

    def test_default_connectors_produce_evidence(self):
        agent = RetrievalAgent()
        result = agent.retrieve("Test Co", [])
        assert len(result) > 0

    def test_connectors_produce_expected_dimensions(self):
        agent = RetrievalAgent()
        evidence = agent.retrieve("Test Co", [])
        dims = {e.dimension for e in evidence}
        assert RiskDimension.SANCTIONS in dims

    def test_conflict_detection(self):
        ev = [
            _ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official"),
            _ev("e2", RiskDimension.SANCTIONS, "sanctions_status", "reported_sanctioned", "Reuters", "tier1_news"),
        ]
        assert RetrievalAgent.detect_conflicts(ev) is True

    def test_no_conflict_when_signals_agree(self):
        ev = [
            _ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official"),
            _ev("e2", RiskDimension.REGULATORY, "regulatory_enforcement", "no_recent_enforcement", "SEC", "regulator"),
        ]
        assert RetrievalAgent.detect_conflicts(ev) is False

    def test_entity_resolution_marks_ambiguous_match(self):
        class _BadMatchConnector(RetrievalConnector):
            connector_name = "bad_match"

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
                        entity_match_confidence=0.95,
                        source_confidence=0.9,
                        metadata={"matched_name": "Completely Unrelated Entity"},
                    )
                ]

        agent = RetrievalAgent()
        agent.connectors = [_BadMatchConnector()]
        evidence = agent.retrieve("Apple Inc", [])
        assert any(item.signal == "entity_resolution_ambiguous" for item in evidence)


# ── analysis forecasting ──────────────────────────────────────────────────────

class TestAnalysis:
    def test_safe_evidence_low_score(self):
        ev = [
            _ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official"),
            _ev("e2", RiskDimension.ESG, "esg_incident", "none_material", "ESGDB", "secondary"),
        ]
        scores = AnalysisForecastingAgent().score(ev)
        assert scores["composite_quant_score"] == 0.0

    def test_sanctions_flag_raises_score(self):
        ev = [_ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "sanctioned_entity", "OFAC", "official")]
        scores = AnalysisForecastingAgent().score(ev)
        assert scores["composite_quant_score"] > 0.0

    def test_retrieval_failure_penalizes_score(self):
        ev = [_ev("e1", RiskDimension.OPERATIONAL, "retrieval_health", "insufficient_live_data", "system", "secondary")]
        scores = AnalysisForecastingAgent().score(ev)
        assert scores["composite_quant_score"] > 0.0


# ── conflict resolution ───────────────────────────────────────────────────────

class TestConflictResolution:
    def test_no_conflict_when_signals_uniform(self):
        ev = [_ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official")]
        result = ConflictResolutionAgent().resolve(ev, [])
        assert not result.conflict_detected

    def test_conflict_detected_on_differing_values(self):
        ev = [
            _ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official"),
            _ev("e2", RiskDimension.SANCTIONS, "sanctions_status", "reported_sanctioned", "Reuters", "tier1_news"),
        ]
        result = ConflictResolutionAgent().resolve(ev, [])
        assert result.conflict_detected
        assert result.winner is not None

    def test_cold_start_forces_manual_review(self):
        ev = [
            _ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official"),
            _ev("e2", RiskDimension.SANCTIONS, "sanctions_status", "reported_sanctioned", "Reuters", "tier1_news"),
        ]
        result = ConflictResolutionAgent().resolve(ev, historical_facts=[])
        assert result.requires_manual_review

    def test_official_source_wins_conflict(self):
        ev = [
            _ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official", 0.95),
            _ev("e2", RiskDimension.SANCTIONS, "sanctions_status", "reported_sanctioned", "Blog", "secondary", 0.4),
        ]
        result = ConflictResolutionAgent().resolve(ev, [])
        assert result.winner is not None
        assert "OFAC" in result.winner.interpretation

    def test_tot_beam_width_is_bounded(self):
        ev = [
            _ev("e1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official", 0.95),
            _ev("e2", RiskDimension.SANCTIONS, "sanctions_status", "reported_sanctioned", "Reuters", "tier1_news", 0.85),
            _ev("e3", RiskDimension.SANCTIONS, "sanctions_status", "under_review", "BlogA", "secondary", 0.8),
            _ev("e4", RiskDimension.SANCTIONS, "sanctions_status", "possible_match", "BlogB", "secondary", 0.79),
        ]
        result = ConflictResolutionAgent().resolve(ev, [])
        assert result.winner is not None
        assert len(result.alternatives) <= 2
        assert "beam width 3" in result.rationale


# ── calibration ───────────────────────────────────────────────────────────────

class TestCalibration:
    def _make_safe_result(self):
        from app.contracts import AssessmentDecision, ConflictResolutionResult, RiskRating, ConfidenceLevel
        return (
            AssessmentDecision(risk_rating=RiskRating.SAFE, confidence=ConfidenceLevel.HIGH,
                               summary="safe", recommended_next_steps=[]),
            ConflictResolutionResult(conflict_detected=False, rationale="none"),
        )

    def test_beta_smoothed_reliability_between_0_1(self):
        agent = CalibrationAgent()
        decision, conflict = self._make_safe_result()
        record = agent.build_record("TestCo", "test_source", decision, conflict, [])
        assert 0.0 < record.reliability_score <= 1.0

    def test_manual_review_reduces_reliability(self):
        from app.contracts import AssessmentDecision, ConflictResolutionResult, RiskRating, ConfidenceLevel
        agent = CalibrationAgent()
        # Use no-conflict scenario so both records score as TP (risky, uncontested).
        # manual_review=True reduces effective_sample_size by 0.7, lowering the
        # weighted success contribution and therefore the reliability score.
        no_conflict = ConflictResolutionResult(conflict_detected=False, rationale="no conflict")
        decision_mr = AssessmentDecision(
            risk_rating=RiskRating.WATCH, confidence=ConfidenceLevel.MEDIUM,
            summary="watch", recommended_next_steps=[], requires_manual_review=True,
        )
        decision_clean = AssessmentDecision(
            risk_rating=RiskRating.WATCH, confidence=ConfidenceLevel.MEDIUM,
            summary="watch", recommended_next_steps=[], requires_manual_review=False,
        )
        r_mr = agent.build_record("TestCo", "s", decision_mr, no_conflict, [])
        r_clean = agent.build_record("TestCo", "s", decision_clean, no_conflict, [])
        assert r_mr.reliability_score < r_clean.reliability_score

    def test_sparse_evidence_keeps_calibration_uncertain(self):
        from app.contracts import AssessmentDecision, ConflictResolutionResult, RiskRating, ConfidenceLevel
        agent = CalibrationAgent()
        decision = AssessmentDecision(
            risk_rating=RiskRating.WATCH,
            confidence=ConfidenceLevel.MEDIUM,
            summary="watch",
            recommended_next_steps=[],
            requires_manual_review=True,
        )
        conflict = ConflictResolutionResult(conflict_detected=True, rationale="conflict")
        sparse_evidence = [
            _ev("sp1", RiskDimension.SANCTIONS, "sanctions_status", "reported_sanctioned", "SingleSource", "secondary", 0.55)
        ]
        record = agent.build_record("SparseCo", "single_source", decision, conflict, sparse_evidence)
        assert record.uncertainty_score > 0.6
        assert 0.45 <= record.reliability_score <= 0.65


# ── orchestrator integration ──────────────────────────────────────────────────

class TestOrchestratorIntegration:
    def test_assess_returns_valid_response(self, orchestrator_instance):
        req = AssessRequest(query=UserQuery(company_name="IntegCo", question="Is this company safe?"))
        result = orchestrator_instance.assess(req)
        assert result.decision.risk_rating is not None
        assert result.decision.confidence is not None
        assert result.escalation is not None

    def test_assess_with_safe_evidence_not_restricted(self, orchestrator_instance):
        req = AssessRequest(
            query=UserQuery(company_name="SafeCo", question="safe?"),
            evidence=[
                _ev("s1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official", 0.95),
                _ev("s2", RiskDimension.REGULATORY, "regulatory_enforcement", "no_recent_enforcement", "SEC", "regulator", 0.9),
            ],
        )
        result = orchestrator_instance.assess(req)
        assert result.decision.risk_rating.value != "restricted"

    def test_conflict_evidence_triggers_review(self, orchestrator_instance):
        req = AssessRequest(
            query=UserQuery(company_name="ConflictCo", question="safe?"),
            evidence=[
                _ev("c1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official"),
                _ev("c2", RiskDimension.SANCTIONS, "sanctions_status", "reported_sanctioned", "Reuters", "tier1_news"),
            ],
        )
        result = orchestrator_instance.assess(req)
        assert result.decision.requires_manual_review

    def test_assess_persists_memory(self, orchestrator_instance):
        req = AssessRequest(query=UserQuery(company_name="MemCo", question="check?"))
        orchestrator_instance.assess(req)
        history = orchestrator_instance.memory.load_historical_context("MemCo", top_k=5)
        assert len(history) > 0

    def test_source_reliability_shrinks_for_low_sample_size(self):
        vs = PineconeVectorStore()
        mem = MemoryManagerAgent(vs)
        rec = CalibrationRecord(
            calibration_id="cal-1",
            entity_id="EntityA",
            source_name="LowSampleSource",
            signal_type="assessment_outcome",
            true_positive=1,
            false_positive=0,
            true_negative=0,
            false_negative=0,
            total_outcomes=1,
            effective_sample_size=0.1,
            uncertainty_score=1.0,
            reliability_score=1.0,
            updated_at=datetime.now(timezone.utc),
        )
        mem.persist_calibration(rec)
        reliabilities = mem.load_source_reliability("EntityA")
        assert 0.49 <= reliabilities["LowSampleSource"] <= 0.6

    def test_entity_resolution_ambiguity_triggers_escalation_reason(self, orchestrator_instance):
        req = AssessRequest(
            query=UserQuery(company_name="Apple Inc", question="safe?"),
            evidence=[
                _ev("a1", RiskDimension.SANCTIONS, "entity_resolution_ambiguous", "requires_review", "system", "secondary"),
            ],
        )
        result = orchestrator_instance.assess(req)
        assert result.escalation is not None
        assert "ENTITY_RESOLUTION_REVIEW_REQUIRED" in result.escalation.reason_codes

