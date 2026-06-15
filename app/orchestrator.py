from __future__ import annotations

from datetime import datetime, timezone

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.retrieval import RetrievalAgent
from app.contracts import (
    AssessRequest,
    AssessmentDecision,
    AssessmentResponse,
    CalibrationRecord,
    ConfidenceLevel,
    MemoryFact,
    RiskRating,
)


class OrchestratorAgent:
    def __init__(
        self,
        retrieval: RetrievalAgent,
        analysis: AnalysisForecastingAgent,
        conflict: ConflictResolutionAgent,
        memory: MemoryManagerAgent,
    ) -> None:
        self.retrieval = retrieval
        self.analysis = analysis
        self.conflict = conflict
        self.memory = memory

    def assess(self, request: AssessRequest) -> AssessmentResponse:
        # Think: initialize context and gather prior memory.
        evidence = self.retrieval.retrieve(request.query.company_name, request.evidence)
        self.memory.initialize_working_memory(request.query, evidence)
        historical = self.memory.load_historical_context(entity_id=request.query.company_name, top_k=5)

        # Observe + Revise: conflict handling.
        conflict_result = self.conflict.resolve(evidence, historical)

        # Act: quantitative scoring.
        quant_scores = self.analysis.score(evidence)
        quant = quant_scores["composite_quant_score"]

        rating = RiskRating.SAFE
        confidence = ConfidenceLevel.HIGH
        requires_manual_review = False
        summary = "No critical red flags identified from current evidence."
        next_steps = ["Continue monitoring with periodic reassessment."]

        if conflict_result.conflict_detected and conflict_result.winner:
            requires_manual_review = conflict_result.requires_manual_review
            confidence = conflict_result.winner.confidence
            summary = (
                f"Conflicting signals detected; primary branch selected: "
                f"{conflict_result.winner.interpretation}."
            )
            next_steps = conflict_result.winner.proposed_actions

        if quant >= 0.65:
            rating = RiskRating.HIGH_RISK
            confidence = ConfidenceLevel.MEDIUM if confidence == ConfidenceLevel.HIGH else confidence
            summary = "Quantitative indicators point to elevated risk."
            next_steps.append("Escalate to compliance review.")
        elif quant >= 0.35 or conflict_result.conflict_detected:
            rating = RiskRating.WATCH

        if requires_manual_review:
            next_steps.append("Manual analyst review required before final decision.")

        decision = AssessmentDecision(
            risk_rating=rating,
            confidence=confidence,
            summary=summary,
            recommended_next_steps=next_steps,
            requires_manual_review=requires_manual_review,
        )

        # Conclude: persist stable memory and calibration artifacts.
        now = datetime.now(timezone.utc)
        stable_facts = [
            MemoryFact(
                fact_id=f"{request.query.company_name}-{idx}",
                entity_id=request.query.company_name,
                summary=f"{item.dimension.value}:{item.signal}={item.value}",
                dimension=item.dimension,
                severity=1.0 - item.source_confidence,
                source_reference=item.provenance_url,
                timestamp=item.timestamp,
            )
            for idx, item in enumerate(evidence[:5], start=1)
        ]
        self.memory.persist_facts(stable_facts)
        self.memory.persist_calibration(
            CalibrationRecord(
                calibration_id=f"{request.query.company_name}-{now.date()}",
                entity_id=request.query.company_name,
                source_name="system",
                signal_type="assessment_outcome",
                true_positive=0,
                false_positive=0,
                true_negative=1 if rating == RiskRating.SAFE else 0,
                false_negative=0,
                reliability_score=0.75 if rating != RiskRating.HIGH_RISK else 0.6,
                updated_at=now,
            )
        )

        return AssessmentResponse(
            query=request.query,
            decision=decision,
            evidence_chain=evidence,
            conflict_result=conflict_result if conflict_result.conflict_detected else None,
            model_metadata={"quant_scores": quant_scores, "memory_records_written": len(stable_facts)},
        )
