from __future__ import annotations

from datetime import datetime, timezone

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.contracts import (
    AssessRequest,
    AssessmentResponse,
    MemoryFact,
)


class OrchestratorAgent:
    def __init__(
        self,
        retrieval: RetrievalAgent,
        analysis: AnalysisForecastingAgent,
        conflict: ConflictResolutionAgent,
        memory: MemoryManagerAgent,
        composer: OutputComposerAgent,
        calibration: CalibrationAgent,
    ) -> None:
        self.retrieval = retrieval
        self.analysis = analysis
        self.conflict = conflict
        self.memory = memory
        self.composer = composer
        self.calibration = calibration

    def assess(self, request: AssessRequest) -> AssessmentResponse:
        # Think: initialize context and gather prior memory.
        evidence = self.retrieval.retrieve(request.query.company_name, request.evidence)
        self.memory.initialize_working_memory(request.query, evidence)
        historical = self.memory.load_historical_context(entity_id=request.query.company_name, top_k=5)
        source_reliability = self.memory.load_source_reliability(entity_id=request.query.company_name)

        # Observe + Revise: conflict handling.
        conflict_result = self.conflict.resolve(evidence, historical, source_reliability=source_reliability)

        # Act: quantitative scoring.
        quant_scores = self.analysis.score(evidence)
        quant = quant_scores["composite_quant_score"]
        decision = self.composer.compose(evidence, conflict_result, quant_score=quant)

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
            self.calibration.build_record(
                entity_id=request.query.company_name,
                source_name="system_orchestrator",
                decision=decision,
                conflict_result=conflict_result,
                evidence=evidence,
            )
        )

        return AssessmentResponse(
            query=request.query,
            decision=decision,
            evidence_chain=evidence,
            conflict_result=conflict_result if conflict_result.conflict_detected else None,
            model_metadata={
                "quant_scores": quant_scores,
                "memory_records_written": len(stable_facts),
                "evaluated_at": now.isoformat(),
            },
        )
