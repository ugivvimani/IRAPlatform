from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.contracts import AssessRequest, AssessmentResponse, MemoryFact
from app.observability import build_assessment_telemetry
from app.storage.base import StorageRepository

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    def __init__(
        self,
        retrieval: RetrievalAgent,
        analysis: AnalysisForecastingAgent,
        conflict: ConflictResolutionAgent,
        memory: MemoryManagerAgent,
        composer: OutputComposerAgent,
        calibration: CalibrationAgent,
        llm_client=None,
        storage_repo: StorageRepository | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.analysis = analysis
        self.conflict = conflict
        self.memory = memory
        self.composer = composer
        self.calibration = calibration
        self.storage_repo = storage_repo
        # Wire LLM into all agents that support it
        if llm_client is not None:
            if not self.conflict.llm_client:
                self.conflict.llm_client = llm_client
            if not self.composer.llm_client:
                self.composer.llm_client = llm_client
            if not self.analysis.llm_client:
                self.analysis.llm_client = llm_client
            if not self.memory.llm_client:
                self.memory.llm_client = llm_client

    def _apply_policy_thresholds(self) -> None:
        """Load active policy thresholds from storage and push them into agents."""
        if self.storage_repo is None:
            return
        try:
            thresholds = self.storage_repo.get_active_policy_thresholds()
            if "auto_hold_threshold" in thresholds:
                self.composer.AUTO_HOLD_THRESHOLD = thresholds["auto_hold_threshold"].threshold_value
            if "tot_stable_threshold" in thresholds:
                self.conflict.TOT_STABLE_THRESHOLD = thresholds["tot_stable_threshold"].threshold_value
            if "tot_beam_width" in thresholds:
                self.conflict.TOT_BEAM_WIDTH = max(1, int(thresholds["tot_beam_width"].threshold_value))
        except Exception as exc:
            logger.warning("policy_threshold_load_failed error=%s (using agent defaults)", exc)

    def assess(self, request: AssessRequest) -> AssessmentResponse:
        started = time.perf_counter()
        entity_id = request.query.company_name
        logger.info("assessment_stage=start entity=%s", entity_id)

        # Apply any policy threshold overrides stored in DB before reasoning starts
        self._apply_policy_thresholds()

        # Think: initialize context and gather prior memory.
        logger.info("assessment_stage=retrieval_start entity=%s", entity_id)
        evidence = self.retrieval.retrieve(request.query.company_name, request.evidence)
        logger.info("assessment_stage=retrieval_done entity=%s evidence_count=%d", entity_id, len(evidence))

        logger.info("assessment_stage=memory_init_start entity=%s", entity_id)
        self.memory.initialize_working_memory(request.query, evidence)
        historical = self.memory.load_historical_context(
            entity_id=request.query.company_name,
            top_k=5,
            query=request.query.question or request.query.company_name,
        )

        # Cold-start warm-up: seed global priors for any source that has no history yet.
        # This ensures conflict resolution starts from informed reliability weights,
        # not blind 0.5 defaults, even on the very first encounter with an entity.
        is_cold_start = len(historical) == 0
        if is_cold_start:
            live_source_names = list({e.source_name for e in evidence if e.source_name.lower() != "system"})
            self.memory.seed_cold_start(entity_id=request.query.company_name, source_names=live_source_names)
            logger.info("assessment_stage=cold_start_seeded entity=%s sources=%s", entity_id, live_source_names)

        source_reliability = self.memory.load_source_reliability(entity_id=request.query.company_name)
        # Load past conflict notes so the conflict agent can boost temporal_coherence
        # when it recognises a recurring contradiction pattern for this entity.
        conflict_history = self.memory.load_conflict_history(entity_id=request.query.company_name, top_k=3)
        logger.info(
            "assessment_stage=memory_init_done entity=%s historical_count=%d source_reliability_count=%d cold_start=%s conflict_history=%d",
            entity_id,
            len(historical),
            len(source_reliability),
            is_cold_start,
            len(conflict_history),
        )

        # Observe + Revise: conflict handling.
        logger.info("assessment_stage=conflict_resolution_start entity=%s", entity_id)
        conflict_result = self.conflict.resolve(
            evidence, historical,
            source_reliability=source_reliability,
            conflict_history=conflict_history,
        )
        logger.info(
            "assessment_stage=conflict_resolution_done entity=%s conflict_detected=%s alternatives=%d",
            entity_id,
            conflict_result.conflict_detected,
            len(conflict_result.alternatives),
        )

        # ── Revise step ───────────────────────────────────────────────────────
        # Re-retrieve targeted supplemental evidence when:
        #   a) conflict was detected, OR
        #   b) winning branch score is below 0.75 (weak / uncertain resolution)
        # Supplemental items are merged into evidence before scoring so the
        # quantitative pass and memory persist benefit from the fuller picture.
        winner_score = conflict_result.winner.composite_score if conflict_result.winner else 1.0
        needs_revise = conflict_result.conflict_detected or winner_score < 0.75
        if needs_revise:
            supplemental = self.retrieval.retrieve_supplemental(
                query_company=request.query.company_name,
                conflict_result=conflict_result,
                existing_evidence=evidence,
            )
            if supplemental:
                evidence = evidence + supplemental
                logger.info(
                    "assessment_stage=revise_done entity=%s supplemental_items=%d total_evidence=%d",
                    entity_id, len(supplemental), len(evidence),
                )

        # Act: quantitative scoring on conflict-resolved evidence only.
        # Suppressed items (rejected by conflict resolution) are excluded so the
        # composite score reflects the resolved interpretation, not contradicted signals.
        logger.info("assessment_stage=scoring_start entity=%s", entity_id)
        suppressed = set(conflict_result.suppressed_evidence_ids)
        scoring_evidence = [e for e in evidence if e.evidence_id not in suppressed]
        if suppressed:
            logger.info(
                "assessment_stage=scoring_evidence_filtered entity=%s suppressed=%d scoring=%d",
                entity_id, len(suppressed), len(scoring_evidence),
            )
        quant_scores = self.analysis.score(scoring_evidence)
        quant = quant_scores["composite_quant_score"]
        decision = self.composer.compose(evidence, conflict_result, quant_score=quant)
        escalation = self.composer.build_escalation_context(decision, conflict_result, evidence, quant)
        logger.info(
            "assessment_stage=scoring_done entity=%s quant=%.4f risk=%s confidence=%s manual_review=%s",
            entity_id,
            quant,
            decision.risk_rating.value,
            decision.confidence.value,
            decision.requires_manual_review,
        )

        # Conclude: persist stable memory and calibration artifacts.
        # Use scoring_evidence (conflict-resolved, suppressed items removed) so memory
        # only stores facts that survived conflict resolution, not the rejected side.
        # Cap at 10 to give broader source coverage while avoiding noise.
        logger.info("assessment_stage=persist_start entity=%s", entity_id)
        now = datetime.now(timezone.utc)
        stable_facts = [
            MemoryFact(
                fact_id=f"{request.query.company_name}-{idx}",
                entity_id=request.query.company_name,
                # Use raw_content when available — gives the LLM summarizer rich prose
                # to compress. Falls back to the structured key-value signal.
                summary=item.raw_content or f"{item.dimension.value}:{item.signal}={item.value}",
                dimension=item.dimension,
                severity=1.0 - item.source_confidence,
                source_reference=item.provenance_url,
                timestamp=item.timestamp,
            )
            for idx, item in enumerate(scoring_evidence[:10], start=1)
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
        # Persist conflict resolution note so future assessments can check whether
        # similar contradictions have occurred before (feeds temporal_coherence scoring).
        if conflict_result.conflict_detected:
            self.memory.persist_conflict_note(request.query.company_name, conflict_result)
        # Persist the LLM-generated assessment narrative for semantic retrieval of similar past cases.
        self.memory.persist_assessment_narrative(
            entity_id=request.query.company_name,
            risk_rating=decision.risk_rating.value,
            confidence=decision.confidence.value,
            summary=decision.summary,
            requires_manual_review=decision.requires_manual_review,
        )
        logger.info("assessment_stage=persist_done entity=%s memory_records_written=%d", entity_id, len(stable_facts))

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("assessment_stage=complete entity=%s total_ms=%d", entity_id, elapsed_ms)

        return AssessmentResponse(
            query=request.query,
            decision=decision,
            evidence_chain=evidence,
            conflict_result=conflict_result if conflict_result.conflict_detected else None,
            escalation=escalation,
            model_metadata={
                "quant_scores": quant_scores,
                "memory_records_written": len(stable_facts),
                "evaluated_at": now.isoformat(),
                "telemetry": build_assessment_telemetry(
                    request=request,
                    decision=decision,
                    conflict_result=conflict_result if conflict_result.conflict_detected else None,
                    evidence_count=len(evidence),
                    quant_scores=quant_scores,
                    evaluated_at=now,
                ),
            },
        )
