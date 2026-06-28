from __future__ import annotations

from datetime import datetime, timezone

from app.contracts import (
    AssessmentDecision,
    CalibrationRecord,
    ConflictResolutionResult,
    EvidenceItem,
    RiskRating,
)


class CalibrationAgent:
    """Builds calibration updates from each completed assessment."""

    def build_record(
        self,
        entity_id: str,
        source_name: str,
        decision: AssessmentDecision,
        conflict_result: ConflictResolutionResult,
        evidence: list[EvidenceItem],
    ) -> CalibrationRecord:
        predicted_risky = decision.risk_rating in {RiskRating.WATCH, RiskRating.HIGH_RISK, RiskRating.RESTRICTED}
        conflict_present = conflict_result.conflict_detected
        insufficient_data = any(item.value == "insufficient_live_data" for item in evidence)
        evidence_count = len(evidence)
        source_diversity = len({item.source_name for item in evidence})
        avg_signal_quality = (
            sum(item.entity_match_confidence * item.source_confidence for item in evidence) / evidence_count
            if evidence_count
            else 0.5
        )

        # Outcome classification:
        # TP: predicted risky with uncontested evidence (confident, reliable finding)
        # FP: predicted risky but evidence was contested (risky call under uncertainty)
        # TN: predicted safe with clean evidence (no signals, no conflict)
        # FN: predicted safe despite conflicting signals (possible missed risk)
        true_positive = 1 if predicted_risky and not conflict_present else 0
        false_positive = 1 if predicted_risky and conflict_present else 0
        true_negative = 1 if (not predicted_risky and not conflict_present and not insufficient_data) else 0
        false_negative = 1 if (not predicted_risky and conflict_present) else 0
        total_outcomes = true_positive + false_positive + true_negative + false_negative

        # Weight sparse/noisy outcomes down to avoid overfitting early calibrations.
        support = min((evidence_count + source_diversity) / 6.0, 1.0)
        effective_sample_size = max(0.1, support * avg_signal_quality)
        if insufficient_data:
            effective_sample_size *= 0.5
        if decision.requires_manual_review:
            effective_sample_size *= 0.7

        weighted_success = (true_positive + true_negative) * effective_sample_size
        weighted_failures = (false_positive + false_negative) * effective_sample_size

        # Beta(1,1) posterior mean keeps reliability centered at 0.5 with low evidence.
        reliability = (1.0 + weighted_success) / (2.0 + weighted_success + weighted_failures)
        uncertainty = 1.0 / (1.0 + weighted_success + weighted_failures)

        return CalibrationRecord(
            calibration_id=f"{entity_id}-{datetime.now(timezone.utc).date()}",
            entity_id=entity_id,
            source_name=source_name,
            signal_type="assessment_outcome",
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
            total_outcomes=total_outcomes,
            effective_sample_size=effective_sample_size,
            uncertainty_score=max(0.0, min(uncertainty, 1.0)),
            reliability_score=max(0.0, min(reliability, 1.0)),
            updated_at=datetime.now(timezone.utc),
        )
