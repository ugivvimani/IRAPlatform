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

        true_positive = 1 if predicted_risky and conflict_present else 0
        false_positive = 1 if predicted_risky and not conflict_present else 0
        true_negative = 1 if (not predicted_risky and not conflict_present and not insufficient_data) else 0
        false_negative = 1 if (not predicted_risky and conflict_present) else 0

        # Beta-smoothed reliability prior of 0.5 to avoid early instability.
        reliability = (true_positive + true_negative + 1) / (
            true_positive + true_negative + false_positive + false_negative + 2
        )
        if decision.requires_manual_review:
            reliability *= 0.9

        return CalibrationRecord(
            calibration_id=f"{entity_id}-{datetime.now(timezone.utc).date()}",
            entity_id=entity_id,
            source_name=source_name,
            signal_type="assessment_outcome",
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
            reliability_score=max(0.0, min(reliability, 1.0)),
            updated_at=datetime.now(timezone.utc),
        )
