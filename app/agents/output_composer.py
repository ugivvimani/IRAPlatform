from __future__ import annotations

from app.contracts import (
    AssessmentDecision,
    ConfidenceLevel,
    ConflictResolutionResult,
    EvidenceItem,
    RiskRating,
)


class OutputComposerAgent:
    def compose(
        self,
        evidence: list[EvidenceItem],
        conflict_result: ConflictResolutionResult,
        quant_score: float,
    ) -> AssessmentDecision:
        rating = RiskRating.SAFE
        confidence = ConfidenceLevel.HIGH
        summary = "No critical red flags identified from current evidence."
        steps = ["Continue monitoring with periodic reassessment."]
        requires_manual_review = False

        if conflict_result.conflict_detected and conflict_result.winner:
            requires_manual_review = conflict_result.requires_manual_review
            confidence = conflict_result.winner.confidence
            summary = f"Conflicting signals detected; selected branch: {conflict_result.winner.interpretation}."
            steps = list(conflict_result.winner.proposed_actions)

        if quant_score >= 0.65:
            rating = RiskRating.HIGH_RISK
            if confidence == ConfidenceLevel.HIGH:
                confidence = ConfidenceLevel.MEDIUM
            summary = "Quantitative and qualitative signals indicate elevated partner integrity risk."
            steps.append("Escalate to compliance review.")
        elif quant_score >= 0.35 or conflict_result.conflict_detected:
            rating = RiskRating.WATCH

        if any(item.value == "insufficient_live_data" for item in evidence):
            confidence = ConfidenceLevel.LOW
            requires_manual_review = True
            summary = "Live retrieval coverage was insufficient; decision requires analyst validation."
            steps.append("Run manual verification against sanctions and regulator portals.")

        if requires_manual_review:
            steps.append("Manual analyst review required before final decision.")

        return AssessmentDecision(
            risk_rating=rating,
            confidence=confidence,
            summary=summary,
            recommended_next_steps=steps,
            requires_manual_review=requires_manual_review,
        )
