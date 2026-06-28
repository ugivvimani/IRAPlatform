from __future__ import annotations

import logging

from app.contracts import (
    AssessmentDecision,
    ConfidenceLevel,
    ConflictResolutionResult,
    EscalationContext,
    EvidenceItem,
    RiskRating,
)

logger = logging.getLogger(__name__)


class OutputComposerAgent:
    AUTO_HOLD_THRESHOLD = 0.75

    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client

    def _llm_summary(
        self,
        evidence: list[EvidenceItem],
        rating: RiskRating,
        quant_score: float,
        conflict_detected: bool,
    ) -> str | None:
        if not self.llm_client:
            return None
        try:
            ev_lines = "\n".join(
                f"- [{e.source_name}/{e.dimension.value}] {e.signal}={e.value} (confidence={e.source_confidence:.2f})"
                for e in evidence[:10]
            )
            user_prompt = (
                f"Risk rating: {rating.value}\n"
                f"Quantitative risk score: {quant_score:.2f}\n"
                f"Conflict detected: {conflict_detected}\n"
                f"Evidence:\n{ev_lines}\n\n"
                "Write a 2-3 sentence plain-English assessment summary a compliance analyst would understand. "
                "Be factual and concise. Do not include risk scores or technical identifiers."
            )
            return self.llm_client.complete(
                system_prompt="You are a compliance risk analyst summarizing an integrity risk assessment.",
                user_prompt=user_prompt,
                max_tokens=180,
            )
        except Exception as exc:
            logger.warning("LLM summary generation failed: %s", exc)
            return None

    def _llm_next_steps(self, rating: RiskRating, reason_codes: list[str]) -> list[str] | None:
        if not self.llm_client or rating == RiskRating.SAFE:
            return None
        try:
            user_prompt = (
                f"Risk rating: {rating.value}\n"
                f"Escalation reason codes: {', '.join(reason_codes) or 'none'}\n\n"
                "List 2-3 concise recommended next steps a compliance team should take. "
                "Each step should be a single action sentence. No bullets or numbering."
            )
            text = self.llm_client.complete(
                system_prompt="You are a compliance risk analyst recommending next steps after an integrity risk assessment.",
                user_prompt=user_prompt,
                max_tokens=150,
            )
            steps = [s.strip() for s in text.strip().split("\n") if s.strip()]
            return steps[:3] if steps else None
        except Exception as exc:
            logger.warning("LLM next-steps generation failed: %s", exc)
            return None

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

        if any(item.signal == "entity_not_found" for item in evidence):
            confidence = ConfidenceLevel.LOW
            requires_manual_review = True
            rating = RiskRating.WATCH  # Cannot confirm safe without any data
            summary = (
                "No records found for this entity across sanctions, regulatory, news, "
                "and financial sources. This may indicate a private, foreign, or "
                "unregistered entity. Manual verification is required before proceeding."
            )
            steps = [
                "Verify the exact legal entity name and any known aliases.",
                "Search manually against OFAC, EU, and UN sanctions portals.",
                "Request counterparty documentation (registration, ownership structure).",
            ]

        if any(item.value == "insufficient_live_data" for item in evidence):
            confidence = ConfidenceLevel.LOW
            requires_manual_review = True
            summary = "Live retrieval coverage was insufficient; decision requires analyst validation."
            steps.append("Run manual verification against sanctions and regulator portals.")

        if any(item.signal == "entity_resolution_ambiguous" for item in evidence):
            requires_manual_review = True
            if confidence == ConfidenceLevel.HIGH:
                confidence = ConfidenceLevel.MEDIUM
            summary = "Entity matching was ambiguous; decision requires analyst validation."
            steps.append("Validate entity identity and aliases before final action.")

        if requires_manual_review:
            steps.append("Manual analyst review required before final decision.")

        # LLM-enhanced summary and next steps (override rule-based defaults when available)
        llm_summary = self._llm_summary(evidence, rating, quant_score, conflict_result.conflict_detected)
        if llm_summary:
            summary = llm_summary

        reason_codes_preview = []
        if quant_score >= self.AUTO_HOLD_THRESHOLD:
            reason_codes_preview.append("HIGH_RISK_SCORE")
        if requires_manual_review:
            reason_codes_preview.append("MANUAL_REVIEW_REQUIRED")
        llm_steps = self._llm_next_steps(rating, reason_codes_preview)
        if llm_steps:
            steps = llm_steps

        return AssessmentDecision(
            risk_rating=rating,
            confidence=confidence,
            summary=summary,
            recommended_next_steps=steps,
            requires_manual_review=requires_manual_review,
        )

    def build_escalation_context(
        self,
        decision: AssessmentDecision,
        conflict_result: ConflictResolutionResult,
        evidence: list[EvidenceItem],
        quant_score: float,
    ) -> EscalationContext:
        reason_codes: list[str] = []
        has_insufficient_data = any(item.value == "insufficient_live_data" for item in evidence)
        has_entity_resolution_review = any(
            item.signal == "entity_resolution_ambiguous"
            or bool(item.metadata.get("entity_resolution_requires_review"))
            for item in evidence
        )

        if quant_score >= self.AUTO_HOLD_THRESHOLD:
            reason_codes.append("RISK_SCORE_AT_OR_ABOVE_AUTO_HOLD_THRESHOLD")
        if decision.requires_manual_review:
            reason_codes.append("DECISION_REQUIRES_MANUAL_REVIEW")
        if has_insufficient_data:
            reason_codes.append("INSUFFICIENT_LIVE_DATA")
        if has_entity_resolution_review:
            reason_codes.append("ENTITY_RESOLUTION_REVIEW_REQUIRED")
        if conflict_result.conflict_detected:
            reason_codes.append("CONFLICTING_EVIDENCE_PRESENT")
            if conflict_result.requires_manual_review:
                reason_codes.append("CONFLICT_REQUIRES_REVIEW")

        escalation_required = len(reason_codes) > 0
        auto_hold = quant_score >= self.AUTO_HOLD_THRESHOLD or has_insufficient_data or has_entity_resolution_review

        message = (
            "Escalation required. Please review the evidence trail and conflict notes before taking action."
            if escalation_required
            else "No escalation required."
        )

        return EscalationContext(
            escalation_required=escalation_required,
            auto_hold=auto_hold,
            risk_score=max(0.0, min(quant_score, 1.0)),
            threshold=self.AUTO_HOLD_THRESHOLD,
            reason_codes=reason_codes,
            review_message=message,
        )
