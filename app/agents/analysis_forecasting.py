from __future__ import annotations

from app.contracts import EvidenceItem, RiskDimension


class AnalysisForecastingAgent:
    """Quantitative scoring placeholder for phase-5 model integration."""

    def score(self, evidence: list[EvidenceItem]) -> dict[str, float]:
        sanctions_red_flags = sum(
            1
            for item in evidence
            if item.dimension == RiskDimension.SANCTIONS and "sanction" in item.value.lower()
        )
        reputational_flags = sum(
            1
            for item in evidence
            if item.dimension == RiskDimension.REPUTATIONAL and "negative" in item.value.lower()
        )
        regulatory_flags = sum(
            1
            for item in evidence
            if item.dimension == RiskDimension.REGULATORY and "enforcement" in item.signal.lower() and "no_recent" not in item.value.lower()
        )
        retrieval_penalty = sum(1 for item in evidence if item.signal == "retrieval_health")
        raw = min(1.0, sanctions_red_flags * 0.35 + reputational_flags * 0.2 + regulatory_flags * 0.2 + retrieval_penalty * 0.25)
        return {
            "composite_quant_score": raw,
            "financial_distress_risk": min(raw + 0.15, 1.0),
            "liquidity_risk": min(raw + 0.1, 1.0),
            "regulatory_pressure_risk": min(raw + regulatory_flags * 0.1, 1.0),
        }
