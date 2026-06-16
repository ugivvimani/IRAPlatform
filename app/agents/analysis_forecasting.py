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
        raw = min(1.0, sanctions_red_flags * 0.35 + reputational_flags * 0.2)
        return {
            "composite_quant_score": raw,
            "financial_distress_risk": min(raw + 0.15, 1.0),
            "liquidity_risk": min(raw + 0.1, 1.0),
        }
