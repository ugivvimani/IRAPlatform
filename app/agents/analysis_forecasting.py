from __future__ import annotations

import logging
from typing import Any

from app.contracts import EvidenceItem, RiskDimension
from app.policy import recency_score, source_priority

logger = logging.getLogger(__name__)

# ── Dimension weights for composite score ─────────────────────────────────────
# Sanctions carries the highest weight: direct legal exposure, unambiguous.
# ESG and operational are informative but rarely decisive alone.
DIMENSION_WEIGHTS: dict[RiskDimension, float] = {
    RiskDimension.SANCTIONS:     0.40,
    RiskDimension.REGULATORY:    0.20,
    RiskDimension.REPUTATIONAL:  0.15,
    RiskDimension.FINANCIAL:     0.15,
    RiskDimension.ESG:           0.07,
    RiskDimension.OPERATIONAL:   0.03,
}

# Maximum source authority tier value (OFFICIAL = 4)
_MAX_AUTHORITY = 4.0

# ── Signal taxonomy ───────────────────────────────────────────────────────────
# Maps signal name → { value_substring → base_risk }.
# "__default__" is the fallback when no value substring matches.
# Negative base_risk = evidence of safety (softens the dimension score).
_SIGNAL_RISK: dict[str, dict[str, float]] = {
    "sanctions_listed": {
        "yes":  1.0,
        "no":   -0.3,
        "__default__": 0.5,
    },
    "sanctions_status": {
        "not_sanctioned":      -0.3,
        "reported_sanctioned":  0.80,
        "sanctioned_entity":    1.0,
        "under_review":         0.40,
        "possible_match":       0.35,
        "__default__":          0.25,
    },
    "regulatory_enforcement": {
        "no_recent_enforcement": -0.15,
        "enforcement_action":    0.85,
        "under_investigation":   0.70,
        "investigation_closed":  0.20,
        "__default__":           0.20,
    },
    "sec_filings_found": {
        # Many filings = active company (not inherently risky).
        # Zero filings for a supposedly large company = unusual = mild flag.
        "0":         0.10,
        "__default__": 0.0,
    },
    "news_sentiment": {
        "negative":  0.55,
        "positive":  -0.05,
        "neutral":    0.05,
        "__default__": 0.10,
    },
    "reputation_signal": {
        "negative_press":    0.50,
        "positive_coverage": -0.05,
        "__default__":        0.10,
    },
    "esg_rating": {
        # Numeric ESG score from providers: low score = high risk.
        # Handled separately in _signal_base_risk().
        "__numeric_esg__": True,
        "__default__":      0.10,
    },
    "esg_incident": {
        "none_material":     -0.05,
        "material_incident":  0.65,
        "critical_incident":  1.0,
        "__default__":        0.15,
    },
    "retrieval_health": {
        "insufficient_live_data": 0.25,
        "__default__":             0.10,
    },
    "entity_resolution_ambiguous": {
        "requires_review": 0.20,
        "__default__":      0.15,
    },
    # ── SEC EDGAR Financial signals ───────────────────────────────────────────
    "debt_to_equity_ratio": {
        # Numeric D/E ratio: >3 = highly leveraged, <0 = negative equity (critical)
        "__numeric_dte__": True,
        "__default__":     0.10,
    },
    "profit_margin": {
        # Numeric margin: negative = losing money
        "__numeric_margin__": True,
        "__default__":        0.05,
    },
    "net_income_trend": {
        "positive": -0.05,
        "negative":  0.45,
        "__default__": 0.10,
    },
    "late_filing_notice": {
        "yes": 0.70,   # NT 10-K/10-Q = significant financial distress signal
        "__default__": 0.10,
    },
    "sec_registered": {
        # Not being SEC-registered is neutral for private companies, slight flag for
        # companies that should be registered (context-dependent).
        "not_sec_registered": 0.05,
        "__default__":         0.0,
    },
    "entity_not_found": {
        # No data across all sources: neutral score — absence of evidence ≠ safe.
        # The OutputComposer handles this with LOW confidence + WATCH + manual review.
        "no_data_across_all_sources": 0.0,
        "__default__":                0.0,
    },
}


def _signal_base_risk(signal: str, value: str) -> float:
    """
    Look up the base risk magnitude for a (signal, value) pair.

    Returns a float in [-0.5, 1.0]:
    - Positive = risk indicator
    - Negative = evidence of safety (reduces dimension risk)
    """
    taxonomy = _SIGNAL_RISK.get(signal)

    if taxonomy is None:
        # Unknown signal — apply generic value sentiment
        return _generic_value_risk(value)

    # Special case: numeric ESG score (low = risky)
    if taxonomy.get("__numeric_esg__"):
        try:
            score = float(value)
            # ESG scores are 0-100; below 30 = high risk, above 70 = low risk
            return max(0.0, (50.0 - score) / 100.0)
        except ValueError:
            pass

    # Special case: numeric debt-to-equity ratio
    # <0 (negative equity) = 1.0, >3 = high, 1-3 = moderate, <1 = low
    if taxonomy.get("__numeric_dte__"):
        try:
            dte = float(value)
            if dte < 0:
                return 1.0   # Negative equity — critical
            if dte > 5.0:
                return 0.85
            if dte > 3.0:
                return 0.65
            if dte > 1.5:
                return 0.30
            return -0.05     # Low leverage is a mild positive signal
        except ValueError:
            pass

    # Special case: numeric profit margin (negative = losing money)
    if taxonomy.get("__numeric_margin__"):
        try:
            margin = float(value)
            if margin < -0.20:
                return 0.75   # Severe losses
            if margin < 0:
                return 0.45   # Unprofitable
            if margin > 0.10:
                return -0.05  # Healthy margin (mild positive)
            return 0.05       # Thin but positive
        except ValueError:
            pass

    # Substring match against value
    value_norm = value.lower().strip()
    for key, risk in taxonomy.items():
        if key.startswith("__"):
            continue
        if key in value_norm:
            return float(risk)

    return float(taxonomy.get("__default__", 0.1))


def _generic_value_risk(value: str) -> float:
    """Fallback risk estimate for signals not in the taxonomy."""
    v = value.lower().strip()
    # Explicit safe prefix patterns
    if v.startswith(("no_", "not_", "none_", "clean_", "clear_")):
        return -0.1
    if v.endswith("_none"):
        return -0.1
    # Explicit risk keyword patterns
    risk_tokens = ("sanction", "violation", "fraud", "negative", "enforcement",
                   "listed", "flagged", "breach", "default", "bankruptcy",
                   "lawsuit", "investigation", "material")
    if any(t in v for t in risk_tokens):
        return 0.45
    return 0.05  # unknown neutral


def _evidence_weight(item: EvidenceItem) -> float:
    """
    Quality multiplier for a single evidence item.
    Combines source authority, recency, entity match confidence, source reliability.
    Returns a value in (0, 1].
    """
    authority = source_priority(item.source_tier) / _MAX_AUTHORITY        # [0.25, 1.0]
    recency   = recency_score(item.timestamp)                              # [0, 1]
    quality   = item.entity_match_confidence * item.source_confidence      # [0, 1]
    # Recency dampens slightly less than quality to avoid discarding recent weak signals
    return authority * (0.4 * recency + 0.6 * quality)


def _dimension_score(items: list[EvidenceItem]) -> float:
    """
    Aggregate evidence items within a single risk dimension into a [0, 1] score.

    Strategy: worst-case awareness (70% max) + corroboration (30% mean).
    Negative evidence (safe signals) softens the score proportionally.
    """
    if not items:
        return 0.0

    weighted_risks: list[float] = []
    for item in items:
        base   = _signal_base_risk(item.signal, item.value)
        weight = _evidence_weight(item)
        weighted_risks.append(base * weight)

    # Separate positive risk signals from negative (safe) ones
    risk_signals = [r for r in weighted_risks if r > 0]
    safe_signals = [abs(r) for r in weighted_risks if r < 0]

    if not risk_signals:
        return 0.0

    raw_max  = max(risk_signals)
    raw_mean = sum(risk_signals) / len(risk_signals)
    # Blend: worst-case + corroboration
    raw = 0.7 * raw_max + 0.3 * raw_mean

    # Safe signals from high-authority sources meaningfully offset risk
    if safe_signals:
        safe_offset = sum(safe_signals) / len(safe_signals) * 0.35
        raw = max(0.0, raw - safe_offset)

    return min(raw, 1.0)


class AnalysisForecastingAgent:
    """
    Quantitative risk scoring agent.

    Produces per-dimension and composite risk scores by combining:
    - Signal taxonomy (known signals → base risk magnitude)
    - Evidence quality weighting (source authority × recency × confidence)
    - Dimension-weighted aggregation
    - Safe-signal offsetting (negative evidence lowers risk)

    Accepts an optional ``llm_client`` for LLM-enhanced narrative interpretation;
    without one the agent runs fully deterministic rule-based scoring.
    """

    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client

    def score(self, evidence: list[EvidenceItem]) -> dict[str, Any]:
        """
        Score evidence and return a dict with:
            composite_quant_score   – primary [0, 1] risk score
            <dimension>_risk        – per-dimension scores
            data_coverage           – fraction of dimensions with real evidence [0, 1]
            uncertainty             – how much to discount the composite (high = less evidence)
            llm_interpretation      – optional narrative (present only when LLM is available)
        """
        # Group by dimension
        by_dim: dict[RiskDimension, list[EvidenceItem]] = {d: [] for d in RiskDimension}
        for item in evidence:
            by_dim[item.dimension].append(item)

        # Per-dimension scores
        dim_scores: dict[RiskDimension, float] = {}
        for dim, items in by_dim.items():
            dim_scores[dim] = _dimension_score(items)

        # Weighted composite
        composite = sum(
            dim_scores[dim] * weight
            for dim, weight in DIMENSION_WEIGHTS.items()
        )
        composite = min(max(composite, 0.0), 1.0)

        # Data coverage — dimensions that had at least one non-system evidence item
        non_system_dims = {
            item.dimension
            for item in evidence
            if item.source_name.lower() != "system"
        }
        coverage = len(non_system_dims) / len(RiskDimension) if evidence else 0.0

        # Uncertainty: low coverage + operational-only evidence = high uncertainty
        base_uncertainty = 1.0 - coverage
        has_only_operational = all(
            item.dimension == RiskDimension.OPERATIONAL for item in evidence
        ) if evidence else True
        if has_only_operational:
            base_uncertainty = min(base_uncertainty + 0.4, 1.0)
        uncertainty = round(base_uncertainty, 4)

        result: dict[str, Any] = {
            "composite_quant_score": round(composite, 4),
            "sanctions_risk":     round(dim_scores[RiskDimension.SANCTIONS], 4),
            "regulatory_risk":    round(dim_scores[RiskDimension.REGULATORY], 4),
            "reputational_risk":  round(dim_scores[RiskDimension.REPUTATIONAL], 4),
            "financial_risk":     round(dim_scores[RiskDimension.FINANCIAL], 4),
            "esg_risk":           round(dim_scores[RiskDimension.ESG], 4),
            "operational_risk":   round(dim_scores[RiskDimension.OPERATIONAL], 4),
            "data_coverage":      round(coverage, 4),
            "uncertainty":        uncertainty,
        }

        # Optional LLM narrative
        llm_text = self._llm_interpretation(evidence, result)
        if llm_text:
            result["llm_interpretation"] = llm_text

        logger.debug(
            "analysis_scoring composite=%.4f coverage=%.2f uncertainty=%.2f",
            composite, coverage, uncertainty,
        )
        return result

    def _llm_interpretation(
        self, evidence: list[EvidenceItem], scores: dict[str, Any]
    ) -> str | None:
        if not self.llm_client:
            return None
        try:
            top_evidence = sorted(
                (e for e in evidence if e.source_name.lower() != "system"),
                key=lambda e: e.source_confidence * e.entity_match_confidence,
                reverse=True,
            )[:6]
            ev_lines = "\n".join(
                f"- [{e.source_name}/{e.dimension.value}] {e.signal}={e.value}"
                for e in top_evidence
            )
            score_summary = (
                f"composite={scores['composite_quant_score']:.2f}, "
                f"sanctions={scores['sanctions_risk']:.2f}, "
                f"regulatory={scores['regulatory_risk']:.2f}, "
                f"coverage={scores['data_coverage']:.0%}"
            )
            user_prompt = (
                f"Risk scores: {score_summary}\n"
                f"Top evidence:\n{ev_lines}\n\n"
                "In 1-2 sentences, explain the key drivers behind these quantitative risk scores "
                "for a compliance analyst. Be factual, cite specific signals."
            )
            return self.llm_client.complete(
                system_prompt=(
                    "You are a quantitative risk analyst interpreting evidence-based risk scores "
                    "for an integrity risk assessment system."
                ),
                user_prompt=user_prompt,
                max_tokens=120,
            )
        except Exception as exc:
            logger.warning("LLM interpretation failed: %s", exc)
            return None
