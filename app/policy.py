from __future__ import annotations

from datetime import datetime, timezone

from app.contracts import CriticScoreVector, SourceTier


SOURCE_PRIORITY: dict[SourceTier, int] = {
    SourceTier.OFFICIAL: 4,
    SourceTier.REGULATOR: 3,
    SourceTier.TIER1_NEWS: 2,
    SourceTier.SECONDARY: 1,
}

SCORING_WEIGHTS: dict[str, float] = {
    "authority":                  0.26,
    "recency":                    0.13,
    "entity_certainty":           0.18,
    "corroboration":              0.18,
    "temporal_coherence":         0.09,
    "contradiction_penalty":      0.06,   # was 0.04 — increased to penalise contested signals more
    "evidence_sufficiency_penalty": 0.10, # was 0.04 — sparse evidence now meaningfully lowers score
}


def source_priority(tier: SourceTier) -> int:
    return SOURCE_PRIORITY[tier]


def recency_score(ts: datetime, half_life_days: int = 30) -> float:
    age_days = max((datetime.now(timezone.utc) - ts).days, 0)
    decay = 0.5 ** (age_days / max(half_life_days, 1))
    return max(min(decay, 1.0), 0.0)


def composite_score(vector: CriticScoreVector) -> float:
    score = (
        vector.authority * SCORING_WEIGHTS["authority"]
        + vector.recency * SCORING_WEIGHTS["recency"]
        + vector.entity_certainty * SCORING_WEIGHTS["entity_certainty"]
        + vector.corroboration * SCORING_WEIGHTS["corroboration"]
        + vector.temporal_coherence * SCORING_WEIGHTS["temporal_coherence"]
        - vector.contradiction_penalty * SCORING_WEIGHTS["contradiction_penalty"]
        - vector.evidence_sufficiency_penalty * SCORING_WEIGHTS["evidence_sufficiency_penalty"]
    )
    return max(min(score, 1.0), 0.0)


def sparsity_dampener(source_count: int, cold_start: bool) -> float:
    """
    Multiplicative dampener applied to `composite_score` when evidence is sparse.

    Prevents a single high-authority source from silently winning beam search
    with a misleadingly high score. The dampener never amplifies a score above 1.0.

    | sources | cold_start | factor |
    |---------|------------|--------|
    |    1    |    True    |  0.68  |
    |    1    |    False   |  0.80  |
    |    2    |    True    |  0.88  |
    |    2    |    False   |  0.94  |
    |   3+    |    any     |  1.00  |
    """
    if source_count >= 3:
        return 1.00
    if source_count == 2:
        return 0.88 if cold_start else 0.94
    # source_count == 1
    return 0.68 if cold_start else 0.80


def score_bounds(vector: CriticScoreVector) -> tuple[float, float]:
    """
    Return (pessimistic, optimistic) score bounds that capture the uncertainty
    in components that are hard to estimate precisely:
    - `entity_certainty` may be off by ±0.15 (entity disambiguation is fuzzy)
    - `corroboration` may be off by ±0.10 (more sources may corroborate later)
    - `temporal_coherence` is estimated; treat as ±0.10

    The spread = optimistic - pessimistic is used as the stability gate:
    when spread > 0.20, the branch outcome is not yet reliable enough to
    suppress manual review even if the central score looks good.
    """
    delta_entity   = 0.15 * SCORING_WEIGHTS["entity_certainty"]
    delta_corr     = 0.10 * SCORING_WEIGHTS["corroboration"]
    delta_coherence = 0.10 * SCORING_WEIGHTS["temporal_coherence"]
    half_spread = delta_entity + delta_corr + delta_coherence  # ≈ 0.045

    central = composite_score(vector)
    pessimistic = max(central - half_spread, 0.0)
    optimistic  = min(central + half_spread, 1.0)
    return pessimistic, optimistic
