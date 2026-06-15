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
    "authority": 0.28,
    "recency": 0.14,
    "entity_certainty": 0.2,
    "corroboration": 0.2,
    "temporal_coherence": 0.1,
    "contradiction_penalty": 0.04,
    "evidence_sufficiency_penalty": 0.04,
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
