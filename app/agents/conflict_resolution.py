from __future__ import annotations

from collections import defaultdict

from app.contracts import (
    ConfidenceLevel,
    ConflictResolutionResult,
    CriticScoreVector,
    EvidenceItem,
    HypothesisBranch,
    MemoryFact,
)
from app.policy import composite_score, recency_score, source_priority


class ConflictResolutionAgent:
    def resolve(
        self,
        evidence: list[EvidenceItem],
        historical_facts: list[MemoryFact],
        source_reliability: dict[str, float] | None = None,
    ) -> ConflictResolutionResult:
        source_reliability = source_reliability or {}
        grouped: dict[tuple[str, str], list[EvidenceItem]] = defaultdict(list)
        for item in evidence:
            grouped[(item.dimension.value, item.signal)].append(item)

        conflicting_groups = [group for group in grouped.values() if len({x.value for x in group}) > 1]
        if not conflicting_groups:
            return ConflictResolutionResult(
                conflict_detected=False,
                rationale="No contradictory signals were detected.",
            )

        branches: list[HypothesisBranch] = []
        cold_start = len(historical_facts) == 0
        for idx, group in enumerate(conflicting_groups, start=1):
            best = sorted(
                group,
                key=lambda x: (
                    source_priority(x.source_tier) * source_reliability.get(x.source_name, 0.5),
                    recency_score(x.timestamp),
                    x.entity_match_confidence,
                    x.source_confidence,
                ),
                reverse=True,
            )[0]
            corroboration = min(1.0, len({x.source_name for x in group}) / 3.0)
            has_historical_pattern = any(best.dimension == fact.dimension for fact in historical_facts)
            source_reliability_score = source_reliability.get(best.source_name, 0.5)
            authority_score = min((source_priority(best.source_tier) / 4.0) * source_reliability_score, 1.0)
            low_entity_certainty = best.entity_match_confidence < 0.8
            evidence_penalty = 0.45 if len(group) < 2 else 0.1
            if cold_start:
                evidence_penalty = max(evidence_penalty, 0.5)
            score_vector = CriticScoreVector(
                authority=authority_score,
                recency=recency_score(best.timestamp),
                entity_certainty=best.entity_match_confidence,
                corroboration=corroboration,
                temporal_coherence=1.0 if has_historical_pattern else (0.55 if cold_start else 0.65),
                contradiction_penalty=0.2 if len(group) > 1 else 0.0,
                evidence_sufficiency_penalty=evidence_penalty,
            )
            score = composite_score(score_vector)
            branches.append(
                HypothesisBranch(
                    branch_id=f"branch-{idx}-{best.evidence_id}",
                    interpretation=f"Prioritize {best.source_name} signal: {best.value}",
                    proposed_actions=[
                        "Expand cross-jurisdiction retrieval",
                        "Verify regulator press releases",
                    ],
                    assumptions=["Entity match confidence remains above threshold.", "Source reliability weighting applied."],
                    score=score_vector,
                    composite_score=score,
                    confidence=(
                        ConfidenceLevel.HIGH
                        if score >= 0.78 and not low_entity_certainty and not cold_start
                        else (ConfidenceLevel.MEDIUM if score >= 0.62 else ConfidenceLevel.LOW)
                    ),
                )
            )

        branches.sort(key=lambda b: b.composite_score, reverse=True)
        winner = branches[0]
        alternatives = branches[1:3]
        requires_manual_review = winner.composite_score < 0.65
        confidence_margin = winner.composite_score - (alternatives[0].composite_score if alternatives else 0.0)
        if confidence_margin < 0.12:
            requires_manual_review = True
        if cold_start and winner.composite_score < 0.8:
            requires_manual_review = True

        return ConflictResolutionResult(
            conflict_detected=True,
            winner=winner,
            alternatives=alternatives,
            requires_manual_review=requires_manual_review,
            rationale=(
                "Winner selected by authority-recency-entity weighted scoring with corroboration and "
                "sparsity penalties; alternatives preserved for audit and cold-start safety. "
                "Low-margin outcomes are forced to manual review."
            ),
        )
