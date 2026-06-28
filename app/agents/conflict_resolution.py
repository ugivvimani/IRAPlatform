from __future__ import annotations

import logging
from collections import defaultdict

from app.contracts import (
    ConfidenceLevel,
    ConflictResolutionResult,
    CriticScoreVector,
    EvidenceItem,
    HypothesisBranch,
    MemoryFact,
)
from app.policy import composite_score, recency_score, score_bounds, source_priority, sparsity_dampener

logger = logging.getLogger(__name__)


class ConflictResolutionAgent:
    TOT_BEAM_WIDTH = 3
    TOT_DEPTH_LIMIT = 5
    TOT_STABLE_THRESHOLD = 0.85

    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client

    def _llm_rationale(self, winner: HypothesisBranch, alternatives: list[HypothesisBranch], entity_hints: str) -> str:
        if not self.llm_client:
            return (
                f"Winner selected by bounded ToT ranking (beam width {self.TOT_BEAM_WIDTH}, depth limit {self.TOT_DEPTH_LIMIT}) "
                "using authority-recency-entity weighted scoring with corroboration and sparsity penalties; "
                "alternatives preserved for audit and cold-start safety. Low-margin or unstable outcomes are forced to manual review."
            )
        try:
            alt_text = "; ".join(f"{b.interpretation} (score={b.composite_score:.2f})" for b in alternatives) or "none"
            user_prompt = (
                f"Entity context: {entity_hints}\n"
                f"Winning branch: {winner.interpretation} (score={winner.composite_score:.2f}, confidence={winner.confidence.value})\n"
                f"Alternative branches: {alt_text}\n\n"
                "In 2-3 concise sentences, explain why the winning branch was selected over the alternatives, "
                "referencing source authority, recency, and corroboration where relevant."
            )
            return self.llm_client.complete(
                system_prompt="You are a compliance analyst summarizing a conflict-resolution decision for an integrity risk assessment.",
                user_prompt=user_prompt,
                max_tokens=200,
            )
        except Exception as exc:
            logger.warning("LLM rationale generation failed: %s", exc)
            return (
                f"Winner selected by bounded ToT ranking (beam width {self.TOT_BEAM_WIDTH}, depth limit {self.TOT_DEPTH_LIMIT}) "
                "using authority-recency-entity weighted scoring."
            )

    def resolve(
        self,
        evidence: list[EvidenceItem],
        historical_facts: list[MemoryFact],
        source_reliability: dict[str, float] | None = None,
        conflict_history: list[str] | None = None,
    ) -> ConflictResolutionResult:
        source_reliability = source_reliability or {}
        conflict_history = conflict_history or []
        # A recurring contradiction pattern (same entity conflicted before) boosts
        # temporal_coherence for the top-ranked candidate since it's a known issue.
        has_prior_conflict_pattern = len(conflict_history) > 0
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
        suppressed_ids: list[str] = []
        cold_start = len(historical_facts) == 0
        for idx, group in enumerate(conflicting_groups, start=1):
            ranked_candidates = sorted(
                group,
                key=lambda x: (
                    source_priority(x.source_tier) * source_reliability.get(x.source_name, 0.5),
                    recency_score(x.timestamp),
                    x.entity_match_confidence,
                    x.source_confidence,
                ),
                reverse=True,
            )
            # The top-ranked candidate wins this group; all others are suppressed.
            # Suppressed IDs are communicated back to the orchestrator so scoring
            # operates only on the resolved (winner) evidence set.
            winning_value = ranked_candidates[0].value
            for loser in ranked_candidates[1:]:
                if loser.value != winning_value:
                    suppressed_ids.append(loser.evidence_id)
            for depth, candidate in enumerate(ranked_candidates[: self.TOT_DEPTH_LIMIT], start=1):
                supporting_sources = {x.source_name for x in group if x.value == candidate.value}
                corroboration = min(1.0, len(supporting_sources) / 3.0)
                has_historical_pattern = any(candidate.dimension == fact.dimension for fact in historical_facts)
                source_reliability_score = source_reliability.get(candidate.source_name, 0.5)
                authority_score = min((source_priority(candidate.source_tier) / 4.0) * source_reliability_score, 1.0)
                low_entity_certainty = candidate.entity_match_confidence < 0.8
                evidence_penalty = 0.35 if len(supporting_sources) < 2 else 0.1
                if cold_start:
                    evidence_penalty = max(evidence_penalty, 0.5)
                score_vector = CriticScoreVector(
                    authority=authority_score,
                    recency=recency_score(candidate.timestamp),
                    entity_certainty=candidate.entity_match_confidence,
                    corroboration=corroboration,
                    # Known prior conflict pattern → temporal coherence gets a boost
                    # because this is a recognised recurring pattern, not novel noise.
                    temporal_coherence=(
                        1.0 if has_historical_pattern
                        else (0.72 if has_prior_conflict_pattern
                              else (0.55 if cold_start else 0.65))
                    ),
                    contradiction_penalty=0.2 if len(group) > 1 else 0.0,
                    evidence_sufficiency_penalty=evidence_penalty,
                )
                raw_score = composite_score(score_vector)

                # Apply sparsity dampener: single-source branches cannot silently win
                source_count = len(supporting_sources)
                damped_score = raw_score * sparsity_dampener(source_count, cold_start)

                # Score spread: estimates how much the score could shift under uncertainty
                pess, opt = score_bounds(score_vector)
                spread = round(
                    (opt - pess) * sparsity_dampener(source_count, cold_start), 4
                )

                branches.append(
                    HypothesisBranch(
                        branch_id=f"branch-{idx}-d{depth}-{candidate.evidence_id}",
                        interpretation=f"Prioritize {candidate.source_name} signal: {candidate.value}",
                        proposed_actions=[
                            "Expand cross-jurisdiction retrieval",
                            "Verify regulator press releases",
                        ],
                        assumptions=[
                            "Entity match confidence remains above threshold.",
                            "Source reliability weighting applied.",
                            f"Bounded ToT depth {depth}/{self.TOT_DEPTH_LIMIT} considered.",
                        ],
                        score=score_vector,
                        composite_score=round(min(damped_score, 1.0), 4),
                        source_count=source_count,
                        score_spread=spread,
                        confidence=(
                            ConfidenceLevel.HIGH
                            if damped_score >= self.TOT_STABLE_THRESHOLD and not low_entity_certainty and not cold_start
                            else (ConfidenceLevel.MEDIUM if damped_score >= 0.62 else ConfidenceLevel.LOW)
                        ),
                    )
                )

        branches.sort(key=lambda b: b.composite_score, reverse=True)
        branches = branches[: self.TOT_BEAM_WIDTH]
        winner = branches[0]
        alternatives = branches[1:3]

        # --- Stability gates ---
        # 1. Score below absolute floor
        requires_manual_review = winner.composite_score < 0.65
        # 2. Too-narrow margin between winner and first alternative
        confidence_margin = winner.composite_score - (alternatives[0].composite_score if alternatives else 0.0)
        if confidence_margin < 0.12:
            requires_manual_review = True
        # 3. Cold-start: even a good score is not trustworthy without history
        if cold_start and winner.composite_score < 0.8:
            requires_manual_review = True
        # 4. Below stable threshold with a live alternative
        if winner.composite_score < self.TOT_STABLE_THRESHOLD and len(branches) > 1:
            requires_manual_review = True
        # 5. High score spread (uncertainty too wide to trust the central estimate)
        if winner.score_spread > 0.20:
            requires_manual_review = True

        entity_hints = ", ".join({e.source_name for e in evidence})
        rationale = self._llm_rationale(winner, alternatives, entity_hints)

        return ConflictResolutionResult(
            conflict_detected=True,
            winner=winner,
            alternatives=alternatives,
            requires_manual_review=requires_manual_review,
            rationale=rationale,
            suppressed_evidence_ids=suppressed_ids,
        )
