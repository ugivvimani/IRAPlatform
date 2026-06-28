from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskDimension(str, Enum):
    FINANCIAL = "financial"
    SANCTIONS = "sanctions"
    REGULATORY = "regulatory"
    ESG = "esg"
    REPUTATIONAL = "reputational"
    OPERATIONAL = "operational"


class SourceTier(str, Enum):
    OFFICIAL = "official"
    REGULATOR = "regulator"
    TIER1_NEWS = "tier1_news"
    SECONDARY = "secondary"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskRating(str, Enum):
    SAFE = "safe"
    WATCH = "watch"
    HIGH_RISK = "high_risk"
    RESTRICTED = "restricted"


class UserQuery(BaseModel):
    company_name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    requested_dimensions: list[RiskDimension] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str
    dimension: RiskDimension
    signal: str
    value: str
    source_name: str
    source_tier: SourceTier
    timestamp: datetime
    entity_match_confidence: float = Field(ge=0.0, le=1.0)
    source_confidence: float = Field(ge=0.0, le=1.0)
    provenance_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Raw human-readable content from the source (article text, filing excerpt, etc.).
    # Used by MemoryManagerAgent to generate a concise summary for vector storage.
    raw_content: str | None = None


class CriticScoreVector(BaseModel):
    authority: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    entity_certainty: float = Field(ge=0.0, le=1.0)
    corroboration: float = Field(ge=0.0, le=1.0)
    temporal_coherence: float = Field(ge=0.0, le=1.0)
    contradiction_penalty: float = Field(ge=0.0, le=1.0)
    evidence_sufficiency_penalty: float = Field(ge=0.0, le=1.0)


class HypothesisBranch(BaseModel):
    branch_id: str
    interpretation: str
    proposed_actions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    score: CriticScoreVector
    composite_score: float = Field(ge=0.0, le=1.0)
    confidence: ConfidenceLevel
    # Stability metadata — used by conflict resolution stability gate and audit trail
    source_count: int = Field(default=1, ge=1)          # number of distinct sources supporting this branch
    score_spread: float = Field(default=0.0, ge=0.0)    # pessimistic-to-optimistic score band; high = unstable


class ConflictResolutionResult(BaseModel):
    conflict_detected: bool
    winner: HypothesisBranch | None = None
    alternatives: list[HypothesisBranch] = Field(default_factory=list)
    requires_manual_review: bool = False
    rationale: str
    # evidence_ids of items whose values were rejected during conflict resolution;
    # scoring should exclude these to avoid diluting the resolved interpretation.
    suppressed_evidence_ids: list[str] = Field(default_factory=list)


class MemoryFact(BaseModel):
    fact_id: str
    entity_id: str
    summary: str = Field(min_length=1, max_length=2000)
    dimension: RiskDimension
    severity: float = Field(ge=0.0, le=1.0)
    source_reference: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationRecord(BaseModel):
    calibration_id: str
    entity_id: str
    source_name: str
    signal_type: str
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    total_outcomes: int = 0
    effective_sample_size: float = Field(default=0.0, ge=0.0)
    uncertainty_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reliability_score: float = Field(ge=0.0, le=1.0)
    updated_at: datetime


class AssessmentDecision(BaseModel):
    risk_rating: RiskRating
    confidence: ConfidenceLevel
    summary: str
    recommended_next_steps: list[str] = Field(default_factory=list)
    requires_manual_review: bool = False


class EscalationContext(BaseModel):
    escalation_required: bool = False
    auto_hold: bool = False
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)
    review_message: str = ""


class AssessmentResponse(BaseModel):
    query: UserQuery
    decision: AssessmentDecision
    evidence_chain: list[EvidenceItem] = Field(default_factory=list)
    conflict_result: ConflictResolutionResult | None = None
    escalation: EscalationContext | None = None
    model_metadata: dict[str, Any] = Field(default_factory=dict)


class CompactAssessmentResponse(BaseModel):
    assessment_id: int
    company_name: str
    risk_rating: RiskRating
    confidence: ConfidenceLevel
    summary: str
    recommended_next_steps: list[str] = Field(default_factory=list)
    requires_manual_review: bool = False
    evaluated_at: datetime


class AssessRequest(BaseModel):
    query: UserQuery
    evidence: list[EvidenceItem] = Field(default_factory=list)


class WatchlistEntry(BaseModel):
    entity_id: str
    company_name: str
    added_at: datetime = Field(default_factory=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    notes: str = ""


class WatchlistStatus(BaseModel):
    entity_id: str
    company_name: str
    notes: str
    current_risk_rating: RiskRating | None = None
    last_assessed_at: datetime | None = None


class AssessmentAuditRecord(BaseModel):
    assessment_id: int
    entity_id: str
    company_name: str
    question: str
    risk_rating: RiskRating
    confidence: ConfidenceLevel
    requires_manual_review: bool
    created_at: datetime


class PolicyThresholdRecord(BaseModel):
    policy_key: str
    threshold_value: float
    version: int
    approved_by: str
    approval_notes: str = ""
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class PolicyThresholdUpsert(BaseModel):
    threshold_value: float
    approved_by: str = Field(min_length=1)
    approval_notes: str = ""
