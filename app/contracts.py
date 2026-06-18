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


class ConflictResolutionResult(BaseModel):
    conflict_detected: bool
    winner: HypothesisBranch | None = None
    alternatives: list[HypothesisBranch] = Field(default_factory=list)
    requires_manual_review: bool = False
    rationale: str


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


class AssessmentResponse(BaseModel):
    query: UserQuery
    decision: AssessmentDecision
    evidence_chain: list[EvidenceItem] = Field(default_factory=list)
    conflict_result: ConflictResolutionResult | None = None
    model_metadata: dict[str, Any] = Field(default_factory=dict)


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
