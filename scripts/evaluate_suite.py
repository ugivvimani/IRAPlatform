from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.contracts import AssessRequest, EvidenceItem, RiskDimension, UserQuery
from app.main import memory_agent, orchestrator


def _evidence(
    evidence_id: str,
    dimension: RiskDimension,
    signal: str,
    value: str,
    source_name: str,
    source_tier: str,
    confidence: float = 0.9,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        dimension=dimension,
        signal=signal,
        value=value,
        source_name=source_name,
        source_tier=source_tier,
        timestamp=datetime.now(timezone.utc),
        entity_match_confidence=0.9,
        source_confidence=confidence,
        provenance_url=f"https://example.org/{source_name.lower()}",
        metadata={},
    )


def run() -> int:
    failures: list[str] = []

    # 1) Cold-start conflict should force manual review.
    cold_request = AssessRequest(
        query=UserQuery(company_name="ColdStartCo", question="Is this company safe?"),
        evidence=[
            _evidence("c1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official", 0.95),
            _evidence("c2", RiskDimension.SANCTIONS, "sanctions_status", "reported_sanctioned", "Reuters", "tier1_news", 0.75),
        ],
    )
    cold_result = orchestrator.assess(cold_request)
    if not cold_result.decision.requires_manual_review:
        failures.append("Cold-start conflicting signals did not require manual review.")

    # 2) Clean non-conflicting evidence should not be high-risk.
    safe_request = AssessRequest(
        query=UserQuery(company_name="StableCo", question="Is this company safe?"),
        evidence=[
            _evidence("s1", RiskDimension.SANCTIONS, "sanctions_status", "not_sanctioned", "OFAC", "official", 0.95),
            _evidence("s2", RiskDimension.REGULATORY, "regulatory_enforcement", "no_recent_enforcement", "SEC", "regulator", 0.9),
            _evidence("s3", RiskDimension.ESG, "esg_incident", "none_material", "ESGDB", "secondary", 0.7),
        ],
    )
    safe_result = orchestrator.assess(safe_request)
    if safe_result.decision.risk_rating.value == "high_risk":
        failures.append("Stable non-conflicting scenario incorrectly rated as high risk.")

    # 3) Retrieval-health degradation should lower confidence and force manual review.
    degraded_request = AssessRequest(
        query=UserQuery(company_name="SparseDataCo", question="Is this company safe?"),
        evidence=[
            _evidence("d1", RiskDimension.OPERATIONAL, "retrieval_health", "insufficient_live_data", "system", "secondary", 0.2),
        ],
    )
    degraded_result = orchestrator.assess(degraded_request)
    if degraded_result.decision.confidence.value != "low" or not degraded_result.decision.requires_manual_review:
        failures.append("Insufficient live data did not produce low-confidence manual review output.")

    # 4) Calibration should produce/retain reliability entries.
    reliability = memory_agent.load_source_reliability("ColdStartCo")
    if "system_orchestrator" not in reliability:
        failures.append("Calibration reliability for system_orchestrator was not persisted.")

    if failures:
        print("Evaluation suite failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("Evaluation suite passed.")
    print(
        "Scenarios: cold-start conflict, stable non-conflict, sparse data fallback, calibration persistence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
