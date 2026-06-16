from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import orchestrator
from app.contracts import AssessRequest, UserQuery


def run() -> None:
    scenarios = [
        ("Company X", "Is Company X safe to partner with?"),
        ("Company Y", "Any sanctions or integrity red flags for Company Y?"),
        ("Company Z", "Do we need manual review before onboarding Company Z?"),
    ]
    for company, question in scenarios:
        result = orchestrator.assess(AssessRequest(query=UserQuery(company_name=company, question=question)))
        print(
            f"{company}: rating={result.decision.risk_rating.value}, "
            f"confidence={result.decision.confidence.value}, manual_review={result.decision.requires_manual_review}"
        )


if __name__ == "__main__":
    run()
