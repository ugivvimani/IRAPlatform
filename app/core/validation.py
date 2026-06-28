from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from fastapi import HTTPException

from app.contracts import AssessRequest


def _is_allowed_provenance(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return True
    if parsed.scheme == "internal":
        return True
    return False


def validate_assess_request(request: AssessRequest) -> AssessRequest:
    query = request.query.model_copy(
        update={
            "company_name": request.query.company_name.strip(),
            "question": request.query.question.strip(),
        }
    )

    if not query.company_name:
        raise HTTPException(status_code=422, detail="company_name must not be blank.")
    if not query.question:
        raise HTTPException(status_code=422, detail="question must not be blank.")

    now = datetime.now(timezone.utc)
    future_threshold = now + timedelta(days=1)

    for item in request.evidence:
        if not _is_allowed_provenance(item.provenance_url):
            raise HTTPException(
                status_code=422,
                detail=f"Evidence '{item.evidence_id}' has unverifiable provenance_url.",
            )
        if item.timestamp > future_threshold:
            raise HTTPException(
                status_code=422,
                detail=f"Evidence '{item.evidence_id}' has an invalid future timestamp.",
            )

    return request.model_copy(update={"query": query})
