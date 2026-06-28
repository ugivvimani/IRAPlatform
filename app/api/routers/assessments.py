from __future__ import annotations

import logging
import time
from typing import Union

from fastapi import APIRouter, Body, Depends, Query

from app.api.deps import get_orchestrator, get_storage_repo
from app.api.schemas import AsyncAssessRequest, AsyncTaskResponse
from app.contracts import (
    AssessRequest,
    AssessmentAuditRecord,
    AssessmentResponse,
    CompactAssessmentResponse,
)
from app.core.security import User, get_authenticated_user, require_write_access
from app.core.validation import validate_assess_request
from app.services.webjobs import enqueue_assessment_job, get_job_status

logger = logging.getLogger(__name__)
router = APIRouter(tags=["assessments"])


@router.post("/assess", response_model=Union[CompactAssessmentResponse, AssessmentResponse])
async def assess(
    request: AssessRequest = Body(...),
    include_details: bool = Query(default=False),
    user: User = Depends(require_write_access),
    orchestrator=Depends(get_orchestrator),
    storage_repo=Depends(get_storage_repo),
) -> Union[CompactAssessmentResponse, AssessmentResponse]:
    started = time.perf_counter()
    validated_request = validate_assess_request(request)
    logger.info(
        "assessment_request_received user=%s entity=%s",
        user.user_id,
        validated_request.query.company_name,
    )
    result = orchestrator.assess(validated_request)
    assessment_id = storage_repo.insert_assessment(result)
    logger.info(
        "assessment_request_completed user=%s entity=%s total_ms=%d",
        user.user_id,
        validated_request.query.company_name,
        int((time.perf_counter() - started) * 1000),
    )
    if include_details:
        return result
    evaluated_at = result.model_metadata.get("evaluated_at")
    if not isinstance(evaluated_at, str):
        evaluated_at = "1970-01-01T00:00:00+00:00"
    return CompactAssessmentResponse(
        assessment_id=assessment_id,
        company_name=result.query.company_name,
        risk_rating=result.decision.risk_rating,
        confidence=result.decision.confidence,
        summary=result.decision.summary,
        recommended_next_steps=result.decision.recommended_next_steps,
        requires_manual_review=result.decision.requires_manual_review,
        evaluated_at=evaluated_at,
    )


@router.post("/assess/async", response_model=AsyncTaskResponse, status_code=202)
async def assess_async(request: AsyncAssessRequest = Body(...), user: User = Depends(require_write_access)) -> AsyncTaskResponse:
    del user
    task_id = enqueue_assessment_job(request.company_name, request.question)
    return AsyncTaskResponse(task_id=task_id, status="queued")


@router.get("/tasks/{task_id}")
async def task_status(task_id: str, user: User = Depends(get_authenticated_user)) -> dict:
    del user
    return get_job_status(task_id)


@router.get("/assessments/{entity_id}", response_model=list[AssessmentAuditRecord])
async def list_assessments(
    entity_id: str,
    limit: int = 25,
    user: User = Depends(get_authenticated_user),
    storage_repo=Depends(get_storage_repo),
) -> list[AssessmentAuditRecord]:
    del user
    return storage_repo.list_assessments(entity_id=entity_id, limit=limit)

