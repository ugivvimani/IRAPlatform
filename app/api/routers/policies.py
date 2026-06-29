from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_storage_repo
from app.contracts import PolicyThresholdRecord, PolicyThresholdUpsert
from app.core.security import require_api_key

router = APIRouter(prefix="/policies", tags=["policies"], dependencies=[Depends(require_api_key)])


@router.get("/active", response_model=dict[str, PolicyThresholdRecord])
async def get_active_policies(storage_repo=Depends(get_storage_repo)) -> dict[str, PolicyThresholdRecord]:
    return storage_repo.get_active_policy_thresholds()


@router.put("/{policy_key}", response_model=PolicyThresholdRecord)
async def upsert_policy_threshold(
    policy_key: str,
    payload: PolicyThresholdUpsert,
    storage_repo=Depends(get_storage_repo),
) -> PolicyThresholdRecord:
    return storage_repo.upsert_policy_threshold(policy_key=policy_key, payload=payload)

