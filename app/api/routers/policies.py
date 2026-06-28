from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_storage_repo
from app.contracts import PolicyThresholdRecord, PolicyThresholdUpsert
from app.core.security import User, get_authenticated_user, require_admin

router = APIRouter(prefix="/policies", tags=["policies"])


@router.get("/active", response_model=dict[str, PolicyThresholdRecord])
async def get_active_policies(
    user: User = Depends(get_authenticated_user),
    storage_repo=Depends(get_storage_repo),
) -> dict[str, PolicyThresholdRecord]:
    del user
    return storage_repo.get_active_policy_thresholds()


@router.put("/{policy_key}", response_model=PolicyThresholdRecord)
async def upsert_policy_threshold(
    policy_key: str,
    payload: PolicyThresholdUpsert,
    user: User = Depends(require_admin),
    storage_repo=Depends(get_storage_repo),
) -> PolicyThresholdRecord:
    del user
    return storage_repo.upsert_policy_threshold(policy_key=policy_key, payload=payload)

