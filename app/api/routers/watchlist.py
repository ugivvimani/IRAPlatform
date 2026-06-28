from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_orchestrator, get_storage_repo
from app.contracts import AssessRequest, UserQuery, WatchlistEntry, WatchlistStatus
from app.core.security import User, get_authenticated_user, require_write_access

router = APIRouter(tags=["watchlist"])


@router.post("/watchlist", response_model=WatchlistEntry, status_code=201)
async def add_to_watchlist(entry: WatchlistEntry, user: User = Depends(require_write_access), storage_repo=Depends(get_storage_repo)) -> WatchlistEntry:
    del user
    return storage_repo.upsert_watchlist(entry)


@router.get("/watchlist/{entity_id}", response_model=WatchlistStatus)
async def get_watchlist_status(
    entity_id: str,
    refresh: bool = Query(default=False, description="Set true to trigger a new live assessment"),
    user: User = Depends(get_authenticated_user),
    orchestrator=Depends(get_orchestrator),
    storage_repo=Depends(get_storage_repo),
) -> WatchlistStatus:
    del user
    entry = storage_repo.get_watchlist(entity_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not on watchlist.")

    # Default: return last stored assessment (fast, no connector calls)
    # ?refresh=true triggers a new full live assessment and stores it
    last_assessments = storage_repo.list_assessments(entity_id=entity_id, limit=1)

    if refresh or not last_assessments:
        result = orchestrator.assess(
            AssessRequest(query=UserQuery(company_name=entry.company_name, question="Watchlist status check."))
        )
        storage_repo.insert_assessment(result)
        return WatchlistStatus(
            entity_id=entry.entity_id,
            company_name=entry.company_name,
            notes=entry.notes,
            current_risk_rating=result.decision.risk_rating,
            last_assessed_at=datetime.now(timezone.utc),
        )

    last = last_assessments[0]
    return WatchlistStatus(
        entity_id=entry.entity_id,
        company_name=entry.company_name,
        notes=entry.notes,
        current_risk_rating=last.risk_rating,
        last_assessed_at=last.created_at,
    )


@router.get("/watchlist", response_model=list[WatchlistEntry])
async def list_watchlist(user: User = Depends(get_authenticated_user), storage_repo=Depends(get_storage_repo)) -> list[WatchlistEntry]:
    del user
    return storage_repo.list_watchlist()


@router.delete("/watchlist/{entity_id}", status_code=200)
async def remove_from_watchlist(entity_id: str, user: User = Depends(require_write_access), storage_repo=Depends(get_storage_repo)) -> None:
    del user
    storage_repo.delete_watchlist(entity_id)
