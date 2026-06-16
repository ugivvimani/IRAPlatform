from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.contracts import (
    AssessmentAuditRecord,
    AssessRequest,
    AssessmentResponse,
    UserQuery,
    WatchlistEntry,
    WatchlistStatus,
)
from app.llm.factory import build_llm_client
from app.orchestrator import OrchestratorAgent
from app.settings import load_settings
from app.storage.sqlite_repo import SQLiteRepository
from app.vector_store.factory import build_vector_store

app = FastAPI(title="Integrity Risk Assessment Agent", version="0.2.0")
settings = load_settings()

vector_store = build_vector_store()
memory_agent = MemoryManagerAgent(vector_store)
llm_client = build_llm_client()
sqlite_repo = SQLiteRepository(settings.sqlite_db_path)
orchestrator = OrchestratorAgent(
    retrieval=RetrievalAgent(),
    analysis=AnalysisForecastingAgent(),
    conflict=ConflictResolutionAgent(),
    memory=memory_agent,
    composer=OutputComposerAgent(),
    calibration=CalibrationAgent(),
)

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "vector_backend": settings.vector_backend,
        "llm_backend": type(llm_client).__name__,
        "sqlite_db_path": settings.sqlite_db_path,
    }


def _assess_and_store(request: AssessRequest) -> AssessmentResponse:
    result = orchestrator.assess(request)
    sqlite_repo.insert_assessment(result)
    return result


@app.post("/assess", response_model=AssessmentResponse)
def assess(request: AssessRequest) -> AssessmentResponse:
    return _assess_and_store(request)


@app.post("/watchlist", response_model=WatchlistEntry, status_code=201)
def add_to_watchlist(entry: WatchlistEntry) -> WatchlistEntry:
    return sqlite_repo.upsert_watchlist(entry)


@app.get("/watchlist/{entity_id}", response_model=WatchlistStatus)
def get_watchlist_status(entity_id: str) -> WatchlistStatus:
    entry = sqlite_repo.get_watchlist(entity_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not on watchlist.")
    result = _assess_and_store(
        AssessRequest(query=UserQuery(company_name=entry.company_name, question="Watchlist status check."))
    )
    return WatchlistStatus(
        entity_id=entry.entity_id,
        company_name=entry.company_name,
        notes=entry.notes,
        current_risk_rating=result.decision.risk_rating,
        last_assessed_at=datetime.now(timezone.utc),
    )


@app.get("/watchlist", response_model=list[WatchlistEntry])
def list_watchlist() -> list[WatchlistEntry]:
    return sqlite_repo.list_watchlist()


@app.get("/assessments/{entity_id}", response_model=list[AssessmentAuditRecord])
def list_assessments(entity_id: str, limit: int = 25) -> list[AssessmentAuditRecord]:
    return sqlite_repo.list_assessments(entity_id=entity_id, limit=limit)
