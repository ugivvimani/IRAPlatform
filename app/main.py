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
    AssessRequest,
    AssessmentResponse,
    UserQuery,
    WatchlistEntry,
    WatchlistStatus,
)
from app.llm.factory import build_llm_client
from app.orchestrator import OrchestratorAgent
from app.settings import load_settings
from app.vector_store.factory import build_vector_store

app = FastAPI(title="Integrity Risk Assessment Agent", version="0.2.0")
settings = load_settings()

vector_store = build_vector_store()
memory_agent = MemoryManagerAgent(vector_store)
llm_client = build_llm_client()
orchestrator = OrchestratorAgent(
    retrieval=RetrievalAgent(),
    analysis=AnalysisForecastingAgent(),
    conflict=ConflictResolutionAgent(),
    memory=memory_agent,
    composer=OutputComposerAgent(),
    calibration=CalibrationAgent(),
)

_watchlist: dict[str, WatchlistEntry] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "vector_backend": settings.vector_backend,
        "llm_backend": type(llm_client).__name__,
    }


@app.post("/assess", response_model=AssessmentResponse)
def assess(request: AssessRequest) -> AssessmentResponse:
    return orchestrator.assess(request)


@app.post("/watchlist", response_model=WatchlistEntry, status_code=201)
def add_to_watchlist(entry: WatchlistEntry) -> WatchlistEntry:
    _watchlist[entry.entity_id] = entry
    return entry


@app.get("/watchlist/{entity_id}", response_model=WatchlistStatus)
def get_watchlist_status(entity_id: str) -> WatchlistStatus:
    entry = _watchlist.get(entity_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not on watchlist.")
    result = orchestrator.assess(
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
    return list(_watchlist.values())
