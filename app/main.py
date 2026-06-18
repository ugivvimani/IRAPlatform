from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import Body, Depends, FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.auth import TokenManager, User, get_authenticated_user, require_write_access
from app.connectors import ConnectorConfig, MultiSourceConnector
from app.contracts import (
    AssessRequest,
    AssessmentAuditRecord,
    AssessmentResponse,
    UserQuery,
    WatchlistEntry,
    WatchlistStatus,
)
from app.llm.factory import build_llm_client
from app.observability import HealthCheckService, MetricsMiddleware, instrument_assessment, setup_observability
from app.orchestrator import OrchestratorAgent
from app.settings import load_settings
from app.storage.factory import build_storage_repository
from app.vector_store.factory import build_vector_store
from app.webjobs import enqueue_assessment_job, get_job_status

logger = logging.getLogger(__name__)


class AsyncAssessRequest(BaseModel):
    company_name: str
    question: str


class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str


try:
    from app.embeddings import EmbeddingFactory
except Exception:
    class EmbeddingFactory:  # type: ignore[override]
        @staticmethod
        def create(_embedding_type: str = "hybrid"):
            class _FallbackEmbedding:
                def embed_sync(self, texts: list[str]) -> list[list[float]]:
                    return [[0.0] * 8 for _ in texts]

            return _FallbackEmbedding()


app = FastAPI(
    title="Integrity Risk Assessment Agent",
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

setup_observability(app)
app.add_middleware(MetricsMiddleware)

settings = load_settings()
vector_store = build_vector_store()
memory_agent = MemoryManagerAgent(vector_store)
llm_client = build_llm_client()
storage_repo = build_storage_repository(settings)
embedding_model = EmbeddingFactory.create(os.getenv("EMBEDDING_TYPE", "hybrid"))
connectors = MultiSourceConnector(ConnectorConfig())

orchestrator = OrchestratorAgent(
    retrieval=RetrievalAgent(),
    analysis=AnalysisForecastingAgent(),
    conflict=ConflictResolutionAgent(),
    memory=memory_agent,
    composer=OutputComposerAgent(),
    calibration=CalibrationAgent(),
)

health_service = HealthCheckService(vector_store, storage_repo, llm_client)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "env": settings.app_env,
        "vector_backend": settings.vector_backend,
        "llm_backend": type(llm_client).__name__,
        "storage_backend": settings.db_backend,
    }


@app.get("/ready")
async def readiness() -> dict:
    return await health_service.get_readiness()


@app.post("/auth/token")
async def login(username: str, password: str) -> dict:
    token = TokenManager.create_access_token(subject=username, role="analyst")
    return {"access_token": token, "token_type": "bearer"}


@app.post("/assess", response_model=AssessmentResponse)
async def assess(request: AssessRequest = Body(...), user: User = Depends(require_write_access)) -> AssessmentResponse:
    logger.info("Assessment requested by %s for %s", user.user_id, request.query.company_name)
    result = orchestrator.assess(request)
    storage_repo.insert_assessment(result)
    return result


@app.post("/assess/async", response_model=AsyncTaskResponse)
async def assess_async(request: AsyncAssessRequest = Body(...), user: User = Depends(require_write_access)) -> AsyncTaskResponse:
    task_id = enqueue_assessment_job(request.company_name, request.question)
    return AsyncTaskResponse(task_id=task_id, status="queued")


@app.get("/tasks/{task_id}")
async def task_status(task_id: str, user: User = Depends(get_authenticated_user)) -> dict:
    return get_job_status(task_id)


@app.post("/watchlist", response_model=WatchlistEntry, status_code=201)
async def add_to_watchlist(entry: WatchlistEntry, user: User = Depends(require_write_access)) -> WatchlistEntry:
    return storage_repo.upsert_watchlist(entry)


@app.get("/watchlist/{entity_id}", response_model=WatchlistStatus)
async def get_watchlist_status(entity_id: str, user: User = Depends(get_authenticated_user)) -> WatchlistStatus:
    entry = storage_repo.get_watchlist(entity_id)
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
async def list_watchlist(user: User = Depends(get_authenticated_user)) -> list[WatchlistEntry]:
    return storage_repo.list_watchlist()


@app.delete("/watchlist/{entity_id}", status_code=200)
async def remove_from_watchlist(entity_id: str, user: User = Depends(require_write_access)) -> None:
    storage_repo.delete_watchlist(entity_id)


@app.get("/assessments/{entity_id}", response_model=list[AssessmentAuditRecord])
async def list_assessments(
    entity_id: str, limit: int = 25, user: User = Depends(get_authenticated_user)
) -> list[AssessmentAuditRecord]:
    return storage_repo.list_assessments(entity_id=entity_id, limit=limit)


@app.get("/debug/connectors/{entity_name}")
async def debug_connectors(entity_name: str, user: User = Depends(get_authenticated_user)) -> dict:
    evidence = await connectors.fetch_all(entity_name)
    return {
        "entity": entity_name,
        "evidence_count": len(evidence),
        "evidence": [
            {
                "signal": e.signal,
                "value": e.value,
                "source": e.source_name,
                "confidence": e.source_confidence,
            }
            for e in evidence
        ],
    }


@app.post("/debug/embed")
async def debug_embed(texts: list[str], user: User = Depends(get_authenticated_user)) -> dict:
    embeddings = embedding_model.embed_sync(texts)
    return {
        "texts": texts,
        "embeddings_count": len(embeddings),
        "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
    }




