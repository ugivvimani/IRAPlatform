from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends
from prometheus_client import make_asgi_app

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.auth import (
    get_authenticated_user,
    require_write_access,
    User,
    TokenManager,
)
from app.connectors import MultiSourceConnector, ConnectorConfig
from app.contracts import (
    AssessmentAuditRecord,
    AssessRequest,
    AssessmentResponse,
    UserQuery,
    WatchlistEntry,
    WatchlistStatus,
)
from app.embeddings import EmbeddingFactory
from app.llm.factory import build_llm_client
from app.observability import (
    MetricsMiddleware,
    HealthCheckService,
    setup_observability,
    instrument_assessment,
    registry,
)
from app.orchestrator import OrchestratorAgent
from app.settings import load_settings
from app.storage.factory import build_storage_repository
from app.vector_store.factory import build_vector_store

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Integrity Risk Assessment Agent",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Set up observability
setup_observability(app)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app(registry=registry)
app.mount("/metrics", metrics_app)

settings = load_settings()

# Initialize core components
vector_store = build_vector_store()
memory_agent = MemoryManagerAgent(vector_store)
llm_client = build_llm_client()
storage_repo = build_storage_repository(settings)
embedding_model = EmbeddingFactory.create(os.getenv("EMBEDDING_TYPE", "hybrid"))
connector_config = ConnectorConfig()
connectors = MultiSourceConnector(connector_config)
orchestrator = OrchestratorAgent(
    retrieval=RetrievalAgent(),
    analysis=AnalysisForecastingAgent(),
    conflict=ConflictResolutionAgent(),
    memory=memory_agent,
    composer=OutputComposerAgent(),
    calibration=CalibrationAgent(),
)

# Health check service
health_service = HealthCheckService(vector_store, storage_repo, llm_client)


# ============================================================================
# Health & Readiness Endpoints
# ============================================================================

@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return await health_service.get_health()


@app.get("/ready")
async def readiness() -> dict:
    """Readiness check endpoint."""
    return await health_service.get_readiness()


# ============================================================================
# Authentication & Token Endpoints
# ============================================================================

@app.post("/auth/token")
async def login(username: str, password: str) -> dict:
    """
    Generate a JWT token for API access.
    
    Args:
        username: User ID
        password: Password (in production, validate against secure store)
    
    Returns:
        JWT token for subsequent API calls
    """
    # TODO: In production, validate against secure password store
    token = TokenManager.create_access_token(
        subject=username,
        role="analyst",  # Determine role from user store
    )
    return {"access_token": token, "token_type": "bearer"}


# ============================================================================
# Assessment Endpoints
# ============================================================================

@app.post("/assess", response_model=AssessmentResponse)
@instrument_assessment
async def assess(
    request: AssessRequest,
    user: User = Depends(require_write_access),
) -> AssessmentResponse:
    """
    Perform a risk assessment for an entity.
    Requires write access (analyst role or higher).
    """
    logger.info(f"Assessment requested by {user.user_id} for {request.query.company_name}")
    result = orchestrator.assess(request)
    storage_repo.insert_assessment(result)
    return result


# ============================================================================
# Watchlist Endpoints
# ============================================================================

@app.post("/watchlist", response_model=WatchlistEntry, status_code=201)
async def add_to_watchlist(
    entry: WatchlistEntry,
    user: User = Depends(require_write_access),
) -> WatchlistEntry:
    """Add or update an entity on the watchlist."""
    logger.info(f"Watchlist entry created by {user.user_id} for {entry.entity_id}")
    return storage_repo.upsert_watchlist(entry)


@app.get("/watchlist/{entity_id}", response_model=WatchlistStatus)
async def get_watchlist_status(
    entity_id: str,
    user: User = Depends(get_authenticated_user),
) -> WatchlistStatus:
    """Get current status of a watchlisted entity."""
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
async def list_watchlist(
    user: User = Depends(get_authenticated_user),
) -> list[WatchlistEntry]:
    """List all entities on the watchlist."""
    return storage_repo.list_watchlist()


@app.delete("/watchlist/{entity_id}", status_code=204)
async def remove_from_watchlist(
    entity_id: str,
    user: User = Depends(require_write_access),
) -> None:
    """Remove an entity from the watchlist."""
    logger.info(f"Watchlist entry deleted by {user.user_id} for {entity_id}")
    storage_repo.delete_watchlist(entity_id)


# ============================================================================
# Assessment Audit Endpoints
# ============================================================================

@app.get("/assessments/{entity_id}", response_model=list[AssessmentAuditRecord])
async def list_assessments(
    entity_id: str,
    limit: int = 25,
    user: User = Depends(get_authenticated_user),
) -> list[AssessmentAuditRecord]:
    """Get assessment audit trail for an entity."""
    return storage_repo.list_assessments(entity_id=entity_id, limit=limit)


# ============================================================================
# External Data Connector Endpoints (for testing/debugging)
# ============================================================================

@app.get("/debug/connectors/{entity_name}")
async def debug_connectors(
    entity_name: str,
    user: User = Depends(get_authenticated_user),
) -> dict:
    """
    Debug endpoint: Fetch evidence from all external connectors.
    WARNING: For development/testing only. Remove in production.
    """
    logger.info(f"Debug connector call by {user.user_id} for {entity_name}")
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


# ============================================================================
# Embedding Endpoints (for testing/debugging)
# ============================================================================

@app.post("/debug/embed")
async def debug_embed(
    texts: list[str],
    user: User = Depends(get_authenticated_user),
) -> dict:
    """
    Debug endpoint: Generate embeddings for text.
    WARNING: For development/testing only. Remove in production.
    """
    embeddings = embedding_model.embed_sync(texts)
    return {
        "texts": texts,
        "embeddings_count": len(embeddings),
        "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
    }


import os
