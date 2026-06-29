from __future__ import annotations

import os

# Load .env before anything else so all env vars are available at import time
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # override=False: real env vars take precedence
except ImportError:
    pass

from fastapi import FastAPI

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.api.deps import AppState
from app.api.routers.assessments import router as assessments_router
from app.api.routers.health import router as health_router
from app.api.routers.policies import router as policies_router
from app.api.routers.watchlist import router as watchlist_router
from app.core.config import load_settings
from app.core.observability import HealthCheckService, MetricsMiddleware, setup_observability
from app.llm.factory import build_llm_client
from app.services.connectors import ConnectorConfig, MultiSourceConnector
from app.services.embeddings import EmbeddingFactory
from app.services.orchestrator import OrchestratorAgent
from app.services.webjobs import init_local_worker
from app.storage.factory import build_storage_repository
from app.vector_store.factory import build_vector_store


def create_app() -> FastAPI:
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
    llm_client = build_llm_client()
    storage_repo = build_storage_repository(settings)
    connectors = MultiSourceConnector(ConnectorConfig())

    # Build embedding model and wire it as the vector store embedding function
    embedding_model = EmbeddingFactory.create(os.getenv("EMBEDDING_TYPE", "openrouter"))
    vector_store = build_vector_store(embedding_fn=embedding_model.embed_sync)

    memory_agent = MemoryManagerAgent(vector_store, llm_client=llm_client)

    orchestrator = OrchestratorAgent(
        retrieval=RetrievalAgent(live_connector=connectors),
        analysis=AnalysisForecastingAgent(),
        conflict=ConflictResolutionAgent(),
        memory=memory_agent,
        composer=OutputComposerAgent(),
        calibration=CalibrationAgent(),
        llm_client=llm_client,
        storage_repo=storage_repo,
    )

    app.state.app_state = AppState(
        settings=settings,
        llm_client=llm_client,
        vector_store=vector_store,
        storage_repo=storage_repo,
        orchestrator=orchestrator,
        health_service=HealthCheckService(vector_store, storage_repo, llm_client),
        connectors=connectors,
        embedding_model=embedding_model,
    )

    # Start local background worker so /assess/async jobs are actually executed
    if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        init_local_worker(orchestrator, storage_repo=storage_repo)

    app.include_router(health_router)
    app.include_router(assessments_router)
    app.include_router(watchlist_router)
    app.include_router(policies_router)
    return app


app = create_app()
