from __future__ import annotations

from fastapi import FastAPI

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.contracts import AssessRequest, AssessmentResponse
from app.orchestrator import OrchestratorAgent
from app.settings import load_settings
from app.vector_store.factory import build_vector_store

app = FastAPI(title="Integrity Risk Assessment Agent", version="0.1.0")
settings = load_settings()

vector_store = build_vector_store()
memory_agent = MemoryManagerAgent(vector_store)
orchestrator = OrchestratorAgent(
    retrieval=RetrievalAgent(),
    analysis=AnalysisForecastingAgent(),
    conflict=ConflictResolutionAgent(),
    memory=memory_agent,
    composer=OutputComposerAgent(),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "vector_backend": settings.vector_backend,
    }


@app.post("/assess", response_model=AssessmentResponse)
def assess(request: AssessRequest) -> AssessmentResponse:
    return orchestrator.assess(request)
