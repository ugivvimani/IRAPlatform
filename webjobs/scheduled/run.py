from __future__ import annotations

from app.orchestrator import OrchestratorAgent
from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.contracts import AssessRequest, UserQuery
from app.llm.factory import build_llm_client
from app.settings import load_settings
from app.storage.factory import build_storage_repository
from app.vector_store.factory import build_vector_store


def main() -> None:
    settings = load_settings()
    storage = build_storage_repository(settings)
    vector_store = build_vector_store()
    llm_client = build_llm_client()
    orchestrator = OrchestratorAgent(
        retrieval=RetrievalAgent(),
        analysis=AnalysisForecastingAgent(),
        conflict=ConflictResolutionAgent(),
        memory=MemoryManagerAgent(vector_store),
        composer=OutputComposerAgent(),
        calibration=CalibrationAgent(),
        llm_client=llm_client,
    )

    for entry in storage.list_watchlist():
        result = orchestrator.assess(
            AssessRequest(query=UserQuery(company_name=entry.company_name, question="Scheduled watchlist reassessment"))
        )
        storage.insert_assessment(result)


if __name__ == "__main__":
    main()
