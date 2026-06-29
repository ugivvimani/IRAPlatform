from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")

from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.contracts import AssessRequest, UserQuery
from app.llm.factory import build_llm_client
from app.services.embeddings import EmbeddingFactory
from app.services.orchestrator import OrchestratorAgent
from app.vector_store.pinecone_store import PineconeVectorStore


def run() -> None:
    llm_client = build_llm_client()
    embedding_model = EmbeddingFactory.create(os.getenv("EMBEDDING_TYPE", "openrouter"))
    vector_store = PineconeVectorStore(embedding_fn=embedding_model.embed_sync)
    memory_agent = MemoryManagerAgent(vector_store, llm_client=llm_client)
    orchestrator = OrchestratorAgent(
        retrieval=RetrievalAgent(),
        analysis=AnalysisForecastingAgent(),
        conflict=ConflictResolutionAgent(),
        memory=memory_agent,
        composer=OutputComposerAgent(),
        calibration=CalibrationAgent(),
        llm_client=llm_client,
    )

    scenarios = [
        ("Company X", "Is Company X safe to partner with?"),
        ("Company Y", "Any sanctions or integrity red flags for Company Y?"),
        ("Company Z", "Do we need manual review before onboarding Company Z?"),
    ]
    for company, question in scenarios:
        result = orchestrator.assess(AssessRequest(query=UserQuery(company_name=company, question=question)))
        print(
            f"{company}: rating={result.decision.risk_rating.value}, "
            f"confidence={result.decision.confidence.value}, manual_review={result.decision.requires_manual_review}"
        )


if __name__ == "__main__":
    run()
