from __future__ import annotations

import json
import os
import time

from app.orchestrator import OrchestratorAgent
from app.agents.analysis_forecasting import AnalysisForecastingAgent
from app.agents.calibration import CalibrationAgent
from app.agents.conflict_resolution import ConflictResolutionAgent
from app.agents.memory_manager import MemoryManagerAgent
from app.agents.output_composer import OutputComposerAgent
from app.agents.retrieval import RetrievalAgent
from app.contracts import AssessRequest, UserQuery
from app.settings import load_settings
from app.storage.factory import build_storage_repository
from app.vector_store.factory import build_vector_store


def main() -> None:
    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    queue_name = os.getenv("AZURE_WEBJOBS_ASSESSMENT_QUEUE", "assessment-jobs")
    if not storage_conn:
        return

    from azure.storage.queue import QueueClient

    queue = QueueClient.from_connection_string(storage_conn, queue_name)
    queue.create_queue()

    settings = load_settings()
    storage = build_storage_repository(settings)
    vector_store = build_vector_store()
    orchestrator = OrchestratorAgent(
        retrieval=RetrievalAgent(),
        analysis=AnalysisForecastingAgent(),
        conflict=ConflictResolutionAgent(),
        memory=MemoryManagerAgent(vector_store),
        composer=OutputComposerAgent(),
        calibration=CalibrationAgent(),
    )

    while True:
        messages = queue.receive_messages(messages_per_page=5, visibility_timeout=30)
        for message in messages:
            try:
                payload = json.loads(message.content)
                company = payload.get("company_name", "")
                question = payload.get("question", "Queued assessment")
                if company:
                    result = orchestrator.assess(AssessRequest(query=UserQuery(company_name=company, question=question)))
                    storage.insert_assessment(result)
                queue.delete_message(message.id, message.pop_receipt)
            except Exception:
                continue
        time.sleep(5)


if __name__ == "__main__":
    main()
