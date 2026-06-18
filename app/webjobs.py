from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

_JOB_STATE: dict[str, dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def enqueue_assessment_job(company_name: str, question: str) -> str:
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "company_name": company_name,
        "question": question,
        "created_at": _utc_now(),
    }

    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    queue_name = os.getenv("AZURE_WEBJOBS_ASSESSMENT_QUEUE", "assessment-jobs")

    if storage_conn:
        try:
            from azure.storage.queue import QueueClient

            queue = QueueClient.from_connection_string(storage_conn, queue_name)
            queue.create_queue()
            queue.send_message(json.dumps(payload))
            _JOB_STATE[job_id] = {"status": "queued", "provider": "azure_queue"}
            return job_id
        except Exception:
            pass

    _JOB_STATE[job_id] = {"status": "queued", "provider": "in_memory"}
    return job_id


def get_job_status(job_id: str) -> dict[str, Any]:
    return {
        "task_id": job_id,
        "status": _JOB_STATE.get(job_id, {}).get("status", "unknown"),
    }
