from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.orchestrator import OrchestratorAgent

logger = logging.getLogger(__name__)

_JOB_STATE: dict[str, dict[str, Any]] = {}
_LOCAL_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()
_orchestrator_ref: "OrchestratorAgent | None" = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_local_worker(orchestrator: "OrchestratorAgent") -> None:
    """Call once at startup (local mode only) to start the background task worker."""
    global _orchestrator_ref, _worker_started
    _orchestrator_ref = orchestrator
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, daemon=True, name="webjobs-worker")
            t.start()
            _worker_started = True
            logger.info("webjobs_local_worker_started")


def _worker_loop() -> None:
    while True:
        payload = _LOCAL_QUEUE.get()
        job_id = payload["job_id"]
        if _orchestrator_ref is None:
            _JOB_STATE[job_id]["status"] = "failed"
            _JOB_STATE[job_id]["error"] = "orchestrator_not_initialized"
            _LOCAL_QUEUE.task_done()
            continue
        _JOB_STATE[job_id]["status"] = "processing"
        logger.info("webjobs_job_start job_id=%s entity=%s", job_id, payload["company_name"])
        try:
            from app.contracts import AssessRequest, UserQuery
            result = _orchestrator_ref.assess(
                AssessRequest(
                    query=UserQuery(
                        company_name=payload["company_name"],
                        question=payload.get("question", ""),
                    )
                )
            )
            _JOB_STATE[job_id]["status"] = "completed"
            _JOB_STATE[job_id]["result"] = result.model_dump()
            logger.info("webjobs_job_done job_id=%s risk=%s", job_id, result.decision.risk_rating)
        except Exception as exc:
            _JOB_STATE[job_id]["status"] = "failed"
            _JOB_STATE[job_id]["error"] = str(exc)
            logger.exception("webjobs_job_failed job_id=%s error=%s", job_id, exc)
        finally:
            _LOCAL_QUEUE.task_done()


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

            azure_queue = QueueClient.from_connection_string(storage_conn, queue_name)
            azure_queue.create_queue()
            azure_queue.send_message(json.dumps(payload))
            _JOB_STATE[job_id] = {"status": "queued", "provider": "azure_queue"}
            return job_id
        except Exception:
            pass

    # Local in-process queue — worker drains this via _worker_loop
    _JOB_STATE[job_id] = {"status": "queued", "provider": "in_memory"}
    _LOCAL_QUEUE.put(payload)
    return job_id


def get_job_status(job_id: str) -> dict[str, Any]:
    state = _JOB_STATE.get(job_id, {})
    response: dict[str, Any] = {
        "task_id": job_id,
        "status": state.get("status", "unknown"),
    }
    if state.get("status") == "completed" and "result" in state:
        response["result"] = state["result"]
    if state.get("status") == "failed" and "error" in state:
        response["error"] = state["error"]
    return response
