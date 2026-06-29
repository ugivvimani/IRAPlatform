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
    from app.storage.base import StorageRepository

logger = logging.getLogger(__name__)

_JOB_STATE: dict[str, dict[str, Any]] = {}
_LOCAL_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()
_orchestrator_ref: "OrchestratorAgent | None" = None
_storage_ref: "StorageRepository | None" = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_local_worker(
    orchestrator: "OrchestratorAgent",
    storage_repo: "StorageRepository | None" = None,
) -> None:
    """Call once at startup (local mode only) to start the background task worker."""
    global _orchestrator_ref, _storage_ref, _worker_started
    with _worker_lock:
        _orchestrator_ref = orchestrator
        _storage_ref = storage_repo
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

            # Persist to DB so result survives restarts and is visible via
            # GET /assessments/{entity_id} alongside sync assessments.
            assessment_id: int | None = None
            if _storage_ref is not None:
                try:
                    assessment_id = _storage_ref.insert_assessment(result)
                except Exception as db_exc:
                    logger.warning("webjobs_db_persist_failed job_id=%s error=%s", job_id, db_exc)

            result_dict = result.model_dump()
            if assessment_id is not None:
                result_dict["assessment_id"] = assessment_id

            _JOB_STATE[job_id]["status"] = "completed"
            _JOB_STATE[job_id]["result"] = result_dict
            _JOB_STATE[job_id]["completed_at"] = _utc_now()
            logger.info(
                "webjobs_job_done job_id=%s risk=%s assessment_id=%s",
                job_id, result.decision.risk_rating, assessment_id,
            )

            # Persist job status to DB for restart-safe polling
            if _storage_ref is not None:
                try:
                    _storage_ref.upsert_async_job(
                        job_id=job_id,
                        status="completed",
                        assessment_id=assessment_id,
                        entity_id=payload["company_name"],
                    )
                except Exception as db_exc:
                    logger.warning("webjobs_db_job_update_failed job_id=%s error=%s", job_id, db_exc)

            # Webhook delivery — fire-and-forget, non-blocking
            callback_url = payload.get("callback_url")
            if callback_url:
                _deliver_webhook(job_id, callback_url, result_dict)

        except Exception as exc:
            _JOB_STATE[job_id]["status"] = "failed"
            _JOB_STATE[job_id]["error"] = str(exc)
            _JOB_STATE[job_id]["completed_at"] = _utc_now()
            logger.exception("webjobs_job_failed job_id=%s error=%s", job_id, exc)
            if _storage_ref is not None:
                try:
                    _storage_ref.upsert_async_job(
                        job_id=job_id,
                        status="failed",
                        assessment_id=None,
                        entity_id=payload.get("company_name", "unknown"),
                    )
                except Exception:
                    pass
        finally:
            _LOCAL_QUEUE.task_done()


def _deliver_webhook(job_id: str, callback_url: str, result: dict[str, Any]) -> None:
    """POST the completed assessment result to the caller's callback URL.
    Runs in a short-lived daemon thread so it doesn't block the worker queue.
    """
    def _post() -> None:
        try:
            import httpx
            payload = {
                "task_id": job_id,
                "status": "completed",
                "result": result,
                "delivered_at": _utc_now(),
            }
            resp = httpx.post(callback_url, json=payload, timeout=10)
            logger.info(
                "webjobs_webhook_delivered job_id=%s url=%s status=%d",
                job_id, callback_url, resp.status_code,
            )
        except Exception as exc:
            logger.warning("webjobs_webhook_failed job_id=%s url=%s error=%s", job_id, callback_url, exc)

    threading.Thread(target=_post, daemon=True, name=f"webhook-{job_id[:8]}").start()


def enqueue_assessment_job(
    company_name: str,
    question: str,
    callback_url: str | None = None,
) -> str:
    job_id = str(uuid.uuid4())
    payload: dict[str, Any] = {
        "job_id": job_id,
        "company_name": company_name,
        "question": question,
        "created_at": _utc_now(),
    }
    if callback_url:
        payload["callback_url"] = callback_url

    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    queue_name = os.getenv("AZURE_WEBJOBS_ASSESSMENT_QUEUE", "assessment-jobs")

    if storage_conn:
        try:
            from azure.storage.queue import QueueClient
            azure_queue = QueueClient.from_connection_string(storage_conn, queue_name)
            azure_queue.create_queue()
            azure_queue.send_message(json.dumps(payload))
            _JOB_STATE[job_id] = {"status": "queued", "provider": "azure_queue", "created_at": _utc_now()}
            return job_id
        except Exception:
            pass

    # Local in-process queue — worker drains this via _worker_loop
    _JOB_STATE[job_id] = {"status": "queued", "provider": "in_memory", "created_at": _utc_now()}
    _LOCAL_QUEUE.put(payload)
    return job_id


def get_job_status(job_id: str, storage_repo: "StorageRepository | None" = None) -> dict[str, Any]:
    """Return job status.

    Checks in-memory state first (fast path). If the job is unknown (e.g. after
    a server restart), falls back to the DB to find the most recent assessment
    for the entity — so completed results are never silently lost.
    """
    state = _JOB_STATE.get(job_id)
    if state:
        response: dict[str, Any] = {
            "task_id": job_id,
            "status": state.get("status", "unknown"),
            "created_at": state.get("created_at"),
            "completed_at": state.get("completed_at"),
        }
        if state.get("status") == "completed" and "result" in state:
            response["result"] = state["result"]
        if state.get("status") == "failed" and "error" in state:
            response["error"] = state["error"]
        return response

    # Not in memory — check DB for a persisted result keyed by task_id
    if storage_repo is not None:
        try:
            record = storage_repo.get_async_job(job_id)
            if record:
                return {
                    "task_id": job_id,
                    "status": record.get("status", "completed"),
                    "assessment_id": record.get("assessment_id"),
                    "completed_at": record.get("completed_at"),
                    "result": record.get("result"),
                }
        except Exception:
            pass

    return {"task_id": job_id, "status": "unknown"}
