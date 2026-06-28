"""Compatibility layer for legacy task imports.

Celery has been replaced with Azure WebJobs + Azure Storage Queue.
"""
from __future__ import annotations

from app.webjobs import enqueue_assessment_job, get_job_status


def assess_entity_async(entity_name: str, question: str = "") -> dict[str, str]:
    task_id = enqueue_assessment_job(entity_name, question)
    return {"task_id": task_id, "status": "queued"}


def reassess_watchlist() -> dict[str, str]:
    return {"status": "moved_to_webjobs_scheduled"}


def cleanup_stale_assessments() -> dict[str, str]:
    return {"status": "moved_to_webjobs_scheduled"}


def update_calibration_metrics() -> dict[str, str]:
    return {"status": "moved_to_webjobs_scheduled"}


def fetch_external_data(entity_name: str, sources: list[str] | None = None) -> dict[str, str]:
    return {"status": "moved_to_webjobs_continuous", "entity": entity_name}


class _LegacyCeleryShim:
    @staticmethod
    def AsyncResult(task_id: str):
        class _Result:
            def __init__(self, id_: str):
                self.id = id_
                self.status = get_job_status(id_).get("status", "unknown").upper()
                self.result = None

            def successful(self) -> bool:
                return self.status == "SUCCEEDED"

            def failed(self) -> bool:
                return self.status == "FAILED"

        return _Result(task_id)


celery_app = _LegacyCeleryShim()
