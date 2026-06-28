from __future__ import annotations

import logging
import logging.config
import os
import time
from functools import wraps
from typing import Any, Callable

from fastapi import FastAPI

logger = logging.getLogger(__name__)

_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if os.getenv("LOG_FORMAT") == "json" else "default",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "handlers": ["console"],
    },
}

ASSESSMENT_COMPLETED_EVENT = "IRA.AssessmentCompleted"
CALIBRATION_EVALUATED_EVENT = "IRA.CalibrationEvaluated"
MEMORY_WRITE_EVALUATED_EVENT = "IRA.MemoryWriteEvaluated"
TOT_BRANCH_EVALUATED_EVENT = "IRA.ToTBranchEvaluated"



def setup_logging() -> None:
    try:
        import pythonjsonlogger.jsonlogger  # noqa: F401
    except ImportError:
        _LOGGING_CONFIG["formatters"]["json"] = _LOGGING_CONFIG["formatters"]["default"]
    logging.config.dictConfig(_LOGGING_CONFIG)



def setup_appinsights() -> bool:
    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    if not conn:
        return False

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=conn)
        logger.info("Application Insights configured")
        return True
    except Exception as exc:
        logger.warning("Application Insights setup failed: %s", exc)
        return False


class MetricsMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")
        start_time = time.time()

        async def send_with_metrics(message):
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
                duration = time.time() - start_time
                logger.info(
                    "request_completed",
                    extra={
                        "method": method,
                        "path": path,
                        "status_code": status_code,
                        "duration_seconds": duration,
                    },
                )
            await send(message)

        await self.app(scope, receive, send_with_metrics)



def instrument_assessment(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info("assessment_completed", extra={"duration_seconds": duration})
            return result
        except Exception as exc:
            duration = time.time() - start_time
            logger.error("assessment_failed", extra={"error": str(exc), "duration_seconds": duration})
            raise

    return wrapper


class HealthCheckService:
    def __init__(self, vector_store, storage_repo, llm_client):
        self.vector_store = vector_store
        self.storage_repo = storage_repo
        self.llm_client = llm_client

    async def get_health(self) -> dict:
        checks: dict[str, str] = {}

        # Probe vector store
        try:
            self.vector_store.query(namespace="health", text="ping", top_k=1)
            checks["vector_store"] = "ok"
        except Exception as exc:
            checks["vector_store"] = f"error: {exc}"

        # Probe database
        try:
            self.storage_repo.list_watchlist()
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {exc}"

        # LLM — just check it is configured (avoid billing a real call)
        checks["llm_client"] = type(self.llm_client).__name__

        overall = "ok" if all(v == "ok" or "Client" in v or "Stub" in v for v in checks.values()) else "degraded"
        return {"status": overall, "checks": checks}

    async def get_readiness(self) -> dict:
        health = await self.get_health()
        ready = health["status"] != "degraded"
        return {"ready": ready, "timestamp": time.time()}



def setup_observability(app: FastAPI) -> None:
    setup_logging()
    enabled = setup_appinsights()
    logger.info("Observability initialized", extra={"appinsights_enabled": enabled})


_CONFIDENCE_TO_SCORE = {
    "low": 0.33,
    "medium": 0.66,
    "high": 0.9,
}


def confidence_to_score(confidence: Any) -> float:
    return _CONFIDENCE_TO_SCORE[confidence.value if hasattr(confidence, "value") else str(confidence)]



def build_assessment_telemetry(
    request,
    decision,
    conflict_result,
    evidence_count: int,
    quant_scores: dict[str, float] | None = None,
    evaluated_at=None,
) -> dict[str, Any]:
    evaluated_at = evaluated_at or __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    conflict_detected = conflict_result.conflict_detected if conflict_result else False
    alternative_count = len(conflict_result.alternatives) if conflict_result else 0
    telemetry: dict[str, Any] = {
        "event_name": ASSESSMENT_COMPLETED_EVENT,
        "entity_id": request.query.company_name,
        "question": request.query.question,
        "risk_rating": decision.risk_rating.value,
        "confidence": decision.confidence.value,
        "confidence_score": confidence_to_score(decision.confidence),
        "requires_manual_review": decision.requires_manual_review,
        "evidence_count": evidence_count,
        "conflict_detected": conflict_detected,
        "alternative_count": alternative_count,
        "evaluated_at": evaluated_at.isoformat(),
    }
    if quant_scores:
        telemetry["quant_scores"] = quant_scores
    if conflict_result:
        telemetry["conflict_rationale"] = conflict_result.rationale
    return telemetry



def build_calibration_telemetry(
    entity_id: str,
    source_name: str,
    predicted_risky: bool,
    actual_risky: bool,
    confidence_score: float,
    is_correct: bool,
    evaluated_at=None,
) -> dict[str, Any]:
    evaluated_at = evaluated_at or __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    return {
        "event_name": CALIBRATION_EVALUATED_EVENT,
        "entity_id": entity_id,
        "source_name": source_name,
        "predicted_risky": predicted_risky,
        "actual_risky": actual_risky,
        "confidence_score": confidence_score,
        "is_correct": is_correct,
        "evaluated_at": evaluated_at.isoformat(),
    }
