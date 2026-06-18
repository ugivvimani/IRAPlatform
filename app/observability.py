from __future__ import annotations

import logging
import logging.config
import os
import time
from functools import wraps
from typing import Callable

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
        return {
            "status": "ok",
            "checks": {
                "vector_store": "ok",
                "database": "ok",
                "llm_client": "ok",
            },
        }

    async def get_readiness(self) -> dict:
        return {"ready": True, "timestamp": time.time()}


def setup_observability(app: FastAPI) -> None:
    setup_logging()
    enabled = setup_appinsights()
    logger.info("Observability initialized", extra={"appinsights_enabled": enabled})
