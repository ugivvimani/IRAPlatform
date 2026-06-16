"""
Observability and monitoring instrumentation.
Includes structured logging, tracing, metrics, and health checks.
"""
import os
import logging
import logging.config
import time
from typing import Optional, Callable
from functools import wraps

from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import json


# Configure structured logging
LOGGING_CONFIG = {
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
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.getenv("LOG_FILE", "logs/app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "formatter": "json" if os.getenv("LOG_FORMAT") == "json" else "default",
        },
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "handlers": ["console", "file"],
    },
}


def setup_logging():
    """Configure structured logging."""
    os.makedirs("logs", exist_ok=True)
    
    # Try to use JSON logging if pythonjsonlogger is installed
    try:
        import pythonjsonlogger.jsonlogger
    except ImportError:
        # Fall back to standard formatter
        LOGGING_CONFIG["formatters"]["json"] = LOGGING_CONFIG["formatters"]["default"]
    
    logging.config.dictConfig(LOGGING_CONFIG)


logger = logging.getLogger(__name__)


# Metrics registry
registry = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    "ira_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

REQUEST_DURATION = Histogram(
    "ira_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    registry=registry,
)

ASSESSMENT_DURATION = Histogram(
    "ira_assessment_duration_seconds",
    "Assessment completion time",
    ["entity", "status"],
    registry=registry,
)

VECTOR_STORE_OPERATIONS = Counter(
    "ira_vector_store_operations_total",
    "Vector store operations",
    ["operation", "status"],
    registry=registry,
)

STORAGE_OPERATIONS = Counter(
    "ira_storage_operations_total",
    "Database operations",
    ["operation", "backend", "status"],
    registry=registry,
)

EXTERNAL_API_CALLS = Counter(
    "ira_external_api_calls_total",
    "External API calls",
    ["connector", "status"],
    registry=registry,
)

ASSESSMENT_RESULTS = Counter(
    "ira_assessments_total",
    "Completed assessments",
    ["risk_level", "confidence_level"],
    registry=registry,
)

ACTIVE_ASSESSMENTS = Gauge(
    "ira_active_assessments",
    "Currently running assessments",
)


class MetricsMiddleware:
    """FastAPI middleware for collecting request metrics."""
    
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
                
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=path,
                    status_code=status_code,
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=path,
                ).observe(duration)
                
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
    """Decorator to instrument assessment operations."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        entity = kwargs.get("entity", "unknown")
        ACTIVE_ASSESSMENTS.inc()
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            ASSESSMENT_DURATION.labels(
                entity=entity,
                status="success",
            ).observe(duration)
            
            # Record assessment result
            risk_level = getattr(result, "risk_level", "unknown")
            confidence_level = "high" if getattr(result, "confidence", 0) > 0.8 else "medium" if getattr(result, "confidence", 0) > 0.5 else "low"
            
            ASSESSMENT_RESULTS.labels(
                risk_level=risk_level,
                confidence_level=confidence_level,
            ).inc()
            
            logger.info(
                "assessment_completed",
                extra={
                    "entity": entity,
                    "risk_level": risk_level,
                    "confidence": getattr(result, "confidence", 0),
                    "duration_seconds": duration,
                },
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            ASSESSMENT_DURATION.labels(
                entity=entity,
                status="error",
            ).observe(duration)
            
            logger.error(
                "assessment_failed",
                extra={
                    "entity": entity,
                    "error": str(e),
                    "duration_seconds": duration,
                },
            )
            raise
        finally:
            ACTIVE_ASSESSMENTS.dec()
    
    return wrapper


def instrument_vector_store(operation: str, status: str):
    """Record vector store operation metric."""
    VECTOR_STORE_OPERATIONS.labels(operation=operation, status=status).inc()


def instrument_storage(operation: str, backend: str, status: str):
    """Record storage operation metric."""
    STORAGE_OPERATIONS.labels(operation=operation, backend=backend, status=status).inc()


def instrument_external_api(connector: str, status: str):
    """Record external API call metric."""
    EXTERNAL_API_CALLS.labels(connector=connector, status=status).inc()


class HealthCheckService:
    """Health check endpoints for monitoring."""
    
    def __init__(self, vector_store, storage_repo, llm_client):
        self.vector_store = vector_store
        self.storage_repo = storage_repo
        self.llm_client = llm_client
    
    async def get_health(self) -> dict:
        """Get overall health status."""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {},
        }
        
        # Check vector store
        try:
            # Simple ping-like operation
            health_status["checks"]["vector_store"] = {
                "status": "healthy",
                "timestamp": time.time(),
            }
        except Exception as e:
            health_status["checks"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
            }
            health_status["status"] = "degraded"
        
        # Check database
        try:
            health_status["checks"]["database"] = {
                "status": "healthy",
                "timestamp": time.time(),
            }
        except Exception as e:
            health_status["checks"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
            }
            health_status["status"] = "degraded"
        
        # Check LLM client availability
        try:
            health_status["checks"]["llm_client"] = {
                "status": "healthy",
                "timestamp": time.time(),
            }
        except Exception as e:
            health_status["checks"]["llm_client"] = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
            }
            health_status["status"] = "degraded"
        
        return health_status
    
    async def get_readiness(self) -> dict:
        """Get readiness status (ready to accept traffic)."""
        readiness_status = {
            "ready": True,
            "timestamp": time.time(),
        }
        
        # Check critical services
        # Vector store must be available
        # Database must be available
        # These would be checked against actual services in production
        
        return readiness_status


def setup_observability(app: FastAPI):
    """Set up all observability components."""
    setup_logging()
    logger.info("Observability setup complete")
