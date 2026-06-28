from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_health_service, get_llm_client, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings=Depends(get_settings), llm_client=Depends(get_llm_client)) -> dict:
    return {
        "status": "ok",
        "env": settings.app_env,
        "vector_backend": settings.vector_backend,
        "llm_backend": type(llm_client).__name__,
        "storage_backend": settings.db_backend,
    }


@router.get("/ready")
async def readiness(health_service=Depends(get_health_service)) -> dict:
    return await health_service.get_readiness()

