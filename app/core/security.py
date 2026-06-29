"""
Simple API key authentication for the IRA agentic service.
The upstream integrity platform passes a shared key via the X-API-Key header.
Set SERVICE_API_KEY in the environment; leave it unset to run open (local dev).
"""
from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _configured_key() -> str | None:
    return os.getenv("SERVICE_API_KEY", "").strip() or None


async def require_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> None:
    """FastAPI dependency - validates the X-API-Key header.

    - If SERVICE_API_KEY is not set the service runs open (local dev / testing).
    - If SERVICE_API_KEY is set every request must supply the matching key.
    """
    expected = _configured_key()
    if expected is None:
        return  # auth disabled
    if not api_key or api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
