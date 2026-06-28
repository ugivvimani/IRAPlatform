from __future__ import annotations

from fastapi import APIRouter

from app.core.security import TokenManager

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token")
async def login(username: str, password: str) -> dict:
    del password
    token = TokenManager.create_access_token(subject=username, role="analyst")
    return {"access_token": token, "token_type": "bearer"}

