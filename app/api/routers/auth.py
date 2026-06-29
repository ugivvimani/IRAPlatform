from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.security import USERS_DB, PasswordManager, RBACEnforcer, TokenManager

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token")
async def login(username: str, password: str) -> dict:
    user_data = USERS_DB.get(username)
    if not user_data or not PasswordManager.verify_password(password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    role = user_data.get("role", "viewer")
    scopes = RBACEnforcer.ROLE_PERMISSIONS.get(role, ["read"])
    token = TokenManager.create_access_token(subject=username, role=role, scopes=scopes)
    return {"access_token": token, "token_type": "bearer", "role": role}

