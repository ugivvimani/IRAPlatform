"""
Authentication and authorization layer.
Supports API key auth, JWT tokens, and role-based access control (RBAC).
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from functools import lru_cache
import json
import base64

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
try:
    import jwt  # type: ignore
    JWT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    jwt = None
    JWT_AVAILABLE = False

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CryptContext = None
    PASSLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") if PASSLIB_AVAILABLE else None


class TokenData(BaseModel):
    """JWT token payload."""
    sub: str  # Subject (usually user ID or service name)
    role: str  # User role (admin, analyst, viewer)
    scopes: list[str] = []  # Permission scopes
    exp: Optional[datetime] = None


class User(BaseModel):
    """User model."""
    user_id: str
    role: str  # admin, analyst, viewer
    api_key: Optional[str] = None
    permissions: list[str] = []


class SecurityConfig:
    """Security configuration."""
    
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY", "change-me-in-production")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        self.api_key_header_name = os.getenv("API_KEY_HEADER", "X-API-Key")
        self.enable_api_key_auth = os.getenv("ENABLE_API_KEY_AUTH", "false").lower() == "true"
        self.enable_jwt_auth = os.getenv("ENABLE_JWT_AUTH", "false").lower() == "true"


config = SecurityConfig()


class TokenManager:
    """Manage JWT token creation and validation."""
    
    @staticmethod
    def create_access_token(
        subject: str,
        role: str = "viewer",
        scopes: Optional[list[str]] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create a JWT access token."""
        if expires_delta is None:
            expires_delta = timedelta(minutes=config.access_token_expire_minutes)
        
        expire = datetime.now(timezone.utc) + expires_delta
        token_data = {
            "sub": subject,
            "role": role,
            "scopes": scopes or [],
            "exp": expire,
        }
        
        if JWT_AVAILABLE:
            encoded_jwt = jwt.encode(
                token_data,
                config.secret_key,
                algorithm=config.algorithm,
            )
            return encoded_jwt

        raw = json.dumps(token_data, default=str).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8")
    
    @staticmethod
    def verify_token(token: str) -> TokenData:
        """Verify and decode JWT token."""
        try:
            if JWT_AVAILABLE:
                payload = jwt.decode(
                    token,
                    config.secret_key,
                    algorithms=[config.algorithm],
                )
            else:
                payload = json.loads(base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8"))
            subject = payload.get("sub")
            if subject is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                )
            exp_val = payload.get("exp")
            if isinstance(exp_val, str):
                exp_dt = datetime.fromisoformat(exp_val)
            else:
                exp_dt = datetime.fromtimestamp(exp_val or 0, tz=timezone.utc)
            token_data = TokenData(
                sub=subject,
                role=payload.get("role", "viewer"),
                scopes=payload.get("scopes", []),
                exp=exp_dt,
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        return token_data


class PasswordManager:
    """Handle password hashing and verification."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password."""
        if PASSLIB_AVAILABLE and pwd_context is not None:
            try:
                return pwd_context.hash(password)
            except Exception:
                return f"plain::{password}"
        return f"plain::{password}"
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if PASSLIB_AVAILABLE and pwd_context is not None:
            try:
                return pwd_context.verify(plain_password, hashed_password)
            except Exception:
                return hashed_password == f"plain::{plain_password}"
        return hashed_password == f"plain::{plain_password}"


# In-memory user store (in production, use database)
USERS_DB: dict[str, dict] = {
    "admin": {
        "user_id": "admin",
        "password_hash": PasswordManager.hash_password("admin-password-change-me"),
        "role": "admin",
        "api_key": os.getenv("ADMIN_API_KEY", ""),
        "permissions": ["read", "write", "delete", "audit"],
    },
    "analyst": {
        "user_id": "analyst",
        "password_hash": PasswordManager.hash_password("analyst-password-change-me"),
        "role": "analyst",
        "api_key": os.getenv("ANALYST_API_KEY", ""),
        "permissions": ["read", "write"],
    },
    "viewer": {
        "user_id": "viewer",
        "password_hash": PasswordManager.hash_password("viewer-password-change-me"),
        "role": "viewer",
        "api_key": os.getenv("VIEWER_API_KEY", ""),
        "permissions": ["read"],
    },
}


class APIKeyAuthenticator:
    """Authenticate using API keys."""
    
    def __init__(self):
        self.header = APIKeyHeader(name=config.api_key_header_name, auto_error=False)
    
    async def __call__(self, api_key: Optional[str] = Depends(APIKeyHeader(name=config.api_key_header_name, auto_error=False))) -> User:
        """Authenticate request using API key."""
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing API key",
            )
        
        # Look up API key in database
        for user_id, user_data in USERS_DB.items():
            if user_data.get("api_key") == api_key:
                return User(
                    user_id=user_data["user_id"],
                    role=user_data["role"],
                    api_key=api_key,
                    permissions=user_data.get("permissions", []),
                )
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


class JWTAuthenticator:
    """Authenticate using JWT tokens."""
    
    def __init__(self):
        self.bearer = HTTPBearer(auto_error=False)
    
    async def __call__(self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> User:
        """Authenticate request using JWT token."""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing credentials",
            )
        
        token_data = TokenManager.verify_token(credentials.credentials)
        return User(
            user_id=token_data.sub,
            role=token_data.role,
            permissions=RBACEnforcer.ROLE_PERMISSIONS.get(token_data.role, ["read"]),
        )


class HybridAuthenticator:
    """Try both API key and JWT auth."""
    
    def __init__(self):
        self.api_key_auth = APIKeyAuthenticator()
        self.jwt_auth = JWTAuthenticator()
    
    async def __call__(
        self,
        api_key: Optional[str] = Depends(APIKeyHeader(name=config.api_key_header_name, auto_error=False)),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    ) -> User:
        """Try API key first, then JWT."""
        if not config.enable_api_key_auth and not config.enable_jwt_auth:
            return User(user_id="anonymous", role="admin", permissions=["read", "write", "delete", "audit", "manage_users"])

        if api_key:
            try:
                return await self.api_key_auth(api_key=api_key)
            except HTTPException:
                pass
        
        if credentials:
            try:
                return await self.jwt_auth(credentials=credentials)
            except HTTPException:
                pass
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing credentials",
        )


class RBACEnforcer:
    """Role-based access control."""
    
    ROLE_PERMISSIONS = {
        "admin": ["read", "write", "delete", "audit", "manage_users"],
        "analyst": ["read", "write"],
        "viewer": ["read"],
    }
    
    @staticmethod
    def check_permission(user: User, required_permission: str) -> bool:
        """Check if user has required permission."""
        if not config.enable_api_key_auth and not config.enable_jwt_auth:
            return True
        role_perms = RBACEnforcer.ROLE_PERMISSIONS.get(user.role, [])
        return required_permission in role_perms or required_permission in user.permissions
    
    @staticmethod
    def require_permission(required_permission: str):
        """Dependency for enforcing permission."""
        async def permission_checker(user: User = Depends(HybridAuthenticator())) -> User:
            if not RBACEnforcer.check_permission(user, required_permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User {user.user_id} lacks permission: {required_permission}",
                )
            return user
        return permission_checker


def get_authenticated_user(user: User = Depends(HybridAuthenticator())) -> User:
    """Get current authenticated user."""
    return user


def require_admin(user: User = Depends(RBACEnforcer.require_permission("manage_users"))) -> User:
    """Require admin role."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return user


def require_write_access(user: User = Depends(RBACEnforcer.require_permission("write"))) -> User:
    """Require write access."""
    return user


# Helper function for the require_permission dependency
def require_permission(permission: str):
    """Create a dependency for checking specific permission."""
    return RBACEnforcer.require_permission(permission)



