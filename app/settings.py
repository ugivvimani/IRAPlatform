from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppSettings:
    app_env: str
    app_host: str
    app_port: int
    vector_backend: str
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str


def load_settings() -> AppSettings:
    return AppSettings(
        app_env=os.getenv("APP_ENV", "local"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        vector_backend="pinecone",
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index=os.getenv("PINECONE_INDEX", "ira-platform-memory"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
    )
