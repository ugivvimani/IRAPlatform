from __future__ import annotations

from app.settings import AppSettings
from app.storage.base import StorageRepository
from app.storage.postgres_repo import PostgresRepository
from app.storage.sqlite_repo import SQLiteRepository


def build_storage_repository(settings: AppSettings) -> StorageRepository:
    backend = settings.db_backend.lower().strip()
    if backend == "postgres":
        return PostgresRepository(settings.postgres_dsn)
    return SQLiteRepository(settings.sqlite_db_path)
