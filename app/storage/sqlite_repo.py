from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.contracts import AssessmentAuditRecord, AssessmentResponse, WatchlistEntry
from app.storage.base import StorageRepository


class SQLiteRepository(StorageRepository):
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist_entries (
                    entity_id TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    added_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    question TEXT NOT NULL,
                    risk_rating TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    requires_manual_review INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def upsert_watchlist(self, entry: WatchlistEntry) -> WatchlistEntry:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO watchlist_entries (entity_id, company_name, notes, added_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    company_name=excluded.company_name,
                    notes=excluded.notes
                """,
                (
                    entry.entity_id,
                    entry.company_name,
                    entry.notes,
                    entry.added_at.astimezone(timezone.utc).isoformat(),
                ),
            )
        return entry

    def get_watchlist(self, entity_id: str) -> WatchlistEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT entity_id, company_name, notes, added_at FROM watchlist_entries WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
        if not row:
            return None
        return WatchlistEntry(
            entity_id=row["entity_id"],
            company_name=row["company_name"],
            notes=row["notes"],
            added_at=datetime.fromisoformat(row["added_at"]),
        )

    def list_watchlist(self) -> list[WatchlistEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT entity_id, company_name, notes, added_at FROM watchlist_entries ORDER BY added_at DESC"
            ).fetchall()
        return [
            WatchlistEntry(
                entity_id=row["entity_id"],
                company_name=row["company_name"],
                notes=row["notes"],
                added_at=datetime.fromisoformat(row["added_at"]),
            )
            for row in rows
        ]
    
    def delete_watchlist(self, entity_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM watchlist_entries WHERE entity_id = ?",
                (entity_id,),
            )
            return cur.rowcount > 0

    def insert_assessment(self, response: AssessmentResponse) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO assessments (
                    entity_id, company_name, question, risk_rating, confidence, requires_manual_review, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.query.company_name,
                    response.query.company_name,
                    response.query.question,
                    response.decision.risk_rating.value,
                    response.decision.confidence.value,
                    1 if response.decision.requires_manual_review else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def list_assessments(self, entity_id: str, limit: int = 25) -> list[AssessmentAuditRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT assessment_id, entity_id, company_name, question, risk_rating, confidence, requires_manual_review, created_at
                FROM assessments
                WHERE entity_id = ?
                ORDER BY assessment_id DESC
                LIMIT ?
                """,
                (entity_id, limit),
            ).fetchall()
        return [
            AssessmentAuditRecord(
                assessment_id=row["assessment_id"],
                entity_id=row["entity_id"],
                company_name=row["company_name"],
                question=row["question"],
                risk_rating=row["risk_rating"],
                confidence=row["confidence"],
                requires_manual_review=bool(row["requires_manual_review"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]
