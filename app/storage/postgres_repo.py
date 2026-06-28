from __future__ import annotations

from datetime import datetime, timezone

from app.contracts import (
    AssessmentAuditRecord,
    AssessmentResponse,
    PolicyThresholdRecord,
    PolicyThresholdUpsert,
    WatchlistEntry,
)
from app.storage.base import StorageRepository


class PostgresRepository(StorageRepository):
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._init_schema()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS watchlist_entries (
                        entity_id TEXT PRIMARY KEY,
                        company_name TEXT NOT NULL,
                        notes TEXT NOT NULL DEFAULT '',
                        added_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assessments (
                        assessment_id BIGSERIAL PRIMARY KEY,
                        entity_id TEXT NOT NULL,
                        company_name TEXT NOT NULL,
                        question TEXT NOT NULL,
                        risk_rating TEXT NOT NULL,
                        confidence TEXT NOT NULL,
                        requires_manual_review BOOLEAN NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS policy_thresholds (
                        policy_key TEXT NOT NULL,
                        threshold_value DOUBLE PRECISION NOT NULL,
                        version INTEGER NOT NULL,
                        approved_by TEXT NOT NULL,
                        approval_notes TEXT NOT NULL DEFAULT '',
                        is_active BOOLEAN NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY(policy_key, version)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_policy_thresholds_active
                    ON policy_thresholds (policy_key, is_active)
                    """
                )
            conn.commit()

    def upsert_watchlist(self, entry: WatchlistEntry) -> WatchlistEntry:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO watchlist_entries (entity_id, company_name, notes, added_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        notes = EXCLUDED.notes
                    """,
                    (
                        entry.entity_id,
                        entry.company_name,
                        entry.notes,
                        entry.added_at.astimezone(timezone.utc),
                    ),
                )
            conn.commit()
        return entry

    def get_watchlist(self, entity_id: str) -> WatchlistEntry | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT entity_id, company_name, notes, added_at FROM watchlist_entries WHERE entity_id = %s",
                    (entity_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return WatchlistEntry(entity_id=row[0], company_name=row[1], notes=row[2], added_at=row[3])

    def list_watchlist(self) -> list[WatchlistEntry]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT entity_id, company_name, notes, added_at FROM watchlist_entries ORDER BY added_at DESC"
                )
                rows = cur.fetchall()
        return [WatchlistEntry(entity_id=r[0], company_name=r[1], notes=r[2], added_at=r[3]) for r in rows]

    def delete_watchlist(self, entity_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM watchlist_entries WHERE entity_id = %s",
                    (entity_id,),
                )
                rowcount = cur.rowcount
            conn.commit()
        return rowcount > 0

    def insert_assessment(self, response: AssessmentResponse) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO assessments (
                        entity_id, company_name, question, risk_rating, confidence, requires_manual_review, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING assessment_id
                    """,
                    (
                        response.query.company_name,
                        response.query.company_name,
                        response.query.question,
                        response.decision.risk_rating.value,
                        response.decision.confidence.value,
                        response.decision.requires_manual_review,
                        datetime.now(timezone.utc),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row[0])

    def list_assessments(self, entity_id: str, limit: int = 25) -> list[AssessmentAuditRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT assessment_id, entity_id, company_name, question, risk_rating, confidence, requires_manual_review, created_at
                    FROM assessments
                    WHERE entity_id = %s
                    ORDER BY assessment_id DESC
                    LIMIT %s
                    """,
                    (entity_id, limit),
                )
                rows = cur.fetchall()
        return [
            AssessmentAuditRecord(
                assessment_id=row[0],
                entity_id=row[1],
                company_name=row[2],
                question=row[3],
                risk_rating=row[4],
                confidence=row[5],
                requires_manual_review=bool(row[6]),
                created_at=row[7],
            )
            for row in rows
        ]

    def upsert_policy_threshold(self, policy_key: str, payload: PolicyThresholdUpsert) -> PolicyThresholdRecord:
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM policy_thresholds WHERE policy_key = %s",
                    (policy_key,),
                )
                max_version = int(cur.fetchone()[0])
                next_version = max_version + 1

                cur.execute(
                    "UPDATE policy_thresholds SET is_active = false, updated_at = %s WHERE policy_key = %s AND is_active = true",
                    (now, policy_key),
                )

                cur.execute(
                    """
                    INSERT INTO policy_thresholds (
                        policy_key, threshold_value, version, approved_by, approval_notes, is_active, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, true, %s, %s)
                    RETURNING policy_key, threshold_value, version, approved_by, approval_notes, is_active, created_at, updated_at
                    """,
                    (
                        policy_key,
                        float(payload.threshold_value),
                        next_version,
                        payload.approved_by,
                        payload.approval_notes,
                        now,
                        now,
                    ),
                )
                row = cur.fetchone()
            conn.commit()

        return PolicyThresholdRecord(
            policy_key=row[0],
            threshold_value=float(row[1]),
            version=int(row[2]),
            approved_by=row[3],
            approval_notes=row[4],
            is_active=bool(row[5]),
            created_at=row[6],
            updated_at=row[7],
        )

    def get_active_policy_thresholds(self) -> dict[str, PolicyThresholdRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT policy_key, threshold_value, version, approved_by, approval_notes, is_active, created_at, updated_at
                    FROM policy_thresholds
                    WHERE is_active = true
                    """
                )
                rows = cur.fetchall()

        return {
            row[0]: PolicyThresholdRecord(
                policy_key=row[0],
                threshold_value=float(row[1]),
                version=int(row[2]),
                approved_by=row[3],
                approval_notes=row[4],
                is_active=bool(row[5]),
                created_at=row[6],
                updated_at=row[7],
            )
            for row in rows
        }
