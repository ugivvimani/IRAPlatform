from __future__ import annotations

from abc import ABC, abstractmethod

from app.contracts import (
    AssessmentAuditRecord,
    AssessmentResponse,
    PolicyThresholdRecord,
    PolicyThresholdUpsert,
    WatchlistEntry,
)


class StorageRepository(ABC):
    @abstractmethod
    def upsert_watchlist(self, entry: WatchlistEntry) -> WatchlistEntry:
        raise NotImplementedError

    @abstractmethod
    def get_watchlist(self, entity_id: str) -> WatchlistEntry | None:
        raise NotImplementedError

    @abstractmethod
    def list_watchlist(self) -> list[WatchlistEntry]:
        raise NotImplementedError

    @abstractmethod
    def delete_watchlist(self, entity_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def insert_assessment(self, response: AssessmentResponse) -> int:
        raise NotImplementedError

    @abstractmethod
    def list_assessments(self, entity_id: str, limit: int = 25) -> list[AssessmentAuditRecord]:
        raise NotImplementedError

    @abstractmethod
    def upsert_policy_threshold(self, policy_key: str, payload: PolicyThresholdUpsert) -> PolicyThresholdRecord:
        raise NotImplementedError

    @abstractmethod
    def get_active_policy_thresholds(self) -> dict[str, PolicyThresholdRecord]:
        raise NotImplementedError
