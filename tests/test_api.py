from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_shows_llm_backend(self, client):
        r = client.get("/health")
        assert "llm_backend" in r.json()


class TestAssessEndpoint:
    def test_assess_returns_200(self, client):
        r = client.post("/assess", json={
            "query": {"company_name": "TestCo", "question": "Is TestCo safe?"}
        })
        assert r.status_code == 200

    def test_assess_response_has_decision(self, client):
        r = client.post("/assess", json={
            "query": {"company_name": "TestCo", "question": "Is TestCo safe?"}
        })
        data = r.json()
        assert "assessment_id" in data
        assert "company_name" in data
        assert "risk_rating" in data
        assert "confidence" in data
        assert "summary" in data
        assert "requires_manual_review" in data
        assert "evaluated_at" in data

    def test_assess_with_include_details_returns_full_payload(self, client):
        r = client.post("/assess?include_details=true", json={
            "query": {"company_name": "DetailedCo", "question": "Is DetailedCo safe?"}
        })
        data = r.json()
        assert "query" in data
        assert "decision" in data
        assert "evidence_chain" in data
        assert "model_metadata" in data

    def test_assess_with_seeded_evidence(self, client):
        r = client.post("/assess", json={
            "query": {"company_name": "SeededCo", "question": "safe?"},
            "evidence": [{
                "evidence_id": "e1",
                "dimension": "sanctions",
                "signal": "sanctions_status",
                "value": "not_sanctioned",
                "source_name": "OFAC",
                "source_tier": "official",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "entity_match_confidence": 0.95,
                "source_confidence": 0.95,
                "provenance_url": "https://ofac.treasury.gov/",
                "metadata": {},
            }],
        })
        assert r.status_code == 200

    def test_assess_rejects_invalid_provenance(self, client):
        r = client.post("/assess", json={
            "query": {"company_name": "SeededCo", "question": "safe?"},
            "evidence": [{
                "evidence_id": "e-bad",
                "dimension": "sanctions",
                "signal": "sanctions_status",
                "value": "not_sanctioned",
                "source_name": "OFAC",
                "source_tier": "official",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "entity_match_confidence": 0.95,
                "source_confidence": 0.95,
                "provenance_url": "ftp://invalid.example/proof",
                "metadata": {},
            }],
        })
        assert r.status_code == 422

    def test_assessment_history_endpoint(self, client):
        client.post("/assess", json={
            "query": {"company_name": "HistoryCo", "question": "Is HistoryCo safe?"}
        })
        r = client.get("/assessments/HistoryCo")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["entity_id"] == "HistoryCo"


class TestWatchlistEndpoints:
    def test_add_to_watchlist(self, client):
        r = client.post("/watchlist", json={
            "entity_id": "wl-test-001",
            "company_name": "WatchedCo",
            "notes": "Flagged for review",
        })
        assert r.status_code == 201
        assert r.json()["entity_id"] == "wl-test-001"

    def test_get_watchlist_status(self, client):
        client.post("/watchlist", json={
            "entity_id": "wl-test-002",
            "company_name": "MonitoredCo",
            "notes": "",
        })
        r = client.get("/watchlist/wl-test-002")
        assert r.status_code == 200
        assert r.json()["company_name"] == "MonitoredCo"
        assert "current_risk_rating" in r.json()

    def test_get_watchlist_unknown_entity_404(self, client):
        r = client.get("/watchlist/unknown-entity-xyz")
        assert r.status_code == 404

    def test_list_watchlist(self, client):
        r = client.get("/watchlist")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestAsyncAssessEndpoint:
    def test_assess_async_returns_task(self, client):
        r = client.post("/assess/async", json={
            "company_name": "AsyncCo",
            "question": "Is AsyncCo safe?",
        })
        assert r.status_code == 202
        data = r.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    def test_task_status_endpoint(self, client):
        queued = client.post("/assess/async", json={
            "company_name": "AsyncStatusCo",
            "question": "Status check",
        })
        task_id = queued.json()["task_id"]

        r = client.get(f"/tasks/{task_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"] == task_id
        assert "status" in data


class TestPolicyRegistryEndpoints:
    def test_upsert_policy_threshold(self, client):
        r = client.put("/policies/auto_hold_threshold", json={
            "threshold_value": 0.75,
            "approved_by": "admin",
            "approval_notes": "Launch conservative threshold",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["policy_key"] == "auto_hold_threshold"
        assert data["threshold_value"] == 0.75
        assert data["version"] >= 1

    def test_get_active_policies(self, client):
        client.put("/policies/entity_merge_threshold", json={
            "threshold_value": 0.75,
            "approved_by": "admin",
            "approval_notes": "Entity merge review threshold",
        })
        r = client.get("/policies/active")
        assert r.status_code == 200
        data = r.json()
        assert "entity_merge_threshold" in data
        assert data["entity_merge_threshold"]["is_active"] is True

