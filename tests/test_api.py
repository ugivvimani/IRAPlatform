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
        assert "decision" in data
        assert "risk_rating" in data["decision"]
        assert "confidence" in data["decision"]

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
