"""
Unit tests for SEC EDGAR Financial connector and the financial signal taxonomy.
These tests run fully offline using mocked HTTP responses — no real API calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.analysis_forecasting import AnalysisForecastingAgent, _signal_base_risk
from app.contracts import EvidenceItem, RiskDimension, SourceTier
from app.services.connectors import ConnectorConfig, SECFinancialsConnector


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_response(status: int, body: dict) -> MagicMock:
    """Build a mock httpx.Response."""
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = body
    return mock


def _ticker_payload(entries: list[tuple[str, str, str]]) -> dict:
    """Build a mock company_tickers.json payload."""
    return {
        str(i): {"cik_str": cik, "ticker": ticker, "title": title}
        for i, (cik, ticker, title) in enumerate(entries)
    }


def _facts_payload(
    long_term_debt: int | None = None,
    equity: int | None = None,
    net_income: int | None = None,
    revenue: int | None = None,
) -> dict:
    """Build a minimal mock companyfacts JSON."""
    def _entry(val: int) -> list[dict]:
        return [{"val": val, "end": "2024-12-31", "form": "10-K", "filed": "2025-02-01"}]

    usgaap: dict = {}
    if long_term_debt is not None:
        usgaap["LongTermDebt"] = {"units": {"USD": _entry(long_term_debt)}}
    if equity is not None:
        usgaap["StockholdersEquity"] = {"units": {"USD": _entry(equity)}}
    if net_income is not None:
        usgaap["NetIncomeLoss"] = {"units": {"USD": _entry(net_income)}}
    if revenue is not None:
        usgaap["RevenueFromContractWithCustomerExcludingAssessedTax"] = {"units": {"USD": _entry(revenue)}}

    return {"entityName": "Test Corp", "facts": {"us-gaap": usgaap}}


def _submissions_payload(forms: list[str]) -> dict:
    return {"filings": {"recent": {"form": forms}}}


# ── signal taxonomy tests ─────────────────────────────────────────────────────

class TestFinancialSignalTaxonomy:
    def test_debt_to_equity_negative_equity_is_critical(self):
        risk = _signal_base_risk("debt_to_equity_ratio", "-0.5")
        assert risk == 1.0

    def test_debt_to_equity_very_high_is_high_risk(self):
        risk = _signal_base_risk("debt_to_equity_ratio", "6.0")
        assert risk >= 0.80

    def test_debt_to_equity_moderate(self):
        risk = _signal_base_risk("debt_to_equity_ratio", "2.0")
        assert 0.25 <= risk <= 0.50

    def test_debt_to_equity_low_is_mild_positive(self):
        risk = _signal_base_risk("debt_to_equity_ratio", "0.3")
        assert risk < 0  # safe signal

    def test_profit_margin_severe_loss(self):
        risk = _signal_base_risk("profit_margin", "-0.25")
        assert risk >= 0.70

    def test_profit_margin_unprofitable(self):
        risk = _signal_base_risk("profit_margin", "-0.05")
        assert 0.30 <= risk <= 0.60

    def test_profit_margin_healthy(self):
        risk = _signal_base_risk("profit_margin", "0.20")
        assert risk <= 0.0  # healthy margin is a safe signal

    def test_net_income_negative_is_risky(self):
        risk = _signal_base_risk("net_income_trend", "negative")
        assert risk >= 0.40

    def test_net_income_positive_is_safe(self):
        risk = _signal_base_risk("net_income_trend", "positive")
        assert risk < 0

    def test_late_filing_is_high_risk(self):
        risk = _signal_base_risk("late_filing_notice", "yes")
        assert risk >= 0.65

    def test_not_sec_registered_is_near_neutral(self):
        risk = _signal_base_risk("sec_registered", "not_sec_registered")
        assert 0.0 <= risk <= 0.10


# ── connector integration tests (mocked HTTP) ─────────────────────────────────

@pytest.fixture()
def config() -> ConnectorConfig:
    cfg = ConnectorConfig()
    cfg.sec_email = "test@example.com"
    return cfg


class TestSECFinancialsConnector:
    """Tests for SECFinancialsConnector using fully mocked HTTP responses."""

    def _reset_cache(self):
        SECFinancialsConnector._cik_cache = {}
        SECFinancialsConnector._cik_cache_loaded = False

    @pytest.mark.asyncio
    async def test_unknown_company_returns_not_registered_item(self, config):
        self._reset_cache()
        connector = SECFinancialsConnector(config)
        connector.session = MagicMock()

        tickers_resp = _make_response(200, _ticker_payload([("320193", "AAPL", "Apple Inc.")]))
        connector._request_with_retry = AsyncMock(return_value=tickers_resp)

        evidence = await connector.fetch("Totally Unknown Private Corp XYZ")
        assert len(evidence) == 1
        assert evidence[0].signal == "sec_registered"
        assert evidence[0].value == "not_sec_registered"
        assert evidence[0].dimension == RiskDimension.FINANCIAL

    @pytest.mark.asyncio
    async def test_healthy_company_produces_financial_signals(self, config):
        self._reset_cache()
        connector = SECFinancialsConnector(config)
        connector.session = MagicMock()

        tickers_resp = _make_response(200, _ticker_payload([("320193", "AAPL", "Apple Inc.")]))
        facts_resp = _make_response(200, _facts_payload(
            long_term_debt=80_000_000_000,
            equity=60_000_000_000,
            net_income=90_000_000_000,
            revenue=390_000_000_000,
        ))
        subs_resp = _make_response(200, _submissions_payload(["10-K", "10-Q", "8-K"]))

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tickers_resp
            if call_count == 2:
                return facts_resp
            return subs_resp

        connector._request_with_retry = side_effect
        evidence = await connector.fetch("Apple Inc")

        signals = {e.signal for e in evidence}
        assert "debt_to_equity_ratio" in signals
        assert "profit_margin" in signals
        assert "net_income_trend" in signals
        # No late filing for a healthy company
        assert "late_filing_notice" not in signals

    @pytest.mark.asyncio
    async def test_late_filing_notice_detected(self, config):
        self._reset_cache()
        connector = SECFinancialsConnector(config)
        connector.session = MagicMock()

        tickers_resp = _make_response(200, _ticker_payload([("886158", "BBBY", "Bed Bath Beyond Inc")]))
        facts_resp = _make_response(200, _facts_payload(net_income=-500_000_000))
        subs_resp = _make_response(200, _submissions_payload(["10-K", "NT 10-K", "NT 10-Q", "8-K"]))

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tickers_resp
            if call_count == 2:
                return facts_resp
            return subs_resp

        connector._request_with_retry = side_effect
        evidence = await connector.fetch("Bed Bath Beyond Inc")

        late = [e for e in evidence if e.signal == "late_filing_notice"]
        assert len(late) == 1
        assert late[0].value == "yes"

    @pytest.mark.asyncio
    async def test_negative_net_income_produces_negative_trend(self, config):
        self._reset_cache()
        connector = SECFinancialsConnector(config)
        connector.session = MagicMock()

        tickers_resp = _make_response(200, _ticker_payload([("12345", "LOSS", "Loss Corp Inc")]))
        facts_resp = _make_response(200, _facts_payload(net_income=-1_000_000_000))
        subs_resp = _make_response(200, _submissions_payload(["10-K"]))

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tickers_resp
            if call_count == 2:
                return facts_resp
            return subs_resp

        connector._request_with_retry = side_effect
        evidence = await connector.fetch("Loss Corp Inc")

        income_ev = next((e for e in evidence if e.signal == "net_income_trend"), None)
        assert income_ev is not None
        assert income_ev.value == "negative"


# ── scoring integration: financial signals flow through analysis agent ─────────

class TestFinancialSignalScoring:
    def _ev(self, signal: str, value: str) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=f"test-{signal}",
            dimension=RiskDimension.FINANCIAL,
            signal=signal,
            value=value,
            source_name="SEC EDGAR",
            source_tier=SourceTier.REGULATOR,
            timestamp=datetime.now(timezone.utc),
            entity_match_confidence=0.95,
            source_confidence=0.93,
            provenance_url="https://data.sec.gov/",
            metadata={},
        )

    def test_highly_leveraged_company_scores_above_low_leverage(self):
        agent = AnalysisForecastingAgent()
        high_risk = [self._ev("debt_to_equity_ratio", "6.0")]
        low_risk   = [self._ev("debt_to_equity_ratio", "0.3")]
        assert agent.score(high_risk)["financial_risk"] > agent.score(low_risk)["financial_risk"]

    def test_late_filing_raises_financial_risk(self):
        agent = AnalysisForecastingAgent()
        with_late  = [self._ev("late_filing_notice", "yes")]
        without_late = [self._ev("net_income_trend", "positive")]
        assert agent.score(with_late)["financial_risk"] > agent.score(without_late)["financial_risk"]

    def test_profitable_company_has_lower_financial_risk_than_unprofitable(self):
        agent = AnalysisForecastingAgent()
        profitable   = [self._ev("profit_margin", "0.20"), self._ev("net_income_trend", "positive")]
        unprofitable = [self._ev("profit_margin", "-0.15"), self._ev("net_income_trend", "negative")]
        assert agent.score(profitable)["financial_risk"] < agent.score(unprofitable)["financial_risk"]

    def test_financial_risk_contributes_to_composite(self):
        agent = AnalysisForecastingAgent()
        distress = [
            self._ev("debt_to_equity_ratio", "7.0"),
            self._ev("profit_margin", "-0.30"),
            self._ev("late_filing_notice", "yes"),
            self._ev("net_income_trend", "negative"),
        ]
        result = agent.score(distress)
        assert result["financial_risk"] > 0.5
        assert result["composite_quant_score"] > 0.0
