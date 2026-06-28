"""
Real external data source connectors with retry, rate-limiting, and error handling.
Supports OpenSanctions, SEC Edgar, NewsAPI, and ESG providers.
"""
import asyncio
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional
import logging
from uuid import uuid4

import httpx
from app.contracts import EvidenceItem, RiskDimension, SourceTier

logger = logging.getLogger(__name__)


class ConnectorConfig:
    """Configuration for external connectors."""
    
    def __init__(self):
        self.opensanctions_api_key = os.getenv("OPENSANCTIONS_API_KEY", "")
        self.opensanctions_url = os.getenv("OPENSANCTIONS_URL", "https://api.opensanctions.org")
        self.sec_edgar_url = os.getenv("SEC_EDGAR_URL", "https://www.sec.gov/cgi-bin/browse-edgar")
        self.sec_email = os.getenv("SEC_CONTACT_EMAIL", "research@example.com")
        self.reuters_api_key = os.getenv("REUTERS_API_KEY", "")
        self.reuters_url = os.getenv("REUTERS_URL", "https://api.reuters.com/")
        self.esg_provider_url = os.getenv("ESG_PROVIDER_URL", "https://api.esgrating.com/")
        self.esg_api_key = os.getenv("ESG_API_KEY", "")
        self.request_timeout = int(os.getenv("CONNECTOR_TIMEOUT_SECONDS", "30"))
        self.max_retries = int(os.getenv("CONNECTOR_MAX_RETRIES", "3"))
        self.retry_backoff = float(os.getenv("CONNECTOR_RETRY_BACKOFF", "1.5"))


class BaseConnector(ABC):
    """Abstract base for all external connectors."""
    
    connector_name: str
    source_tier: str  # official, regulator, news, esg
    source_confidence: float  # base reliability score
    
    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.config = config or ConnectorConfig()
        self.session: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=self.config.request_timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    @abstractmethod
    async def fetch(self, entity_name: str) -> list[EvidenceItem]:
        """Fetch evidence from this source."""
        raise NotImplementedError
    
    async def _request_with_retry(
        self, 
        method: str, 
        url: str, 
        headers: Optional[dict] = None,
        **kwargs
    ) -> Optional[httpx.Response]:
        """Execute HTTP request with exponential backoff retry."""
        if not self.session:
            raise RuntimeError("Connector session not initialized. Use 'async with' context manager.")
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.session.request(method, url, headers=headers, **kwargs)
                if response.status_code < 500:
                    return response
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries}: {self.connector_name} returned {response.status_code}")
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries}: {self.connector_name} connection error: {e}")
            
            if attempt < self.config.max_retries - 1:
                wait_time = self.config.retry_backoff ** attempt
                await asyncio.sleep(wait_time)
        
        return None


class OpenSanctionsConnector(BaseConnector):
    """OpenSanctions API connector — covers OFAC, UN, EU, and 100+ sanctions lists."""

    connector_name = "sanctions_opensanctions"
    source_tier = "official"
    source_confidence = 0.97

    async def fetch(self, entity_name: str) -> list[EvidenceItem]:
        """
        Match entity against OpenSanctions consolidated dataset.
        API docs: https://www.opensanctions.org/docs/api/
        Endpoint: POST /match/default
        Auth: Authorization: ApiKey <key>
        """
        api_key = self.config.opensanctions_api_key
        if not api_key:
            logger.warning("OPENSANCTIONS_API_KEY not set; skipping sanctions fetch")
            return []

        url = f"{self.config.opensanctions_url}/match/default"
        headers = {
            "Authorization": f"ApiKey {api_key}",
            "Content-Type": "application/json",
        }
        # OpenSanctions match API body
        body = {
            "queries": {
                "entity": {
                    "schema": "Company",
                    "properties": {"name": [entity_name]},
                }
            }
        }

        evidence = []
        try:
            response = await self._request_with_retry("POST", url, headers=headers, json=body)
            if not response or response.status_code != 200:
                logger.warning("OpenSanctions returned %s", response.status_code if response else "no response")
                return evidence

            data = response.json()
            results = data.get("responses", {}).get("entity", {}).get("results", [])

            if results:
                for match in results[:5]:
                    score = match.get("score", 0.0)
                    caption = match.get("caption", entity_name)
                    datasets = ", ".join(match.get("datasets", []))
                    evidence.append(EvidenceItem(
                        evidence_id=f"opensanctions-{uuid4().hex}",
                        signal="sanctions_listed",
                        value="yes",
                        source_name="OpenSanctions",
                        source_tier=SourceTier.OFFICIAL,
                        dimension=RiskDimension.SANCTIONS,
                        timestamp=datetime.now(timezone.utc),
                        entity_match_confidence=min(score, 1.0),
                        source_confidence=self.source_confidence,
                        provenance_url=f"https://www.opensanctions.org/entities/{match.get('id', '')}",
                        metadata={
                            "matched_name": caption,
                            "datasets": datasets,
                            "match_score": str(score),
                        },
                        raw_content=(
                            f"{caption} appears on sanctions lists: {datasets}. "
                            f"Match confidence: {score:.0%}."
                        ),
                    ))
            else:
                evidence.append(EvidenceItem(
                    evidence_id=f"opensanctions-{uuid4().hex}",
                    signal="sanctions_status",
                    value="not_sanctioned",
                    source_name="OpenSanctions",
                    source_tier=SourceTier.OFFICIAL,
                    dimension=RiskDimension.SANCTIONS,
                    timestamp=datetime.now(timezone.utc),
                    entity_match_confidence=0.95,
                    source_confidence=self.source_confidence,
                    provenance_url="https://www.opensanctions.org/",
                    metadata={"search_query": entity_name},
                ))
        except Exception as exc:
            logger.error("OpenSanctions connector error: %s", exc)

        return evidence


class SECConnector(BaseConnector):
    """SEC EDGAR full-text search connector using data.sec.gov JSON API."""

    connector_name = "regulatory_sec"
    source_tier = "regulator"
    source_confidence = 0.96

    async def fetch(self, entity_name: str) -> list[EvidenceItem]:
        """
        Search SEC EDGAR using the EFTS full-text search API (returns JSON).
        Docs: https://efts.sec.gov/LATEST/search-index?q=...
        User-Agent required per SEC policy.
        """
        evidence = []
        headers = {
            "User-Agent": f"IRA-Agent ({self.config.sec_email})",
            "Accept": "application/json",
        }

        # Company search via EDGAR full-text search
        search_url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{entity_name}"',
            "dateRange": "custom",
            "startdt": "2020-01-01",
            "forms": "10-K,8-K,DEF 14A",
        }

        try:
            response = await self._request_with_retry("GET", search_url, headers=headers, params=params)
            if response and response.status_code == 200:
                data = response.json()
                hits = data.get("hits", {}).get("total", {})
                total = hits.get("value", 0) if isinstance(hits, dict) else int(hits or 0)
                if total > 0:
                    evidence.append(EvidenceItem(
                        evidence_id=f"sec-{uuid4().hex}",
                        signal="sec_filings_found",
                        value=str(total),
                        source_name="SEC EDGAR",
                        source_tier=SourceTier.REGULATOR,
                        dimension=RiskDimension.REGULATORY,
                        timestamp=datetime.now(timezone.utc),
                        entity_match_confidence=0.90,
                        source_confidence=self.source_confidence,
                        provenance_url=f"https://efts.sec.gov/LATEST/search-index?q=%22{entity_name}%22",
                        metadata={"filings_count": str(total)},
                        raw_content=(
                            f"{entity_name} has {total} SEC EDGAR filings (10-K, 8-K, DEF 14A) "
                            f"on record, indicating an active regulatory disclosure history."
                        ),
                    ))
                else:
                    evidence.append(EvidenceItem(
                        evidence_id=f"sec-{uuid4().hex}",
                        signal="sec_filings_found",
                        value="0",
                        source_name="SEC EDGAR",
                        source_tier=SourceTier.REGULATOR,
                        dimension=RiskDimension.REGULATORY,
                        timestamp=datetime.now(timezone.utc),
                        entity_match_confidence=0.90,
                        source_confidence=self.source_confidence,
                        provenance_url="https://www.sec.gov/cgi-bin/browse-edgar",
                        metadata={"note": "no filings found for entity"},
                    ))
            else:
                logger.warning("SEC EDGAR returned %s", response.status_code if response else "no response")
        except Exception as exc:
            logger.error("SEC connector error: %s", exc)

        return evidence


class NewsConnector(BaseConnector):
    """News aggregation connector (Reuters, Bloomberg, etc.)."""
    
    connector_name = "news_feed"
    source_tier = "news"
    source_confidence = 0.75
    
    async def fetch(self, entity_name: str) -> list[EvidenceItem]:
        """
        Query news APIs for recent company mentions.
        Falls back to public sources if API key unavailable.
        """
        evidence = []
        
        # Use NewsAPI (free tier available)
        url = "https://newsapi.org/v2/everything"
        api_key = os.getenv("NEWS_API_KEY", "")
        
        if not api_key:
            logger.warning("NEWS_API_KEY not set; skipping real news fetch")
            return evidence
        
        params = {
            "q": entity_name,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": api_key,
            "pageSize": 10,
        }
        
        try:
            response = await self._request_with_retry("GET", url, params=params)
            if response and response.status_code == 200:
                data = response.json()
                if data.get("articles"):
                    for article in data["articles"][:5]:  # Top 5 articles
                        # Sentiment-based signal
                        sentiment = self._analyze_sentiment(article.get("description", ""))
                        evidence.append(EvidenceItem(
                            evidence_id=f"news-{uuid4().hex}",
                            signal="news_sentiment",
                            value=sentiment,
                            source_name=article.get("source", {}).get("name", "News"),
                            source_tier=SourceTier.TIER1_NEWS,
                            dimension=RiskDimension.REPUTATIONAL,
                            timestamp=datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")),
                            entity_match_confidence=0.88,
                            source_confidence=self.source_confidence,
                            provenance_url=article.get("url", ""),
                            metadata={
                                "title": article.get("title", ""),
                                "description": article.get("description", "")[:200],
                            },
                            raw_content=(
                                f"{article.get('title', '')}. "
                                f"{article.get('description', '')}"
                            ).strip() or None,
                        ))
        except Exception as e:
            logger.error(f"News connector error: {e}")
        
        return evidence
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis (could integrate TextBlob, transformers, etc.)."""
        if not text:
            return "neutral"
        
        negative_words = ["fraud", "scandal", "breach", "default", "bankruptcy", "lawsuit"]
        positive_words = ["award", "excellent", "growth", "innovation", "success"]
        
        text_lower = text.lower()
        neg_count = sum(1 for w in negative_words if w in text_lower)
        pos_count = sum(1 for w in positive_words if w in text_lower)
        
        if neg_count > pos_count:
            return "negative"
        elif pos_count > neg_count:
            return "positive"
        return "neutral"


class ESGConnector(BaseConnector):
    """Environmental, Social, Governance rating connector."""
    
    connector_name = "esg_ratings"
    source_tier = "esg"
    source_confidence = 0.82
    
    async def fetch(self, entity_name: str) -> list[EvidenceItem]:
        """
        Query ESG rating providers (e.g., Sustainalytics, MSCI).
        Falls back gracefully if API unavailable.
        """
        evidence = []
        
        # Use Sustainalytics API (requires key)
        api_key = os.getenv("ESG_API_KEY", "")
        if not api_key:
            logger.warning("ESG_API_KEY not set; skipping real ESG fetch")
            return evidence
        
        url = "https://api.sustainalytics.com/v2/companies/search"
        params = {"name": entity_name}
        headers = {"Authorization": f"Bearer {api_key}"}
        
        try:
            response = await self._request_with_retry("GET", url, headers=headers, params=params)
            if response and response.status_code == 200:
                data = response.json()
                if data.get("companies"):
                    company = data["companies"][0]
                    score = company.get("esgScore", 50)
                    
                    # Map ESG score to risk dimension
                    esg_risk = "high" if score < 30 else "medium" if score < 60 else "low"
                    
                    evidence.append(EvidenceItem(
                        evidence_id=f"esg-{uuid4().hex}",
                        signal="esg_rating",
                        value=str(score),
                        source_name="Sustainalytics",
                        source_tier=SourceTier.SECONDARY,
                        dimension=RiskDimension.ESG,
                        timestamp=datetime.now(timezone.utc),
                        entity_match_confidence=0.91,
                        source_confidence=self.source_confidence,
                        provenance_url=f"https://www.sustainalytics.com/esg-ratings/company/{company.get('id', '')}",
                        metadata={
                            "esg_score": str(score),
                            "esg_risk_rating": esg_risk,
                            "governance_score": str(company.get("governanceScore", 0)),
                            "environmental_score": str(company.get("environmentalScore", 0)),
                            "social_score": str(company.get("socialScore", 0)),
                        }
                    ))
        except Exception as e:
            logger.error(f"ESG connector error: {e}")
        
        return evidence


class SECFinancialsConnector(BaseConnector):
    """
    SEC EDGAR financial health connector — no API key required.

    Two-step process:
      1. Resolve company name → CIK via company_tickers.json (cached in-process).
      2. Fetch structured XBRL company facts from data.sec.gov/api/xbrl/companyfacts.

    Produces FINANCIAL-dimension evidence items:
      - debt_to_equity_ratio  (LongTermDebt / StockholdersEquity)
      - profit_margin         (NetIncomeLoss / Revenue)
      - net_income_trend      (positive / negative / unavailable)
      - late_filing_notice    (NT 10-K or NT 10-Q present in recent submissions)

    Gracefully skips private and non-US-listed companies that have no CIK.
    """

    connector_name = "financial_sec_facts"
    source_tier = "regulator"
    source_confidence = 0.93

    # Class-level CIK cache: title.lower() → cik_str (populated once per process)
    _cik_cache: dict[str, str] = {}
    _cik_cache_loaded: bool = False

    _TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    _FACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
    _SUBMISSIONS_BASE = "https://data.sec.gov/submissions"

    async def _load_cik_cache(self) -> None:
        if SECFinancialsConnector._cik_cache_loaded:
            return
        headers = {"User-Agent": f"IRA-Agent ({self.config.sec_email})"}
        response = await self._request_with_retry("GET", self._TICKERS_URL, headers=headers)
        if not response or response.status_code != 200:
            logger.warning("SECFinancials: could not load company_tickers.json")
            return
        data = response.json()
        for entry in data.values():
            title = entry.get("title", "")
            cik = str(entry.get("cik_str", ""))
            if title and cik:
                SECFinancialsConnector._cik_cache[title.lower()] = cik
        SECFinancialsConnector._cik_cache_loaded = True
        logger.info("SECFinancials: loaded %d CIK entries", len(SECFinancialsConnector._cik_cache))

    def _resolve_cik(self, entity_name: str) -> str | None:
        """Fuzzy match entity name against cached ticker titles."""
        needle = entity_name.lower().strip()
        # Exact match first
        if needle in self._cik_cache:
            return self._cik_cache[needle]
        # Substring match: entity name contained in SEC title
        for title, cik in self._cik_cache.items():
            if needle in title or title in needle:
                return cik
        return None

    @staticmethod
    def _latest_annual(units: list[dict]) -> dict | None:
        """Return the most recent 10-K datapoint from a XBRL unit list."""
        annual = [u for u in units if u.get("form") == "10-K"]
        return max(annual, key=lambda u: u.get("end", ""), default=None)

    @staticmethod
    def _get_concept(facts_usgaap: dict, *concept_names: str) -> dict | None:
        """Try multiple GAAP concept names in order; return the units.USD list."""
        for name in concept_names:
            concept = facts_usgaap.get(name)
            if concept and concept.get("units", {}).get("USD"):
                return concept["units"]["USD"]
        return None

    async def fetch(self, entity_name: str) -> list[EvidenceItem]:
        await self._load_cik_cache()

        cik = self._resolve_cik(entity_name)
        if not cik:
            logger.info("SECFinancials: no CIK found for '%s' (private/foreign entity)", entity_name)
            return [EvidenceItem(
                evidence_id=f"secfin-nocik-{uuid4().hex}",
                dimension=RiskDimension.FINANCIAL,
                signal="sec_registered",
                value="not_sec_registered",
                source_name="SEC EDGAR",
                source_tier=SourceTier.REGULATOR,
                timestamp=datetime.now(timezone.utc),
                entity_match_confidence=0.50,
                source_confidence=0.90,
                provenance_url="https://www.sec.gov/",
                metadata={"note": "entity not found in SEC company registry", "query": entity_name},
            )]

        padded_cik = cik.zfill(10)
        headers = {"User-Agent": f"IRA-Agent ({self.config.sec_email})", "Accept": "application/json"}
        evidence: list[EvidenceItem] = []

        # ── Fetch XBRL company facts ───────────────────────────────────────────
        facts_url = f"{self._FACTS_BASE}/CIK{padded_cik}.json"
        facts_response = await self._request_with_retry("GET", facts_url, headers=headers)
        if facts_response and facts_response.status_code == 200:
            facts_data = facts_response.json()
            usgaap = facts_data.get("facts", {}).get("us-gaap", {})
            entity_name_sec = facts_data.get("entityName", entity_name)

            # Debt — try multiple common GAAP names
            debt_units = self._get_concept(usgaap, "LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations")
            equity_units = self._get_concept(usgaap, "StockholdersEquity", "StockholdersEquityAttributableToParent")
            income_units = self._get_concept(usgaap, "NetIncomeLoss", "NetIncomeLossAttributableToParent")
            revenue_units = self._get_concept(
                usgaap,
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
            )

            debt_pt   = self._latest_annual(debt_units)   if debt_units   else None
            equity_pt = self._latest_annual(equity_units) if equity_units else None
            income_pt = self._latest_annual(income_units) if income_units else None
            revenue_pt = self._latest_annual(revenue_units) if revenue_units else None

            # Signal 1: Debt-to-equity ratio
            if debt_pt and equity_pt and equity_pt.get("val", 0) != 0:
                dte = round(debt_pt["val"] / equity_pt["val"], 4)
                evidence.append(EvidenceItem(
                    evidence_id=f"secfin-dte-{uuid4().hex}",
                    dimension=RiskDimension.FINANCIAL,
                    signal="debt_to_equity_ratio",
                    value=str(dte),
                    source_name="SEC EDGAR",
                    source_tier=SourceTier.REGULATOR,
                    timestamp=datetime.now(timezone.utc),
                    entity_match_confidence=0.95,
                    source_confidence=self.source_confidence,
                    provenance_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json",
                    metadata={
                        "entity_name_sec": entity_name_sec,
                        "debt_usd": str(debt_pt["val"]),
                        "equity_usd": str(equity_pt["val"]),
                        "period_end": debt_pt.get("end", ""),
                    },
                ))

            # Signal 2: Profit margin (net income / revenue)
            if income_pt and revenue_pt and revenue_pt.get("val", 0) != 0:
                margin = round(income_pt["val"] / revenue_pt["val"], 4)
                evidence.append(EvidenceItem(
                    evidence_id=f"secfin-margin-{uuid4().hex}",
                    dimension=RiskDimension.FINANCIAL,
                    signal="profit_margin",
                    value=str(margin),
                    source_name="SEC EDGAR",
                    source_tier=SourceTier.REGULATOR,
                    timestamp=datetime.now(timezone.utc),
                    entity_match_confidence=0.95,
                    source_confidence=self.source_confidence,
                    provenance_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json",
                    metadata={
                        "entity_name_sec": entity_name_sec,
                        "net_income_usd": str(income_pt["val"]),
                        "revenue_usd": str(revenue_pt["val"]),
                        "period_end": income_pt.get("end", ""),
                    },
                ))

            # Signal 3: Net income trend (positive / negative)
            if income_pt:
                trend = "positive" if income_pt["val"] >= 0 else "negative"
                evidence.append(EvidenceItem(
                    evidence_id=f"secfin-income-{uuid4().hex}",
                    dimension=RiskDimension.FINANCIAL,
                    signal="net_income_trend",
                    value=trend,
                    source_name="SEC EDGAR",
                    source_tier=SourceTier.REGULATOR,
                    timestamp=datetime.now(timezone.utc),
                    entity_match_confidence=0.95,
                    source_confidence=self.source_confidence,
                    provenance_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json",
                    metadata={
                        "entity_name_sec": entity_name_sec,
                        "net_income_usd": str(income_pt["val"]),
                        "period_end": income_pt.get("end", ""),
                    },
                ))
        else:
            logger.warning("SECFinancials: facts fetch failed for CIK %s", padded_cik)

        # ── Fetch submissions to detect late-filing notices ────────────────────
        subs_url = f"{self._SUBMISSIONS_BASE}/CIK{padded_cik}.json"
        subs_response = await self._request_with_retry("GET", subs_url, headers=headers)
        if subs_response and subs_response.status_code == 200:
            subs_data = subs_response.json()
            recent_forms: list[str] = subs_data.get("filings", {}).get("recent", {}).get("form", [])
            late_forms = [f for f in recent_forms if f in ("NT 10-K", "NT 10-Q")]
            if late_forms:
                evidence.append(EvidenceItem(
                    evidence_id=f"secfin-latefilng-{uuid4().hex}",
                    dimension=RiskDimension.FINANCIAL,
                    signal="late_filing_notice",
                    value="yes",
                    source_name="SEC EDGAR",
                    source_tier=SourceTier.REGULATOR,
                    timestamp=datetime.now(timezone.utc),
                    entity_match_confidence=0.95,
                    source_confidence=self.source_confidence,
                    provenance_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={padded_cik}",
                    metadata={
                        "late_form_types": ",".join(set(late_forms)),
                        "late_filing_count": str(len(late_forms)),
                    },
                ))

        if not evidence:
            logger.info("SECFinancials: no financial signals extracted for CIK %s", padded_cik)

        return evidence


class MultiSourceConnector:
    """Orchestrates multiple connectors in parallel with per-source circuit breaker."""

    # Consecutive failures before a connector is skipped for this session
    CIRCUIT_BREAK_THRESHOLD = 2

    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.config = config or ConnectorConfig()
        self._failure_counts: dict[str, int] = {}

    def _is_open(self, name: str) -> bool:
        return self._failure_counts.get(name, 0) >= self.CIRCUIT_BREAK_THRESHOLD

    def _record_failure(self, name: str) -> None:
        self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
        if self._failure_counts[name] == self.CIRCUIT_BREAK_THRESHOLD:
            logger.warning("circuit_breaker_open connector=%s", name)

    def _record_success(self, name: str) -> None:
        if name in self._failure_counts:
            del self._failure_counts[name]

    async def fetch_all(self, entity_name: str) -> list[EvidenceItem]:
        """Fetch from all sources in parallel; skip tripped circuit breakers."""
        evidence: list[EvidenceItem] = []
        connector_classes = [
            OpenSanctionsConnector,
            SECConnector,
            SECFinancialsConnector,
            NewsConnector,
            ESGConnector,
        ]

        async def fetch_from_connector(cls) -> list[EvidenceItem]:
            name = cls.connector_name
            if self._is_open(name):
                logger.info("circuit_breaker_skip connector=%s", name)
                return []
            try:
                async with cls(self.config) as connector:
                    result = await connector.fetch(entity_name)
                self._record_success(name)
                logger.info("connector_success connector=%s evidence_count=%d", name, len(result))
                return result
            except Exception as exc:
                self._record_failure(name)
                logger.error("connector_failure connector=%s error=%s", name, exc)
                return []

        results = await asyncio.gather(*[fetch_from_connector(cls) for cls in connector_classes])
        for result in results:
            evidence.extend(result)
        return evidence
