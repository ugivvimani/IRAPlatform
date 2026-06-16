"""
Real external data source connectors with retry, rate-limiting, and error handling.
Supports OFAC, SEC Edgar, Reuters, and ESG providers.
"""
import asyncio
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional
import logging

import aiohttp
import httpx
from bs4 import BeautifulSoup

from app.contracts import EvidenceItem, RiskDimension

logger = logging.getLogger(__name__)


class ConnectorConfig:
    """Configuration for external connectors."""
    
    def __init__(self):
        self.ofac_api_key = os.getenv("OFAC_API_KEY", "")
        self.ofac_url = os.getenv("OFAC_URL", "https://api.treasury.gov/ofac/")
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


class OFACConnector(BaseConnector):
    """OFAC Consolidated Sanctions List connector."""
    
    connector_name = "sanctions_ofac"
    source_tier = "official"
    source_confidence = 0.98
    
    async def fetch(self, entity_name: str) -> list[EvidenceItem]:
        """
        Query OFAC Consolidated Sanctions List (CSL).
        Public API: https://webservices.treasury.gov/ofac_sanctions_data/
        """
        evidence = []
        
        # Fall back to public OFAC data endpoint
        url = "https://webservices.treasury.gov/ofac_sanctions_data/v1/search"
        params = {
            "name": entity_name,
            "fuzzymode": 1,  # Enable fuzzy matching
        }
        
        try:
            response = await self._request_with_retry("GET", url, params=params)
            if response and response.status_code == 200:
                data = response.json()
                # Parse OFAC response format
                if "sdnList" in data:
                    for sdn in data["sdnList"]:
                        evidence.append(EvidenceItem(
                            signal="sanctions_listed",
                            value="yes",
                            source_name="OFAC",
                            source_tier=self.source_tier,
                            dimension=RiskDimension.SANCTIONS,
                            timestamp=datetime.now(timezone.utc),
                            entity_match_confidence=0.95,
                            source_confidence=self.source_confidence,
                            provenance_url=f"https://ofac.treasury.gov/SDN-List/consolidated-list-overview",
                            metadata={
                                "sdn_id": sdn.get("entityID", ""),
                                "program": sdn.get("program", ""),
                                "list_date": sdn.get("listingDate", ""),
                            }
                        ))
                else:
                    # No match = not sanctioned
                    evidence.append(EvidenceItem(
                        signal="sanctions_status",
                        value="not_sanctioned",
                        source_name="OFAC",
                        source_tier=self.source_tier,
                        dimension=RiskDimension.SANCTIONS,
                        timestamp=datetime.now(timezone.utc),
                        entity_match_confidence=0.92,
                        source_confidence=self.source_confidence,
                        provenance_url="https://ofac.treasury.gov/SDN-List/consolidated-list-overview",
                        metadata={"search_query": entity_name}
                    ))
        except Exception as e:
            logger.error(f"OFAC connector error: {e}")
        
        return evidence


class SECConnector(BaseConnector):
    """SEC Edgar API connector for regulatory filings and enforcement."""
    
    connector_name = "regulatory_sec"
    source_tier = "regulator"
    source_confidence = 0.96
    
    async def fetch(self, entity_name: str) -> list[EvidenceItem]:
        """
        Query SEC Edgar for company filings and enforcement actions.
        Public API: https://www.sec.gov/cgi-bin/browse-edgar
        """
        evidence = []
        
        # Query SEC Edgar for company information
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "company": entity_name,
            "action": "getcompany",
            "type": "10-K",  # Annual reports
            "dateb": "",
            "owner": "exclude",
            "count": 10,
            "format": "json",
        }
        
        headers = {"User-Agent": f"IRA-Agent ({self.config.sec_email})"}
        
        try:
            response = await self._request_with_retry("GET", url, headers=headers, params=params)
            if response and response.status_code == 200:
                data = response.json()
                if "hits" in data and data["hits"] > 0:
                    evidence.append(EvidenceItem(
                        signal="sec_filings_found",
                        value=f"{data['hits']}",
                        source_name="SEC Edgar",
                        source_tier=self.source_tier,
                        dimension=RiskDimension.REGULATORY,
                        timestamp=datetime.now(timezone.utc),
                        entity_match_confidence=0.94,
                        source_confidence=self.source_confidence,
                        provenance_url=f"https://www.sec.gov/cgi-bin/browse-edgar?company={entity_name}",
                        metadata={"filings_count": str(data["hits"])}
                    ))
                
                # Check for enforcement actions
                enforcement_url = "https://www.sec.gov/cgi-bin/browse-edgar"
                enforcement_params = {
                    "company": entity_name,
                    "action": "getcompany",
                    "type": "424B5",  # Prospectus filings (often related to issues)
                    "owner": "exclude",
                    "count": 5,
                    "format": "json",
                }
                
                enforcement_response = await self._request_with_retry(
                    "GET", enforcement_url, headers=headers, params=enforcement_params
                )
                
                if enforcement_response and enforcement_response.status_code == 200:
                    enforcement_data = enforcement_response.json()
                    if enforcement_data.get("hits", 0) > 0:
                        evidence.append(EvidenceItem(
                            signal="regulatory_filings_unusual",
                            value=f"{enforcement_data['hits']}",
                            source_name="SEC Edgar",
                            source_tier=self.source_tier,
                            dimension=RiskDimension.REGULATORY,
                            timestamp=datetime.now(timezone.utc),
                            entity_match_confidence=0.85,
                            source_confidence=0.90,
                            provenance_url=f"https://www.sec.gov/cgi-bin/browse-edgar?company={entity_name}",
                            metadata={"unusual_filings": str(enforcement_data["hits"])}
                        ))
        except Exception as e:
            logger.error(f"SEC connector error: {e}")
        
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
                            signal="news_sentiment",
                            value=sentiment,
                            source_name=article.get("source", {}).get("name", "News"),
                            source_tier=self.source_tier,
                            dimension=RiskDimension.REPUTATIONAL,
                            timestamp=datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")),
                            entity_match_confidence=0.88,
                            source_confidence=self.source_confidence,
                            provenance_url=article.get("url", ""),
                            metadata={
                                "title": article.get("title", ""),
                                "description": article.get("description", "")[:200],
                            }
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
                        signal="esg_rating",
                        value=str(score),
                        source_name="Sustainalytics",
                        source_tier=self.source_tier,
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


class MultiSourceConnector:
    """Orchestrates multiple connectors in parallel."""
    
    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.config = config or ConnectorConfig()
    
    async def fetch_all(self, entity_name: str) -> list[EvidenceItem]:
        """Fetch from all sources in parallel."""
        evidence = []
        
        connector_classes = [
            OFACConnector,
            SECConnector,
            NewsConnector,
            ESGConnector,
        ]
        
        tasks = []
        for connector_class in connector_classes:
            async def fetch_from_connector(cls=connector_class):
                try:
                    async with cls(self.config) as connector:
                        return await connector.fetch(entity_name)
                except Exception as e:
                    logger.error(f"Error in {cls.connector_name}: {e}")
                    return []
            
            tasks.append(fetch_from_connector())
        
        results = await asyncio.gather(*tasks)
        for result in results:
            evidence.extend(result)
        
        return evidence
