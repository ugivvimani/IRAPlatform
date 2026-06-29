# Integrity Risk Assessment (IRA) Agent

A multi-agent FastAPI system that assesses company integrity risk using live data sources, LLM reasoning, and vector memory. Built Azure-native for App Service / Web App for Containers.

---

## Architecture Overview

The system uses a **hub-and-spoke orchestrator** pattern with 6 specialised agents. Each `POST /assess` request flows through the full pipeline:

```mermaid
sequenceDiagram
    participant Client
    participant Orchestrator
    participant RetrievalAgent
    participant Connectors as Live Connectors<br/>(Sanctions · SEC · News · ESG)
    participant MemoryAgent as MemoryManagerAgent<br/>(Pinecone Vector DB)
    participant ConflictAgent as ConflictResolutionAgent<br/>(Tree-of-Thought)
    participant AnalysisAgent as AnalysisForecastingAgent
    participant OutputAgent as OutputComposerAgent
    participant CalibrationAgent

    Client->>Orchestrator: POST /assess { company_name, question }

    Orchestrator->>RetrievalAgent: retrieve(entity)
    RetrievalAgent->>Connectors: fetch_all(entity) [parallel]
    Connectors-->>RetrievalAgent: EvidenceItem[] (sanctions, filings, news, ESG)
    RetrievalAgent-->>Orchestrator: evidence[]

    Orchestrator->>MemoryAgent: load_historical_context(entity)
    MemoryAgent-->>Orchestrator: MemoryFact[] (LLM-filtered from Pinecone)

    Note over Orchestrator: Cold-start check — seed source priors if first encounter

    Orchestrator->>ConflictAgent: resolve(evidence, history, conflict_history)
    ConflictAgent-->>Orchestrator: ConflictResolutionResult (winner branch, rationale)

    alt conflict_detected OR winner_score < 0.75
        Orchestrator->>RetrievalAgent: retrieve_supplemental(entity, keywords)
        RetrievalAgent-->>Orchestrator: additional evidence (Revise step)
    end

    Orchestrator->>AnalysisAgent: score(evidence)
    AnalysisAgent-->>Orchestrator: composite_quant_score

    Orchestrator->>OutputAgent: compose(evidence, conflict_result, quant_score)
    OutputAgent-->>Orchestrator: RiskDecision (rating, confidence, summary)

    par Persist to vector DB
        Orchestrator->>MemoryAgent: persist_facts(evidence summaries)
        Orchestrator->>MemoryAgent: persist_assessment_narrative(summary)
        Orchestrator->>MemoryAgent: persist_conflict_note (if conflict)
    and Update calibration
        Orchestrator->>CalibrationAgent: build_record(decision, evidence)
        Orchestrator->>MemoryAgent: persist_calibration(record)
    end

    Orchestrator-->>Client: AssessmentResponse { risk_rating, confidence, summary, ... }
```

---

## Agent Responsibilities

| Agent | Role |
|---|---|
| **OrchestratorAgent** | Coordinates the full pipeline; applies policy thresholds; triggers Revise step |
| **RetrievalAgent** | Fetches live evidence from all connectors in parallel; detects entity-not-found |
| **MemoryManagerAgent** | Reads/writes Pinecone vector DB; LLM-summarizes evidence; relevance-filters history |
| **ConflictResolutionAgent** | Tree-of-Thought beam search to resolve contradictory signals; sparsity dampener |
| **AnalysisForecastingAgent** | Scores evidence into a composite quantitative risk score |
| **OutputComposerAgent** | Converts scores into a human-readable risk decision with recommended next steps |
| **CalibrationAgent** | Tracks TP/FP/TN/FN per source; updates reliability scores over time |

---

## Live Data Connectors

| Connector | Source | Dimension |
|---|---|---|
| `OpenSanctionsConnector` | opensanctions.org | Sanctions |
| `SECConnector` | SEC EDGAR full-text search | Regulatory |
| `SECFinancialsConnector` | SEC EDGAR XBRL + filing dates | Financial |
| `NewsConnector` | NewsAPI | Reputational |
| `ESGConnector` | ESG rating provider | ESG |

Set `ENABLE_LIVE_CONNECTORS=true` to activate live API calls. Mock connectors are used by default.

---

## Vector DB Namespaces (Pinecone)

| Namespace | What is stored |
|---|---|
| `historical_facts` | LLM-summarized evidence items per entity per dimension |
| `assessment_narratives` | Full assessment summaries for semantic retrieval of similar past cases |
| `calibration` | Per-source reliability scores (TP/FP/TN/FN counters + Bayesian blend) |
| `conflict_notes` | Conflict resolution rationale for temporal coherence scoring |

---

## Run Locally

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env — set OPENROUTER_API_KEY, PINECONE_API_KEY at minimum
python run_local.py
```

The server starts on `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

---

## Key Environment Variables

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openrouter` (default) \| `openai` \| `azure_openai` \| `stub` |
| `OPENROUTER_API_KEY` | OpenRouter API key (used for both LLM and embeddings) |
| `LLM_MODEL` | LLM model name (default: `openai/gpt-4o-mini`) |
| `EMBEDDING_TYPE` | `openrouter` (default) \| `openai` \| `local` |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX` | Pinecone index name (default: `ira-platform-memory`) |
| `ENABLE_LIVE_CONNECTORS` | `true` to call live APIs; `false` uses mocks (default) |
| `DB_BACKEND` | `sqlite` (default) \| `postgres` |
| `POSTGRES_DSN` | PostgreSQL connection string (when `DB_BACKEND=postgres`) |
| `SQLITE_DB_PATH` | SQLite file path (default: `./data/ira.db`) |
| `NEWS_API_KEY` | NewsAPI key for reputational signal connector |
| `OPENSANCTIONS_API_KEY` | OpenSanctions API key |
| `SEC_CONTACT_EMAIL` | Required User-Agent contact for SEC EDGAR requests |
| `SERVICE_API_KEY` | Shared API key required in `X-API-Key` header; unset = open (local dev) |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Optional — enables Azure Monitor telemetry |

---

## API Endpoints

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/api/docs`

### Health

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Liveness check — returns env, LLM backend, storage backend |
| `GET` | `/ready` | None | Readiness probe — tests vector store and DB connectivity |

### Authentication

All endpoints (except `/health` and `/ready`) require the `X-API-Key` header.

Set `SERVICE_API_KEY` in your environment:
```
SERVICE_API_KEY=your-secret-key
```

The upstream integrity platform includes this header on every request:
```
X-API-Key: your-secret-key
```

If `SERVICE_API_KEY` is not set the service runs open — useful for local development and testing.

**`POST /auth/token` has been removed.** JWT and role-based access control are not used; this is an internal service-to-service API.

### Assessments

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/assess` | API Key | Synchronous company risk assessment |
| `POST` | `/assess/async` | API Key | Queue an async assessment job (returns `task_id`) |
| `GET` | `/tasks/{task_id}` | API Key | Poll async job status and retrieve result |
| `GET` | `/assessments/{entity_id}` | API Key | Assessment history for an entity (`?limit=25`) |

**`POST /assess` request body:**
```json
{
  "query": {
    "company_name": "Tesla Inc",
    "question": "Is Tesla safe to partner with?",
    "requested_dimensions": []
  }
}
```

**`POST /assess/async` request body:**
```json
{
  "company_name": "Tesla Inc",
  "question": "Is Tesla safe to partner with?",
  "callback_url": "https://your-app.com/webhooks/ira"
}
```

`callback_url` is optional. When provided, the service POSTs the completed result to that URL once the job finishes.

**Compact response:**
```json
{
  "assessment_id": 1,
  "company_name": "Tesla Inc",
  "risk_rating": "watch",
  "confidence": "medium",
  "summary": "...",
  "recommended_next_steps": ["..."],
  "requires_manual_review": true,
  "evaluated_at": "2026-06-28T22:00:00Z"
}
```

**`POST /assess` query params:**
- `include_details=true` — returns full `AssessmentResponse` (evidence chain, conflict result, telemetry); default returns compact summary

**Risk ratings:** `safe` | `watch` | `high_risk` | `restricted`  
**Confidence levels:** `low` | `medium` | `high`

### Watchlist

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/watchlist` | API Key | Add or update an entity on the monitoring watchlist |
| `GET` | `/watchlist` | API Key | List all watchlist entities |
| `GET` | `/watchlist/{entity_id}` | API Key | Get last assessment for entity; `?refresh=true` triggers new live assessment |
| `DELETE` | `/watchlist/{entity_id}` | API Key | Remove entity from watchlist |

### Policies

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/policies/active` | API Key | Get all active policy thresholds applied to each assessment |
| `PUT` | `/policies/{policy_key}` | API Key | Create or update a policy threshold |

**Example policy threshold:**
```json
{
  "threshold_value": 0.75,
  "description": "Minimum score before auto-HOLD decision",
  "is_active": true
}
```

---

## Observability

- **Always on:** Structured JSON logging (`LOG_FORMAT=json`) at every pipeline stage
- **Azure Monitor:** Set `APPLICATIONINSIGHTS_CONNECTION_STRING` to stream traces to App Insights via OpenTelemetry — optional, graceful no-op if unset
- **Request metrics:** `MetricsMiddleware` logs method, path, status code, and duration for every request
- **Assessment telemetry:** Each response includes a `model_metadata.telemetry` block with risk rating, confidence score, evidence count, conflict detection flag, and quant scores

---

## Azure Deployment

Deployment is done **locally via Azure CLI** — no deployment secrets are stored in this repository.

```powershell
# One-time: install Azure CLI
winget install --id Microsoft.AzureCLI

# Authenticate
az login

# First deploy (creates App Service + sets all secrets + deploys code)
cd C:\Users\<you>\ira-deploy
.\deploy.ps1 -Create

# Re-deploy after code changes
.\deploy.ps1
```

The deploy script lives **outside the repo** (`~\ira-deploy\deploy.ps1`) and contains all secrets and infra names. It is never committed.

### What the script does

1. Creates a **Linux App Service** (Python 3.11, B1/P0v3 SKU)
2. Sets all environment variables via `az webapp config appsettings set` — secrets stay in Azure Configuration, not in any file
3. Packages the code (zip, excluding `.venv`, `.env`, tests, local data)
4. Deploys via `az webapp deploy --src-path app.zip`

### Startup command

Azure App Service uses this command to start the app (set under **Configuration → General settings**):

```
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app
```

### Required Azure resources

| Resource | Purpose |
|---|---|
| App Service Plan (B1 or P0v3, Linux) | Compute for the API |
| App Service (Python 3.11) | Hosts the FastAPI app |
| Azure Storage Account *(optional)* | Enables Azure Storage Queue for async jobs in production |

### GitHub Actions

The workflow in `.github/workflows/ci-cd.yml` runs **tests only** on push/PR. It does not deploy. No Azure credentials are stored in GitHub secrets.

### Background Workers

| Script | Trigger | Purpose |
|---|---|---|
| `webjobs/continuous/run.py` | Azure Storage Queue | Drains async assessment jobs in production |
| `webjobs/scheduled/run.py` | Cron / timer trigger | Re-assesses all watchlist entities on a schedule |

In local/demo mode the background worker runs as an in-process thread — no Azure Storage Queue is required.

---

## Running Tests

```powershell
python -m pytest tests/ -q
```

73 tests covering agents, connectors, conflict resolution, calibration, cold-start, and scoring consistency.
