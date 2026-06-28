# Integrity Risk Assessment Agent

Azure-first implementation for App Service / Web App for Containers.

## Current platform choices

- **Observability:** Azure Application Insights via OpenTelemetry exporter
- **Background processing:** Azure WebJobs + Azure Storage Queue (Celery removed)
- **Deployment:** GitHub Actions -> Azure Container Registry -> Azure App Service
- **Secrets:** Azure Key Vault (loaded by managed identity at startup)

## Run locally

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
python run_local.py
```

## Key environment variables

- `APPLICATIONINSIGHTS_CONNECTION_STRING`
- `AZURE_KEY_VAULT_URL`
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_WEBJOBS_ASSESSMENT_QUEUE`
- `DB_BACKEND` (`sqlite` or `postgres`)
- `POSTGRES_DSN`
- `SQLITE_DB_PATH`
- `PINECONE_API_KEY`
- `OPENAI_API_KEY`
- `EMBEDDING_TYPE` (`openai`)
- `OPENAI_EMBEDDING_MODEL` (`text-embedding-3-small`)
- `ENABLE_LIVE_CONNECTORS` (`true` to pull live data in `/assess`)
- `NEWS_API_KEY` (for news connector)
- `SEC_CONTACT_EMAIL` (required user-agent contact for SEC requests)
- `ESG_API_KEY` (optional ESG connector)

## API endpoints

- `GET /health`
- `GET /ready`
- `POST /assess`
- `POST /assess/async` (queues assessment job)
- `GET /tasks/{task_id}`
- `POST /watchlist`
- `GET /watchlist`
- `GET /watchlist/{entity_id}`
- `GET /assessments/{entity_id}`

## Azure deployment pipeline

Workflow file: `.github/workflows/ci-cd.yml`

Required GitHub variables/secrets:
- `vars.AZURE_WEBAPP_NAME`
- `vars.AZURE_RESOURCE_GROUP`
- `vars.AZURE_CONTAINER_REGISTRY`
- `secrets.AZURE_CREDENTIALS`

## WebJobs scripts

- `webjobs/continuous/run.py` - queue-driven assessment worker
- `webjobs/scheduled/run.py` - scheduled watchlist reassessment
