# Integrity Risk Assessment Agent

Initial implementation scaffold for the phased BRD plan.

## Run locally

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
python run_local.py
```

Local runtime settings are loaded from environment variables (or `.env`):
- `APP_ENV`
- `APP_HOST`
- `APP_PORT`
- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `PINECONE_NAMESPACE`

## Implemented in this scaffold

- Core contracts for query, evidence, conflict branches, calibration, and output
- Orchestrator loop skeleton (Think/Act/Observe/Revise/Conclude)
- Retrieval, memory, conflict-resolution, and analysis agent placeholders
- Pinecone vector-memory adapter (Pinecone-only mode)
- FastAPI endpoints:
  - `GET /health`
  - `POST /assess`
