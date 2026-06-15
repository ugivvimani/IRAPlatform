# Integrity Risk Assessment Agent

Initial implementation scaffold for the phased BRD plan.

## Run locally

```powershell
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Implemented in this scaffold

- Core contracts for query, evidence, conflict branches, calibration, and output
- Orchestrator loop skeleton (Think/Act/Observe/Revise/Conclude)
- Retrieval, memory, conflict-resolution, and analysis agent placeholders
- Vector store abstraction with configurable backend (`VECTOR_BACKEND=chroma|pinecone`)
- FastAPI endpoints:
  - `GET /health`
  - `POST /assess`
