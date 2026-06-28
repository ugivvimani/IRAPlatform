from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_connectors, get_embedding_model
from app.core.security import User, get_authenticated_user

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/connectors/{entity_name}")
async def debug_connectors(entity_name: str, user: User = Depends(get_authenticated_user), connectors=Depends(get_connectors)) -> dict:
    del user
    evidence = await connectors.fetch_all(entity_name)
    return {
        "entity": entity_name,
        "evidence_count": len(evidence),
        "evidence": [
            {
                "signal": e.signal,
                "value": e.value,
                "source": e.source_name,
                "confidence": e.source_confidence,
            }
            for e in evidence
        ],
    }


@router.post("/embed")
async def debug_embed(texts: list[str], user: User = Depends(get_authenticated_user), embedding_model=Depends(get_embedding_model)) -> dict:
    del user
    embeddings = embedding_model.embed_sync(texts)
    return {
        "texts": texts,
        "embeddings_count": len(embeddings),
        "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
    }

