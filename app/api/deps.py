from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request

from app.contracts import AssessmentResponse


@dataclass(frozen=True, slots=True)
class AppState:
    settings: object
    llm_client: object
    vector_store: object
    storage_repo: object
    orchestrator: object
    health_service: object
    connectors: object
    embedding_model: object


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_settings(request: Request):
    return get_app_state(request).settings


def get_llm_client(request: Request):
    return get_app_state(request).llm_client


def get_storage_repo(request: Request):
    return get_app_state(request).storage_repo


def get_orchestrator(request: Request):
    return get_app_state(request).orchestrator


def get_health_service(request: Request):
    return get_app_state(request).health_service


def get_connectors(request: Request):
    return get_app_state(request).connectors


def get_embedding_model(request: Request):
    return get_app_state(request).embedding_model


def persist_assessment_result(request: Request, result: AssessmentResponse) -> None:
    storage_repo = get_storage_repo(request)
    storage_repo.insert_assessment(result)


def get_watchlist_entry_or_404(request: Request, entity_id: str):
    storage_repo = get_storage_repo(request)
    entry = storage_repo.get_watchlist(entity_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not on watchlist.")
    return entry

