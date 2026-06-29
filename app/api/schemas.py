from __future__ import annotations

from pydantic import BaseModel, HttpUrl


class AsyncAssessRequest(BaseModel):
    company_name: str
    question: str
    # Optional: when set, the service POSTs the completed result to this URL
    callback_url: str | None = None


class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str
    callback_url: str | None = None

