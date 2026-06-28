from __future__ import annotations

from pydantic import BaseModel


class AsyncAssessRequest(BaseModel):
    company_name: str
    question: str


class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str

