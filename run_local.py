from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from app.settings import load_settings


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    _load_dotenv(Path(".env"))
    settings = load_settings()
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
