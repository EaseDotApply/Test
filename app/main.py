"""ASGI entrypoint for the question-answering service."""
from __future__ import annotations

import uvicorn

from app.core.app import create_app
from app.core.config import get_settings

app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        factory=False,
    )
