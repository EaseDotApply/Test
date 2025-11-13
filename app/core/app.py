"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.core.config import AppSettings, get_settings
from app.core.logging import configure_logging, get_logger
from app.api.routes import register_routes


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Lifecycle management for startup and shutdown hooks."""

    settings = get_settings()
    logger = get_logger("app.lifespan")
    logger.info("app.start", environment=settings.environment)
    try:
        yield
    finally:
        logger.info("app.stop")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Application factory used by ASGI servers and tests."""

    configure_logging()

    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="November Member QA",
        description="Answer natural-language questions over member messages using OSS LLMs.",
        version="0.1.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app)

    return app


__all__ = ["create_app"]
