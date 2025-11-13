"""Dependency providers for FastAPI endpoints."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.core.config import AppSettings, get_settings
from app.services.hallucination import HallucinationValidator
from app.services.insights import InsightsService
from app.services.qa import QAService
from app.services.vectorstore import VectorStoreService


@lru_cache(maxsize=1)
def get_vector_service_cached() -> VectorStoreService:
    settings = get_settings()
    return VectorStoreService(settings)


def get_vector_service(_: AppSettings = Depends(get_settings)) -> VectorStoreService:
    return get_vector_service_cached()


@lru_cache(maxsize=1)
def get_validator_cached() -> HallucinationValidator:
    settings = get_settings()
    return HallucinationValidator(settings)


@lru_cache(maxsize=1)
def get_qa_service_cached() -> QAService:
    settings = get_settings()
    return QAService(
        settings,
        retriever=get_vector_service_cached(),
        validator=get_validator_cached(),
    )


def get_qa_service(_: AppSettings = Depends(get_settings)) -> QAService:
    return get_qa_service_cached()


@lru_cache(maxsize=1)
def get_insights_service_cached() -> InsightsService:
    settings = get_settings()
    return InsightsService(settings=settings)


def get_insights_service(_: AppSettings = Depends(get_settings)) -> InsightsService:
    return get_insights_service_cached()


__all__ = [
    "get_vector_service",
    "get_qa_service",
    "get_insights_service",
]
