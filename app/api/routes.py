"""Top-level API router registration."""
from __future__ import annotations

from typing import Sequence, Tuple

from fastapi import APIRouter, Depends, FastAPI, status

from app.api.dependencies import get_insights_service, get_qa_service, get_vector_service
from app.api.schemas import (
    AskRequest,
    AskResponse,
    CitationModel,
    ConversationTurn,
    HealthResponse,
    InsightsResponse,
    RefreshResponse,
    SimpleAskResponse,
)
from app.core.config import AppSettings, get_settings
from app.services.insights import InsightsService
from app.services.models import AnswerResult
from app.services.qa import QAService
from app.services.vectorstore import VectorStoreService


def register_routes(app: FastAPI) -> None:
    """Attach API routers to the FastAPI application."""

    api_router = APIRouter(prefix="/api")
    api_router.include_router(build_health_router())
    api_router.include_router(build_qa_router())
    api_router.include_router(build_insights_router())

    app.include_router(api_router)


def build_health_router() -> APIRouter:
    router = APIRouter(tags=["system"])

    @router.get("/health", response_model=HealthResponse, summary="Service health probe")
    async def health_check(settings: AppSettings = Depends(get_settings)) -> HealthResponse:
        return HealthResponse(status="ok", environment=settings.environment)

    return router


def build_qa_router() -> APIRouter:
    router = APIRouter(tags=["qa"])

    @router.post("/ask", response_model=SimpleAskResponse, status_code=status.HTTP_200_OK)
    async def ask_question(
        request: AskRequest,
        qa_service: QAService = Depends(get_qa_service),
    ) -> SimpleAskResponse:
        """Simple question-answering endpoint returning only the answer."""
        history = _to_history(request.history)
        answer: AnswerResult = await qa_service.ask(
            request.question,
            chat_history=history,
            force_refresh=request.force_refresh,
        )
        return SimpleAskResponse(answer=answer.answer)

    @router.post("/ask/detailed", response_model=AskResponse, status_code=status.HTTP_200_OK)
    async def ask_question_detailed(
        request: AskRequest,
        qa_service: QAService = Depends(get_qa_service),
    ) -> AskResponse:
        """Detailed question-answering endpoint with metadata."""
        history = _to_history(request.history)
        answer: AnswerResult = await qa_service.ask(
            request.question,
            chat_history=history,
            force_refresh=request.force_refresh,
        )
        citations = [
            CitationModel(
                message_id=citation.message_id,
                user_name=citation.user_name,
                timestamp=citation.timestamp,
                snippet=citation.snippet,
            )
            for citation in answer.citations
        ]
        return AskResponse(
            answer=answer.answer,
            confidence=answer.confidence,
            reasoning=answer.reasoning,
            citations=citations,
            guardrails=answer.guardrails or {},
        )

    @router.post("/refresh", response_model=RefreshResponse)
    async def refresh_index(
        vector_service: VectorStoreService = Depends(get_vector_service),
    ) -> RefreshResponse:
        await vector_service.ensure_retriever(force_refresh=True)
        return RefreshResponse(
            message="Vectorstore refreshed successfully.",
            documents_indexed=vector_service.document_count,
        )

    return router


def build_insights_router() -> APIRouter:
    router = APIRouter(tags=["insights"])

    @router.get("/insights", response_model=InsightsResponse)
    async def get_insights(
        insights_service: InsightsService = Depends(get_insights_service),
    ) -> InsightsResponse:
        payload = await insights_service.generate()
        return InsightsResponse(**payload)

    return router


def _to_history(turns: Sequence[ConversationTurn] | None) -> Sequence[Tuple[str, str]]:
    if not turns:
        return []
    return [(turn.role, turn.content) for turn in turns]


__all__ = ["register_routes"]
