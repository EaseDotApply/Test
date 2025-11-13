from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from datetime import datetime, timezone

from app.api.dependencies import get_insights_service, get_qa_service, get_vector_service
from app.core.app import create_app
from app.core.config import AppSettings
from app.services.models import AnswerResult, Citation


@pytest.fixture
def temp_settings(tmp_path: Path) -> AppSettings:
    data_dir = tmp_path / "data"
    vector_dir = tmp_path / "vectorstore"
    reports_dir = tmp_path / "reports"
    return AppSettings(
        data_dir=str(data_dir),
        vectorstore_dir=str(vector_dir),
        reports_dir=str(reports_dir),
        messages_api_url="http://testserver/messages",
        request_page_size=1,
    )


@pytest.fixture
def fastapi_client() -> TestClient:
    app = create_app()

    class DummyQAService:
        async def ask(self, question: str, chat_history=None, force_refresh: bool = False) -> AnswerResult:
            return AnswerResult(
                question=question,
                answer="Layla is travelling to London in June.",
                confidence=0.9,
                citations=[
                    Citation(
                        message_id="msg-1",
                        user_name="Layla",
                        timestamp=datetime.now(timezone.utc),
                        snippet="Planning my trip to London in June!",
                    )
                ],
                reasoning="Derived from Layla's trip message.",
                guardrails={"has_context": True, "hallucination_supported": True},
            )

    class DummyVectorStore:
        document_count = 3

        async def ensure_retriever(self, force_refresh: bool = False):  # noqa: D401
            return object()

    class DummyInsightsService:
        async def generate(self):  # noqa: D401
            return {
                "highlights": ["Most active: Layla"],
                "anomalies": [],
                "generated_at": datetime.now(timezone.utc),
            }

    app.dependency_overrides[get_qa_service] = lambda: DummyQAService()
    app.dependency_overrides[get_vector_service] = lambda: DummyVectorStore()
    app.dependency_overrides[get_insights_service] = lambda: DummyInsightsService()

    return TestClient(app)
