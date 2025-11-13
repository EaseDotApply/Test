"""API request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Sequence

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response payload."""

    status: str = Field(default="ok", description="Overall application health status.")
    environment: str = Field(description="Current deployment environment label.")


class ConversationTurn(BaseModel):
    role: Literal["user", "ai"] = Field(description="Speaker role in the conversation history.")
    content: str = Field(description="Utterance text.")


class AskRequest(BaseModel):
    question: str = Field(description="Natural-language question to answer.")
    history: Sequence[ConversationTurn] | None = Field(
        default=None,
        description="Optional conversation turns to preserve context.",
    )
    force_refresh: bool = Field(
        default=False,
        description="If True, refreshes upstream data and vectorstore before answering.",
    )


class CitationModel(BaseModel):
    message_id: str
    user_name: str
    timestamp: datetime
    snippet: str


class AskResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    citations: List[CitationModel]
    guardrails: dict[str, bool | float | str | None]


class SimpleAskResponse(BaseModel):
    """Simple response format matching assessment requirement."""
    answer: str


class RefreshResponse(BaseModel):
    message: str
    documents_indexed: int | None = Field(default=None, description="Number of documents available after refresh.")


class InsightsResponse(BaseModel):
    highlights: list[str] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    generated_at: datetime


__all__ = [
    "HealthResponse",
    "AskRequest",
    "AskResponse",
    "SimpleAskResponse",
    "ConversationTurn",
    "RefreshResponse",
    "InsightsResponse",
    "CitationModel",
]
