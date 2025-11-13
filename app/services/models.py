"""Shared service-level data structures."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List

from langchain_core.documents import Document


@dataclass
class Citation:
    message_id: str
    user_name: str
    timestamp: datetime
    snippet: str

    @classmethod
    def from_document(cls, document: Document) -> "Citation":
        metadata = document.metadata
        timestamp = metadata.get("timestamp")
        if isinstance(timestamp, str):
            timestamp_dt = datetime.fromisoformat(timestamp)
        else:
            timestamp_dt = timestamp
        return cls(
            message_id=str(metadata.get("id")),
            user_name=str(metadata.get("user_name")),
            timestamp=timestamp_dt,
            snippet=document.page_content[:280],
        )


@dataclass
class AnswerResult:
    question: str
    answer: str
    confidence: float
    citations: List[Citation]
    reasoning: str
    guardrails: dict[str, bool] | None = None

    @classmethod
    def empty(cls, question: str) -> "AnswerResult":
        return cls(question=question, answer="I could not find an answer in the member data.", confidence=0.0, citations=[], reasoning="no_context")

    @classmethod
    def from_answer(
        cls,
        question: str,
        answer: str,
        *,
        confidence: float,
        documents: Iterable[Document],
        reasoning: str,
        guardrails: dict[str, bool] | None = None,
    ) -> "AnswerResult":
        citations = [Citation.from_document(doc) for doc in documents]
        return cls(
            question=question,
            answer=answer.strip(),
            confidence=max(0.0, min(confidence, 1.0)),
            citations=citations,
            reasoning=reasoning,
            guardrails=guardrails,
        )


__all__ = ["AnswerResult", "Citation"]
