"""Models describing enriched, preprocessed messages."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from pydantic import BaseModel, Field


class EntitySpan(BaseModel):
    """Named entity extracted from user messages."""

    text: str = Field(description="Surface form of the entity span.")
    label: str = Field(description="NER label, e.g., PERSON or GPE.")


class ProcessedMessage(BaseModel):
    """Enriched representation of a message after preprocessing."""

    id: str
    user_id: str
    user_name: str
    timestamp_utc: datetime
    message_original: str
    message_clean: str
    temporal_key: str = Field(description="Date key (YYYY-MM-DD) for grouping.")
    token_count: int = Field(description="Number of whitespace-delimited tokens in the cleaned text.")
    entities: list[EntitySpan] = Field(default_factory=list, description="Named entities extracted from the message.")

    @classmethod
    def from_components(
        cls,
        *,
        id: str,
        user_id: str,
        user_name: str,
        timestamp_utc: datetime,
        message_original: str,
        message_clean: str,
        token_count: int,
        entities: Sequence[EntitySpan],
    ) -> "ProcessedMessage":
        temporal_key = timestamp_utc.date().isoformat()
        return cls(
            id=id,
            user_id=user_id,
            user_name=user_name,
            timestamp_utc=timestamp_utc,
            message_original=message_original,
            message_clean=message_clean,
            token_count=token_count,
            temporal_key=temporal_key,
            entities=list(entities),
        )


class ProcessedBundle(BaseModel):
    """Collection of processed messages."""

    messages: list[ProcessedMessage] = Field(default_factory=list)
    source_total: int = Field(description="Total messages seen in the source bundle.")
    processed_at: datetime = Field(description="Timestamp when preprocessing ran.")

    def __len__(self) -> int:
        return len(self.messages)


__all__ = ["EntitySpan", "ProcessedMessage", "ProcessedBundle"]
