"""Domain models representing upstream message data."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from pydantic import BaseModel, Field, HttpUrl


class Message(BaseModel):
    """Single user-authored message."""

    id: str = Field(description="Unique identifier for the message.")
    user_id: str = Field(description="Unique identifier for the member who authored the message.")
    user_name: str = Field(description="Display name for the member.")
    timestamp: datetime = Field(description="Timestamp when the message was recorded.")
    message: str = Field(description="Natural-language contents of the user's message.")

    @property
    def summary(self) -> str:
        """Create a concise string representation for logging and debugging."""

        return f"{self.user_name} ({self.user_id}): {self.message[:64]}"


class MessagesPage(BaseModel):
    """Subset of messages returned from the upstream API."""

    total: int = Field(description="Total messages available upstream.")
    items: list[Message] = Field(default_factory=list)
    page: int | None = Field(default=None, description="Current page number, if provided by upstream.")
    page_size: int | None = Field(default=None, description="Page size used in the response.")
    next_page: int | None = Field(default=None, description="Next page index if provided by upstream.")
    next_url: HttpUrl | None = Field(default=None, description="Absolute URL for the next page, if supplied.")


class MessagesBundle(BaseModel):
    """Aggregate of all messages plus metadata for caching."""

    total: int = Field(description="Total number of messages fetched.")
    messages: list[Message] = Field(default_factory=list)
    etag: str | None = Field(default=None, description="Upstream ETag for cache validation.")
    fetched_at: datetime = Field(description="UTC timestamp for when data was fetched.")

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterable[Message]:
        return iter(self.messages)

    @classmethod
    def from_messages(cls, messages: Sequence[Message], total: int, etag: str | None, fetched_at: datetime) -> "MessagesBundle":
        return cls(total=total, messages=list(messages), etag=etag, fetched_at=fetched_at)


__all__ = ["Message", "MessagesPage", "MessagesBundle"]
