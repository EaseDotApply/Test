"""Persistent repository for caching member messages."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.core.config import AppSettings
from app.core.logging import get_logger
from app.domain.messages import Message, MessagesBundle


class MessagesRepository:
    """Handles persistence of message datasets to local storage."""

    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._logger = get_logger(self.__class__.__name__)
        self._raw_path = settings.raw_messages_path
        self._meta_path = self._raw_path.with_suffix(".meta.json")
        self._raw_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> MessagesBundle | None:
        """Load cached messages bundle if present."""

        if not self._raw_path.exists() or not self._meta_path.exists():
            return None

        try:
            metadata = json.loads(self._meta_path.read_text())
            fetched_at = datetime.fromisoformat(metadata["fetched_at"])
            etag = metadata.get("etag")
            total = metadata.get("total", 0)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self._logger.warning("cache.metadata_invalid", error=str(exc))
            return None

        try:
            frame = pd.read_parquet(self._raw_path)
        except FileNotFoundError:
            return None

        records: Iterable[Message] = (
            Message.model_validate(record) for record in frame.to_dict(orient="records")
        )
        messages = list(records)
        if not messages:
            return None

        self._logger.info("cache.load", count=len(messages), etag=etag)
        return MessagesBundle.from_messages(messages, total=total or len(messages), etag=etag, fetched_at=fetched_at)

    def save(self, bundle: MessagesBundle) -> None:
        """Persist messages and metadata to disk."""

        records = [message.model_dump() for message in bundle.messages]
        frame = pd.DataFrame.from_records(records)
        frame.sort_values(by="timestamp", inplace=True)
        frame.to_parquet(self._raw_path, index=False)

        metadata = {
            "etag": bundle.etag,
            "total": bundle.total,
            "fetched_at": bundle.fetched_at.isoformat(),
        }
        self._meta_path.write_text(json.dumps(metadata, indent=2))
        self._logger.info("cache.save", path=str(self._raw_path), count=len(bundle.messages))

    def clear(self) -> None:
        """Remove cached datasets."""

        if self._raw_path.exists():
            self._raw_path.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()
        self._logger.info("cache.cleared")

    def is_fresh(self, etag: str | None) -> bool:
        """Return True if cache etag matches the provided value."""

        if etag is None or not self._meta_path.exists():
            return False
        try:
            metadata = json.loads(self._meta_path.read_text())
        except json.JSONDecodeError:
            return False
        return metadata.get("etag") == etag


__all__ = ["MessagesRepository"]
