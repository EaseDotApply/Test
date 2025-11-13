"""Persistence helpers for processed message artifacts."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from app.core.config import AppSettings
from app.core.logging import get_logger
from app.domain.processed import EntitySpan, ProcessedBundle, ProcessedMessage


class ProcessedRepository:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._logger = get_logger(self.__class__.__name__)
        self._path = settings.processed_messages_path
        self._meta_path = self._path.with_suffix(".meta.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, bundle: ProcessedBundle) -> None:
        records = [self._to_record(message) for message in bundle.messages]
        frame = pd.DataFrame.from_records(records)
        frame.to_parquet(self._path, index=False)

        metadata = {
            "processed_at": bundle.processed_at.isoformat(),
            "source_total": bundle.source_total,
            "processed_total": len(bundle.messages),
        }
        self._meta_path.write_text(json.dumps(metadata, indent=2))
        self._logger.info(
            "processed.save",
            path=str(self._path),
            processed=len(bundle.messages),
            source_total=bundle.source_total,
        )

    def load(self) -> ProcessedBundle | None:
        if not self._path.exists() or not self._meta_path.exists():
            return None
        try:
            frame = pd.read_parquet(self._path)
            metadata = json.loads(self._meta_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            self._logger.warning("processed.load_failed", error=str(exc))
            return None

        messages = [self._from_record(record) for record in frame.to_dict(orient="records")]
        processed_at = datetime.fromisoformat(metadata["processed_at"])
        source_total = metadata.get("source_total", len(messages))
        return ProcessedBundle(messages=messages, processed_at=processed_at, source_total=source_total)

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()
        self._logger.info("processed.cleared")

    def _to_record(self, message: ProcessedMessage) -> dict[str, object]:
        return {
            "id": message.id,
            "user_id": message.user_id,
            "user_name": message.user_name,
            "timestamp_utc": message.timestamp_utc,
            "message_original": message.message_original,
            "message_clean": message.message_clean,
            "temporal_key": message.temporal_key,
            "token_count": message.token_count,
            "entities": json.dumps([entity.model_dump() for entity in message.entities]),
        }

    def _from_record(self, record: dict[str, object]) -> ProcessedMessage:
        raw_entities = json.loads(record.get("entities", "[]"))
        entities = [EntitySpan.model_validate(entity) for entity in raw_entities]
        return ProcessedMessage(
            id=str(record["id"]),
            user_id=str(record["user_id"]),
            user_name=str(record["user_name"]),
            timestamp_utc=pd.to_datetime(record["timestamp_utc"]).to_pydatetime(),
            message_original=str(record["message_original"]),
            message_clean=str(record["message_clean"]),
            temporal_key=str(record["temporal_key"]),
            token_count=int(record["token_count"]),
            entities=entities,
        )


__all__ = ["ProcessedRepository"]
