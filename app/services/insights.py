"""Generate data quality insights from processed messages."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

import pandas as pd

from app.clients.messages_api import MessagesAPIClient
from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.domain.processed import EntitySpan, ProcessedBundle, ProcessedMessage
from app.pipelines.preprocess import MessagePreprocessor
from app.pipelines.processed_repository import ProcessedRepository


class InsightsService:
    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        preprocessor: MessagePreprocessor | None = None,
        repository: ProcessedRepository | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._preprocessor = preprocessor or MessagePreprocessor(self._settings)
        self._repository = repository or ProcessedRepository(self._settings)

    async def generate(self, *, force_refresh: bool = False) -> dict[str, object]:
        processed = await self._ensure_processed(force_refresh=force_refresh)
        frame = self._to_dataframe(processed)
        highlights = self._build_highlights(processed, frame)
        anomalies = self._detect_anomalies(processed, frame)
        summary = {
            "highlights": highlights,
            "anomalies": anomalies,
            "generated_at": datetime.now(tz=timezone.utc),
        }
        self._write_markdown(summary, frame)
        return summary

    async def _ensure_processed(self, *, force_refresh: bool) -> ProcessedBundle:
        if force_refresh:
            return await self._preprocessor.run(force_refresh=True)
        cached = self._repository.load()
        if cached:
            return cached
        return await self._preprocessor.run(force_refresh=False)

    def _to_dataframe(self, bundle: ProcessedBundle) -> pd.DataFrame:
        records = [
            {
                "id": message.id,
                "user_id": message.user_id,
                "user_name": message.user_name,
                "timestamp": message.timestamp_utc,
                "token_count": message.token_count,
                "message_clean": message.message_clean,
                "entities": [entity.model_dump() for entity in message.entities],
            }
            for message in bundle.messages
        ]
        return pd.DataFrame.from_records(records)

    def _build_highlights(self, bundle: ProcessedBundle, frame: pd.DataFrame) -> list[str]:
        highlights: list[str] = []
        if not frame.empty:
            top_members = frame["user_name"].value_counts().head(3)
            if not top_members.empty:
                formatted = ", ".join(
                    f"{member} ({count} messages)" for member, count in top_members.items()
                )
                highlights.append(f"Most active members: {formatted}.")

            avg_tokens = mean(message.token_count for message in bundle.messages)
            highlights.append(f"Average message length: {avg_tokens:.1f} tokens.")

            entities = self._flatten_entities(bundle.messages)
            if entities:
                city_counts = Counter(
                    entity.text for entity in entities if entity.label.upper() in {"GPE", "LOC"}
                )
                if city_counts:
                    top_cities = ", ".join(
                        f"{name} ({count})" for name, count in city_counts.most_common(5)
                    )
                    highlights.append(f"Top mentioned locations: {top_cities}.")
        return highlights

    def _detect_anomalies(self, bundle: ProcessedBundle, frame: pd.DataFrame) -> list[str]:
        anomalies: list[str] = []
        if frame.empty:
            return ["No messages available to analyse."]

        duplicate_ids = frame[frame.duplicated(subset="id", keep=False)]["id"].unique()
        if duplicate_ids.size > 0:
            anomalies.append(f"Duplicate message IDs detected: {', '.join(map(str, duplicate_ids[:5]))}.")

        future_messages = frame[frame["timestamp"] > datetime.now(tz=timezone.utc)]
        if not future_messages.empty:
            anomalies.append(
                f"{len(future_messages)} messages are timestamped in the future; latest is {future_messages['timestamp'].max().isoformat()}"
            )

        long_messages = frame[frame["token_count"] > frame["token_count"].quantile(0.99)]
        if not long_messages.empty:
            sample = long_messages.iloc[0]
            anomalies.append(
                "Unusually long message detected for "
                f"{sample['user_name']} ({sample['token_count']} tokens)."
            )

        empty_messages = frame[frame["message_clean"].str.len() == 0]
        if not empty_messages.empty:
            anomalies.append(f"{len(empty_messages)} messages are empty after cleaning.")

        return anomalies

    def _write_markdown(self, summary: dict[str, object], frame: pd.DataFrame) -> None:
        reports_dir = self._settings.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / "insights.md"
        generated_at: datetime = summary["generated_at"]  # type: ignore[assignment]

        highlight_lines = summary["highlights"] if summary["highlights"] else ["No highlights derived."]
        anomaly_lines = summary["anomalies"] if summary["anomalies"] else ["No anomalies detected."]

        md_content = [
            "# Data Insights",
            f"Generated: {generated_at.isoformat()}",
            "",
            "## Highlights",
            *(f"- {line}" for line in highlight_lines),
            "",
            "## Anomalies",
            *(f"- {line}" for line in anomaly_lines),
            "",
            f"Total messages analysed: {len(frame)}",
        ]
        md_path.write_text("\n".join(md_content))
        self._logger.info("insights.report_written", path=str(md_path))

    def _flatten_entities(self, messages: Iterable[ProcessedMessage]) -> list[EntitySpan]:
        entities: list[EntitySpan] = []
        for message in messages:
            entities.extend(message.entities)
        return entities


__all__ = ["InsightsService"]
