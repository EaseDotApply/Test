"""Preprocessing pipeline to enrich raw messages."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Iterable, Sequence
from unicodedata import normalize

try:
    import spacy
    from spacy.language import Language
except ImportError:
    spacy = None  # type: ignore[assignment, misc]
    Language = None  # type: ignore[assignment, misc]

from app.clients.messages_api import MessagesAPIClient
from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.domain.messages import Message
from app.domain.processed import EntitySpan, ProcessedBundle, ProcessedMessage
from app.pipelines.processed_repository import ProcessedRepository


class MessagePreprocessor:
    """Transforms raw messages into enriched artefacts for retrieval."""

    CLEAN_RE = re.compile(r"\s+")

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        client: MessagesAPIClient | None = None,
        repository: ProcessedRepository | None = None,
        nlp: Any = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or MessagesAPIClient(self._settings)
        self._repository = repository or ProcessedRepository(self._settings)
        self._logger = get_logger(self.__class__.__name__)
        self._nlp_override = nlp

    async def run(self, *, force_refresh: bool = False) -> ProcessedBundle:
        """Fetch messages and preprocess them, returning the processed bundle."""

        raw_bundle = await self._client.fetch_messages(force_refresh=force_refresh)
        cached = self._repository.load()
        if cached and not force_refresh and len(cached.messages) >= len(raw_bundle.messages):
            self._logger.info("preprocess.cached", count=len(cached.messages))
            return cached

        processed_bundle = self._process(raw_bundle.messages, source_total=raw_bundle.total)
        self._repository.save(processed_bundle)
        return processed_bundle

    def _process(self, messages: Sequence[Message], *, source_total: int) -> ProcessedBundle:
        deduped = self._deduplicate(messages)
        enriched = [self._transform(message) for message in deduped]
        processed = ProcessedBundle(
            messages=enriched,
            source_total=source_total,
            processed_at=datetime.now(tz=timezone.utc),
        )
        self._logger.info(
            "preprocess.completed",
            processed=len(enriched),
            source_total=source_total,
            deduped=len(deduped),
        )
        return processed

    def _deduplicate(self, messages: Sequence[Message]) -> list[Message]:
        seen: dict[str, Message] = {}
        for message in sorted(messages, key=lambda item: item.timestamp):
            seen[message.id] = message
        deduped = list(seen.values())
        self._logger.info("preprocess.deduplicate", before=len(messages), after=len(deduped))
        return deduped

    def _transform(self, message: Message) -> ProcessedMessage:
        timestamp = self._normalise_timestamp(message.timestamp)
        clean_text = self._clean_text(message.message)
        token_count = self._count_tokens(clean_text)
        entities = list(self._extract_entities(message.message))
        return ProcessedMessage.from_components(
            id=message.id,
            user_id=message.user_id,
            user_name=message.user_name,
            timestamp_utc=timestamp,
            message_original=message.message,
            message_clean=clean_text,
            token_count=token_count,
            entities=entities,
        )

    def _normalise_timestamp(self, timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _clean_text(self, content: str) -> str:
        collapsed = self.CLEAN_RE.sub(" ", normalize("NFKC", content)).strip()
        return collapsed

    def _count_tokens(self, content: str) -> int:
        return len([token for token in content.split(" ") if token])

    def _extract_entities(self, content: str) -> Iterable[EntitySpan]:
        if spacy is None:
            return
        try:
            doc = self.nlp(content)
            for ent in doc.ents:
                yield EntitySpan(text=ent.text, label=ent.label_)
        except Exception:
            return

    @cached_property
    def nlp(self) -> Any:
        if spacy is None:
            return None
        if self._nlp_override is not None:
            return self._nlp_override
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            self._logger.warning(
                "preprocess.spacy_fallback", message="Falling back to blank English spaCy model."
            )
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp


__all__ = ["MessagePreprocessor"]
