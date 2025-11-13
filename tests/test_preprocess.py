from __future__ import annotations

from datetime import datetime, timezone

import pytest

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment, misc]

from app.domain.messages import Message
from app.pipelines.preprocess import MessagePreprocessor

pytestmark = pytest.mark.skipif(spacy is None, reason="spacy not installed")


def test_preprocess_deduplicates_and_cleans(temp_settings) -> None:
    nlp = spacy.blank("en")
    preprocessor = MessagePreprocessor(settings=temp_settings, nlp=nlp)
    messages = [
        Message(
            id="1",
            user_id="user-1",
            user_name="Layla",
            timestamp=datetime(2024, 5, 1, tzinfo=timezone.utc),
            message="Planning  trip\n to London!",
        ),
        Message(
            id="1",
            user_id="user-1",
            user_name="Layla",
            timestamp=datetime(2024, 5, 1, tzinfo=timezone.utc),
            message="Planning  trip\n to London!",
        ),
        Message(
            id="2",
            user_id="user-2",
            user_name="Vikram",
            timestamp=datetime(2024, 5, 2, tzinfo=timezone.utc),
            message="Bought   another car yesterday",
        ),
    ]

    processed = preprocessor._process(messages, source_total=len(messages))

    assert len(processed.messages) == 2
    clean_texts = [message.message_clean for message in processed.messages]
    assert "Planning trip to London!" in clean_texts
    assert any(message.token_count == 4 for message in processed.messages)
