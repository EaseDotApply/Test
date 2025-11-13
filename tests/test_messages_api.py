from __future__ import annotations

from datetime import datetime, timezone

import pytest
import respx
from httpx import Response

from app.clients.messages_api import MessagesAPIClient
from app.core.config import AppSettings


@pytest.mark.asyncio
@respx.mock
async def test_fetch_messages_handles_pagination(temp_settings: AppSettings) -> None:
    base_url = str(temp_settings.messages_api_url)

    first_page = {
        "total": 2,
        "page": 1,
        "page_size": 1,
        "next_page": 2,
        "items": [
            {
                "id": "msg-1",
                "user_id": "user-1",
                "user_name": "Layla",
                "timestamp": datetime(2024, 5, 1, tzinfo=timezone.utc).isoformat(),
                "message": "Planning my trip to London in June!",
            }
        ],
    }
    second_page = {
        "total": 2,
        "page": 2,
        "page_size": 1,
        "items": [
            {
                "id": "msg-2",
                "user_id": "user-2",
                "user_name": "Vikram",
                "timestamp": datetime(2024, 5, 2, tzinfo=timezone.utc).isoformat(),
                "message": "I just bought a second car yesterday.",
            }
        ],
    }

    # Mock both with and without trailing slash (client normalizes)
    base_url_with_slash = base_url.rstrip("/") + "/"
    for url in [base_url, base_url_with_slash]:
        respx.get(url, params={"page": 1, "page_size": temp_settings.request_page_size}).mock(
            return_value=Response(200, json=first_page)
        )
        respx.get(url, params={"page": 2, "page_size": temp_settings.request_page_size}).mock(
            return_value=Response(200, json=second_page)
        )
        respx.get(url, params={"page": 3, "page_size": temp_settings.request_page_size}).mock(
            return_value=Response(200, json={"total": 2, "page": 3, "page_size": 1, "items": []})
        )

    client = MessagesAPIClient(settings=temp_settings)
    bundle = await client.fetch_messages(force_refresh=True)

    assert len(bundle.messages) == 2
    assert bundle.total == 2
    assert bundle.messages[0].user_name == "Layla"
    assert bundle.messages[1].message.startswith("I just bought")
