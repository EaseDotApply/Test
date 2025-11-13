"""Async client for the public member messages API."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx

from app.clients.messages_repository import MessagesRepository
from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.domain.messages import Message, MessagesBundle, MessagesPage


class MessagesAPIClient:
    """Fetches member messages with retry and caching support."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        repository: MessagesRepository | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._repository = repository or MessagesRepository(self._settings)

    async def fetch_messages(self, force_refresh: bool = False) -> MessagesBundle:
        """Fetch all messages from the upstream API.

        Utilises conditional requests with ETags to avoid unnecessary downloads. Results are
        cached locally for reuse by downstream pipelines.
        """

        cached_bundle = self._repository.load()
        if cached_bundle and not force_refresh:
            self._logger.info("messages.cached", count=len(cached_bundle), etag=cached_bundle.etag)

        headers: dict[str, str] = {"User-Agent": "november-qa/0.1"}
        if cached_bundle and cached_bundle.etag and not force_refresh:
            headers["If-None-Match"] = cached_bundle.etag

        async with httpx.AsyncClient(
            timeout=self._settings.request_timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = await self._request(client, headers=headers)
            if response.status_code == httpx.codes.NOT_MODIFIED and cached_bundle:
                self._logger.info("messages.not_modified", etag=cached_bundle.etag)
                return cached_bundle

            etag = response.headers.get("ETag")
            messages, total = await self._collect_all(client, response)

        bundle = MessagesBundle.from_messages(
            messages=messages,
            total=total,
            etag=etag,
            fetched_at=datetime.now(tz=timezone.utc),
        )
        self._repository.save(bundle)
        return bundle

    async def _request(self, client: httpx.AsyncClient, headers: dict[str, str]) -> httpx.Response:
        attempts = self._settings.request_retry_attempts
        base_url = str(self._settings.messages_api_url)
        params = {"page": 1, "page_size": self._settings.request_page_size}

        for attempt in range(1, attempts + 1):
            try:
                response = await client.get(base_url, params=params, headers=headers)
                if response.status_code == httpx.codes.NOT_MODIFIED:
                    return response
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:  # noqa: PERF203
                retryable = isinstance(exc, httpx.RequestError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code >= 500
                )
                if attempt >= attempts or not retryable:
                    self._logger.error("messages.request_failed", attempt=attempt, error=str(exc))
                    raise
                sleep_duration = min(2 ** (attempt - 1), 10)
                self._logger.warning(
                    "messages.request_retry", attempt=attempt, wait_seconds=sleep_duration, error=str(exc)
                )
                await asyncio.sleep(sleep_duration)
        raise RuntimeError("Unreachable retry loop exit")

    async def _collect_all(
        self,
        client: httpx.AsyncClient,
        first_response: httpx.Response,
    ) -> tuple[list[Message], int]:
        body = first_response.json()
        page = MessagesPage.model_validate(body)
        messages = list(self._parse_messages(page.items))
        total = page.total or len(messages)

        next_page = self._next_page_hint(page, len(messages))
        next_url = page.next_url
        headers: dict[str, str] = {"User-Agent": "november-qa/0.1"}

        while True:
            if next_url:
                response = await self._paged_request(client, url=str(next_url), headers=headers)
            elif next_page is not None:
                params = {"page": next_page, "page_size": self._settings.request_page_size}
                response = await self._paged_request(
                    client,
                    url=str(self._settings.messages_api_url),
                    headers=headers,
                    params=params,
                )
            else:
                break

            payload = response.json()
            page = MessagesPage.model_validate(payload)
            messages.extend(self._parse_messages(page.items))
            total = page.total or total
            next_page = self._next_page_hint(page, len(page.items))
            next_url = page.next_url

        self._logger.info("messages.fetched", count=len(messages), total=total)
        return messages, total

    async def _paged_request(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        # Normalize URL to handle redirects
        if url.endswith("/messages") and not url.endswith("/messages/"):
            url = url + "/"
        for attempt in range(1, self._settings.request_retry_attempts + 1):
            try:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:  # noqa: PERF203
                retryable = isinstance(exc, httpx.RequestError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code >= 500
                )
                if attempt == self._settings.request_retry_attempts or not retryable:
                    self._logger.error("messages.page_failed", url=url, attempt=attempt, error=str(exc))
                    raise
                await asyncio.sleep(min(2 ** (attempt - 1), 10))
        raise RuntimeError("Unreachable retry loop exit")

    def _next_page_hint(self, page: MessagesPage, items_len: int) -> int | None:
        if page.next_page is not None:
            return page.next_page
        if page.page is not None and page.page_size and items_len >= page.page_size:
            return page.page + 1
        if page.page is None and items_len >= self._settings.request_page_size:
            return (page.page or 1) + 1
        return None

    def _parse_messages(self, items: Iterable[Message]) -> list[Message]:
        return [Message.model_validate(item) if not isinstance(item, Message) else item for item in items]


__all__ = ["MessagesAPIClient"]
