from __future__ import annotations

from http import HTTPStatus


def test_health_endpoint(fastapi_client) -> None:
    response = fastapi_client.get("/api/health")
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["status"] == "ok"


def test_ask_endpoint(fastapi_client) -> None:
    response = fastapi_client.post(
        "/api/ask",
        json={"question": "When is Layla planning her trip to London?"},
    )
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert "answer" in payload
    assert isinstance(payload["answer"], str)
    assert len(payload) == 1  # Only "answer" field per requirement


def test_refresh_endpoint(fastapi_client) -> None:
    response = fastapi_client.post("/api/refresh")
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["documents_indexed"] == 3


def test_insights_endpoint(fastapi_client) -> None:
    response = fastapi_client.get("/api/insights")
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["highlights"]
