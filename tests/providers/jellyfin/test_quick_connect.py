"""Tests for Jellyfin Quick Connect authentication."""

from __future__ import annotations

from collections import deque
from types import TracebackType
from typing import Any, Self

import pytest

from music_assistant.providers.jellyfin.client import JellyfinClient
from music_assistant.providers.jellyfin.quick_connect import (
    QuickConnectAuth,
    QuickConnectRequest,
    initiate_quick_connect,
    wait_for_quick_connect,
)


class JsonResponse:
    """Minimal async response returning a JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Initialize the response payload."""
        self.payload = payload

    async def __aenter__(self) -> Self:
        """Enter the async response context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the async response context."""
        return

    async def json(self) -> dict[str, Any]:
        """Return the configured JSON payload."""
        return self.payload


class QuickConnectSession:
    """Capture Quick Connect HTTP calls and return queued JSON responses."""

    def __init__(
        self,
        *,
        get_payloads: list[dict[str, Any]] | None = None,
        post_payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize queued responses and request capture."""
        self.get_payloads = deque(get_payloads or [])
        self.post_payloads = deque(post_payloads or [])
        self.get_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> JsonResponse:
        """Capture a GET request and return the next queued response."""
        self.get_calls.append({"url": url, **kwargs})
        return JsonResponse(self.get_payloads.popleft())

    def post(self, url: str, **kwargs: Any) -> JsonResponse:
        """Capture a POST request and return the next queued response."""
        self.post_calls.append({"url": url, **kwargs})
        return JsonResponse(self.post_payloads.popleft())


def create_client(session: QuickConnectSession) -> JellyfinClient:
    """Create the unauthenticated client used to initiate Quick Connect."""
    return JellyfinClient(
        session=session,  # type: ignore[arg-type]
        base_url="https://jellyfin.example",
        app_name="Music Assistant",
        app_version="test",
        device_name="test-device",
        device_id="device-id",
        user_id="",
        access_token="",
        verify_ssl=True,
    )


@pytest.mark.asyncio
async def test_initiate_quick_connect_sends_device_authorization_header() -> None:
    """Quick Connect initiation must identify the client and device without a token."""
    session = QuickConnectSession(
        post_payloads=[{"Secret": "secret-value", "Code": "123456"}]
    )
    client = create_client(session)

    result = await initiate_quick_connect(client)

    assert result == QuickConnectRequest(secret="secret-value", code="123456")
    assert len(session.post_calls) == 1
    request = session.post_calls[0]
    assert request["url"] == "https://jellyfin.example/QuickConnect/Initiate"
    assert request["json"] == {}
    assert request["ssl"] is True
    assert request["raise_for_status"] is True
    assert request["headers"]["Authorization"] == (
        'MediaBrowser Client="Music Assistant", Device="test-device", '
        'DeviceId="device-id", Version="test"'
    )
    assert "Token=" not in request["headers"]["Authorization"]


@pytest.mark.asyncio
async def test_wait_for_quick_connect_polls_then_exchanges_secret() -> None:
    """Approved Quick Connect requests must be exchanged for a persistent access token."""
    session = QuickConnectSession(
        get_payloads=[
            {"Authenticated": False},
            {"Authenticated": True},
        ],
        post_payloads=[
            {
                "User": {"Id": "user-id", "Name": "music-user"},
                "AccessToken": "access-token",
            }
        ],
    )
    client = create_client(session)

    result = await wait_for_quick_connect(client, "secret-value", poll_interval=0)

    assert result == QuickConnectAuth(
        user_id="user-id",
        username="music-user",
        access_token="access-token",
    )
    assert len(session.get_calls) == 2
    for request in session.get_calls:
        assert request["url"] == "https://jellyfin.example/QuickConnect/Connect"
        assert request["params"] == {"secret": "secret-value"}
        assert request["headers"] == client.headers
        assert request["ssl"] is True
        assert request["raise_for_status"] is True

    assert len(session.post_calls) == 1
    request = session.post_calls[0]
    assert request["url"] == (
        "https://jellyfin.example/Users/AuthenticateWithQuickConnect"
    )
    assert request["json"] == {"Secret": "secret-value"}
    assert request["headers"] == client.headers
    assert request["ssl"] is True
    assert request["raise_for_status"] is True
