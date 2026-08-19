"""Tests for the internal Jellyfin client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self
from unittest.mock import MagicMock

import pytest

from music_assistant.providers.jellyfin.client import JellyfinClient


class FakeContent:
    """Fake aiohttp response content."""

    async def iter_chunked(self, chunk_size: int) -> AsyncIterator[bytes]:
        """Yield test audio chunks."""
        assert chunk_size == 64 * 1024
        yield b"first"
        yield b"second"


class FakeResponse:
    """Fake aiohttp response."""

    def __init__(self) -> None:
        """Initialize fake response content."""
        self.content = FakeContent()

    async def __aenter__(self) -> Self:
        """Enter async response context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit async response context."""
        return


class FakeHttpSession:
    """Capture Jellyfin stream requests."""

    def __init__(self) -> None:
        """Initialize request capture."""
        self.request: dict[str, Any] | None = None

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """Capture a GET request and return a fake response."""
        self.request = {
            "url": url,
            **kwargs,
        }
        return FakeResponse()


@pytest.mark.asyncio
async def test_stream_audio_uses_authorization_header() -> None:
    """Audio streaming must authenticate by header, never by query-string token."""
    http_session = FakeHttpSession()

    client = JellyfinClient(
        session=http_session,  # type: ignore[arg-type]
        base_url="http://jellyfin.example",
        app_name="Music Assistant",
        app_version="test",
        device_name="test-device",
        device_id="device-id",
        user_id="user-id",
        access_token="secret-token",
        verify_ssl=True,
    )

    chunks = [
        chunk
        async for chunk in client.stream_audio(
            "track-id",
            container="flac,mp3",
            seek_position=42,
        )
    ]

    assert chunks == [b"first", b"second"]

    assert http_session.request is not None
    assert http_session.request["url"] == ("http://jellyfin.example/Audio/track-id/universal")

    params = http_session.request["params"]
    assert params["userId"] == "user-id"
    assert params["deviceId"] == "device-id"
    assert params["container"] == "flac,mp3"
    assert params["startTimeTicks"] == "420000000"
    assert "api_key" not in params

    headers = http_session.request["headers"]
    assert headers["Authorization"] == (
        'MediaBrowser Client="Music Assistant", Device="test-device", '
        'DeviceId="device-id", Version="test", Token="secret-token"'
    )


def test_parse_current_image_path() -> None:
    """Current opaque artwork paths are parsed correctly."""
    client = JellyfinClient(
        session=MagicMock(),
        base_url="https://music.example",
        app_name="Music Assistant",
        app_version="test",
        device_name="test-device",
        device_id="device-id",
        user_id="user-id",
        access_token="secret-token",
    )

    assert client._parse_image_path("jellyfin://image/item-id/Backdrop/0") == (
        "item-id",
        "Backdrop",
        "0",
    )


def test_parse_legacy_image_url_discards_api_key() -> None:
    """Legacy artwork URLs are accepted without reusing their query token."""
    client = JellyfinClient(
        session=MagicMock(),
        base_url="https://music.example",
        app_name="Music Assistant",
        app_version="test",
        device_name="test-device",
        device_id="device-id",
        user_id="user-id",
        access_token="new-token",
    )

    assert client._parse_image_path(
        "https://music.example/Items/item-id/Images/Primary?api_key=old-secret"
    ) == ("item-id", "Primary", None)


def test_reject_legacy_image_url_from_other_host() -> None:
    """Legacy artwork URLs from other servers must not be accepted."""
    client = JellyfinClient(
        session=MagicMock(),
        base_url="https://music.example",
        app_name="Music Assistant",
        app_version="test",
        device_name="test-device",
        device_id="device-id",
        user_id="user-id",
        access_token="secret-token",
    )

    with pytest.raises(FileNotFoundError):
        client._parse_image_path("https://attacker.example/Items/item-id/Images/Primary")
