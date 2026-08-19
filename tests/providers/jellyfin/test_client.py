"""Tests for the internal Jellyfin client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, call

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

    def __init__(self, body: bytes = b"") -> None:
        """Initialize fake response content."""
        self.content = FakeContent()
        self.body = body

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

    async def read(self) -> bytes:
        """Return the configured response body."""
        return self.body


class FakeHttpSession:
    """Capture Jellyfin GET requests."""

    def __init__(self, response_body: bytes = b"") -> None:
        """Initialize request capture."""
        self.request: dict[str, Any] | None = None
        self.response_body = response_body

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """Capture a GET request and return a fake response."""
        self.request = {
            "url": url,
            **kwargs,
        }
        return FakeResponse(self.response_body)


def create_client(session: Any | None = None) -> JellyfinClient:
    """Create an authenticated client for tests."""
    return JellyfinClient(
        session=session or MagicMock(),
        base_url="https://music.example",
        app_name="Music Assistant",
        app_version="test",
        device_name="test-device",
        device_id="device-id",
        user_id="user-id",
        access_token="secret-token",
    )


@pytest.mark.asyncio
async def test_iter_tracks_paginates_until_total_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item iterators must request subsequent pages until all reported items are returned."""
    client = create_client()
    request = AsyncMock(
        side_effect=[
            {
                "Items": [{"Id": "track-1"}, {"Id": "track-2"}],
                "TotalRecordCount": 3,
            },
            {
                "Items": [{"Id": "track-3"}],
                "TotalRecordCount": 3,
            },
        ]
    )
    monkeypatch.setattr(JellyfinClient, "_items_request", request)

    tracks = [track async for track in client.iter_tracks("library-id", ("Path",))]

    assert tracks == [
        {"Id": "track-1"},
        {"Id": "track-2"},
        {"Id": "track-3"},
    ]
    assert request.await_args_list == [
        call(
            endpoint="/Items",
            item_type="Audio",
            parent_id="library-id",
            fields=("Path",),
            enable_user_data=True,
            start_index=0,
            limit=100,
        ),
        call(
            endpoint="/Items",
            item_type="Audio",
            parent_id="library-id",
            fields=("Path",),
            enable_user_data=True,
            start_index=2,
            limit=100,
        ),
    ]


@pytest.mark.asyncio
async def test_get_playlist_tracks_requests_requested_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Playlist paging must preserve start index, limit, fields, and non-recursive semantics."""
    client = create_client()
    get_json = AsyncMock(return_value={"Items": [{"Id": "track-101"}]})
    monkeypatch.setattr(JellyfinClient, "_get_json", get_json)

    tracks = await client.get_playlist_tracks(
        "playlist-id",
        ("Path", "MediaSources"),
        start_index=100,
        limit=50,
    )

    assert tracks == [{"Id": "track-101"}]
    get_json.assert_awaited_once_with(
        "/Playlists/playlist-id/Items",
        {
            "recursive": "false",
            "fields": "Path,MediaSources",
            "enableUserData": "true",
            "startIndex": "100",
            "limit": "50",
        },
    )


@pytest.mark.asyncio
async def test_get_similar_tracks_requests_limit_and_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Similar-track requests must use the dedicated endpoint and requested projection."""
    client = create_client()
    get_json = AsyncMock(
        return_value={
            "Items": [
                {"Id": "similar-1"},
                "not-an-item",
            ]
        }
    )
    monkeypatch.setattr(JellyfinClient, "_get_json", get_json)

    tracks = await client.get_similar_tracks(
        "track-id",
        limit=12,
        fields=("Path", "Genres"),
    )

    assert tracks == [{"Id": "similar-1"}]
    get_json.assert_awaited_once_with(
        "/Items/track-id/Similar",
        {
            "limit": "12",
            "fields": "Path,Genres",
        },
    )


@pytest.mark.asyncio
async def test_resolve_image_uses_authenticated_request() -> None:
    """Artwork resolution must fetch the opaque path with header-based authentication."""
    http_session = FakeHttpSession(b"image-bytes")
    client = create_client(http_session)

    result = await client.resolve_image("jellyfin://image/item-id/Backdrop/2")

    assert result == b"image-bytes"
    assert http_session.request is not None
    assert http_session.request["url"] == (
        "https://music.example/Items/item-id/Images/Backdrop/2"
    )
    assert http_session.request["headers"] == client.headers
    assert http_session.request["ssl"] is True
    assert http_session.request["raise_for_status"] is True
    assert "params" not in http_session.request


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
    client = create_client()

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
    client = create_client()

    with pytest.raises(FileNotFoundError):
        client._parse_image_path("https://attacker.example/Items/item-id/Images/Primary")
