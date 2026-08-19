"""Tests for Jellyfin play-state synchronization."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType

from music_assistant.providers import jellyfin as jellyfin_provider
from music_assistant.providers.jellyfin import JellyfinProvider
from music_assistant.providers.jellyfin.client import JellyfinClient
from music_assistant.providers.jellyfin.playstate import mark_played


class FakeResponse:
    """Minimal async HTTP response context."""

    async def __aenter__(self) -> Self:
        """Enter the response context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the response context."""
        return


class FakeHttpSession:
    """Capture Jellyfin POST requests."""

    def __init__(self) -> None:
        """Initialize request capture."""
        self.request: dict[str, Any] | None = None

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        """Capture a POST request."""
        self.request = {"url": url, **kwargs}
        return FakeResponse()


def create_client(session: Any) -> JellyfinClient:
    """Create an authenticated client for play-state tests."""
    return JellyfinClient(
        session=session,
        base_url="https://music.example",
        app_name="Music Assistant",
        app_version="test",
        device_name="test-device",
        device_id="device-id",
        user_id="user-id",
        access_token="secret-token",
    )


@pytest.mark.asyncio
async def test_mark_played_uses_current_authenticated_userdata_endpoint() -> None:
    """Completed plays must use Jellyfin's current authenticated UserPlayedItems API."""
    session = FakeHttpSession()
    client = create_client(session)

    await mark_played(client, "track/id")

    assert session.request == {
        "url": "https://music.example/UserPlayedItems/track%2Fid",
        "params": {"userId": "user-id"},
        "headers": client.headers,
        "ssl": True,
        "raise_for_status": True,
    }


@pytest.mark.asyncio
async def test_provider_reports_only_completed_stopped_tracks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic playback callbacks must not increment Jellyfin play counts."""
    report = AsyncMock()
    monkeypatch.setattr(jellyfin_provider, "mark_played", report)

    provider = object.__new__(JellyfinProvider)
    provider._client = MagicMock(spec=JellyfinClient)
    media_item = MagicMock()

    await provider.on_played(MediaType.TRACK, "track-id", False, 30, media_item, True)
    await provider.on_played(MediaType.TRACK, "track-id", True, 180, media_item, True)
    await provider.on_played(MediaType.RADIO, "radio-id", True, 180, media_item, False)
    report.assert_not_awaited()

    await provider.on_played(MediaType.TRACK, "track-id", True, 180, media_item, False)
    report.assert_awaited_once_with(provider._client, "track-id")
