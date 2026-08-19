"""Tests for Jellyfin provider authentication startup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from music_assistant.providers import jellyfin
from music_assistant.providers.jellyfin import (
    CONF_ACCESS_TOKEN,
    CONF_DEVICE_ID,
    CONF_URL,
    CONF_USER_ID,
    CONF_USERNAME,
    CONF_VERIFY_SSL,
    JellyfinProvider,
)


@pytest.mark.asyncio
async def test_saved_quick_connect_token_is_reused_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider startup must reuse a saved Quick Connect session without password auth."""
    setup_values: dict[str, Any] = {
        CONF_URL: "https://jellyfin.example/",
        CONF_USERNAME: "music-user",
        CONF_VERIFY_SSL: True,
        CONF_USER_ID: "user-id",
        CONF_ACCESS_TOKEN: "access-token",
        CONF_DEVICE_ID: "device-id",
    }
    provider = object.__new__(JellyfinProvider)
    provider.mass = cast(
        Any,
        SimpleNamespace(
            http_session=object(),
            http_session_no_ssl=object(),
            version="test-version",
            server_id="server-id",
        ),
    )

    def get_setup_value(_self: JellyfinProvider, key: str, default: Any = None) -> Any:
        """Return deterministic setup data for the provider under test."""
        return setup_values.get(key, default)

    monkeypatch.setattr(JellyfinProvider, "get_setup_value", get_setup_value)

    client = MagicMock()
    client.get_media_folders = AsyncMock(return_value={"Items": []})
    client_factory = MagicMock(return_value=client)
    password_authenticate = AsyncMock()
    monkeypatch.setattr(jellyfin, "JellyfinClient", client_factory)
    monkeypatch.setattr(jellyfin, "authenticate", password_authenticate)

    await provider.handle_async_init()

    client_factory.assert_called_once_with(
        session=provider.mass.http_session,
        base_url="https://jellyfin.example",
        app_name="Music Assistant",
        app_version="test-version",
        device_name=ANY,
        device_id="device-id",
        user_id="user-id",
        access_token="access-token",
        verify_ssl=True,
    )
    client.get_media_folders.assert_awaited_once_with()
    password_authenticate.assert_not_awaited()
    assert provider._client is client
