"""Tests for the Jellyfin provider setup flow."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from music_assistant.providers.jellyfin import (
    AUTH_PASSWORD,
    AUTH_QUICK_CONNECT,
    CONF_ACCESS_TOKEN,
    CONF_AUTH_METHOD,
    CONF_DEVICE_ID,
    CONF_PASSWORD,
    CONF_URL,
    CONF_USER_ID,
    CONF_USERNAME,
    CONF_VERIFY_SSL,
)
from music_assistant.providers.jellyfin import setup_flow
from music_assistant.providers.jellyfin.quick_connect import QuickConnectAuth, QuickConnectRequest

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry


class FakeSetupSession:
    """Minimal setup session for exercising the Jellyfin flow."""

    def __init__(
        self,
        *,
        setup_data: dict[str, Any] | None = None,
        form_results: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize setup context and queued form results."""
        self.context = SimpleNamespace(setup_data=setup_data or {})
        self.mass = SimpleNamespace(
            http_session=object(),
            http_session_no_ssl=object(),
            version="test-version",
        )
        self.form_results = deque(form_results or [])
        self.forms: list[dict[str, Any]] = []
        self.progress: dict[str, Any] | None = None
        self.finished_values: dict[str, Any] | None = None

    async def form(self, entries: list[ConfigEntry], **kwargs: Any) -> dict[str, Any]:
        """Capture a form and return the next queued submission."""
        self.forms.append({"entries": entries, **kwargs})
        return self.form_results.popleft()

    async def progress_until(
        self,
        awaitable: Awaitable[Any],
        **kwargs: Any,
    ) -> Any:
        """Capture a progress step and await its completion."""
        self.progress = dict(kwargs)
        return await awaitable

    async def finish(self, values: dict[str, Any]) -> dict[str, str]:
        """Capture the final setup values."""
        self.finished_values = dict(values)
        return {"instance_id": "jellyfin-test"}


@pytest.mark.asyncio
async def test_quick_connect_flow_persists_token_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quick Connect setup must persist the approved account token and stable device id."""
    initiated_device_id: str | None = None

    async def initiate(client: Any) -> QuickConnectRequest:
        """Return a deterministic pending request and capture its device id."""
        nonlocal initiated_device_id
        initiated_device_id = str(client.device_id)
        assert client.access_token == ""
        assert client.user_id == ""
        assert client.base_url == "https://jellyfin.example"
        return QuickConnectRequest(secret="secret-value", code="123456")

    async def wait(client: Any, secret: str) -> QuickConnectAuth:
        """Return deterministic credentials for the approved request."""
        assert str(client.device_id) == initiated_device_id
        assert secret == "secret-value"
        return QuickConnectAuth(
            user_id="user-id",
            username="music-user",
            access_token="access-token",
        )

    monkeypatch.setattr(setup_flow, "initiate_quick_connect", initiate)
    monkeypatch.setattr(setup_flow, "wait_for_quick_connect", wait)

    session = FakeSetupSession(
        form_results=[
            {
                CONF_URL: "https://jellyfin.example/",
                CONF_AUTH_METHOD: AUTH_QUICK_CONNECT,
                CONF_VERIFY_SSL: True,
            }
        ]
    )

    await setup_flow.run_setup(session)  # type: ignore[arg-type]

    assert initiated_device_id
    assert session.progress == {
        "step_id": "quick_connect",
        "text": (
            "**Quick Connect code**\n\n"
            "```text\n123456\n```\n\n"
            "[Open Jellyfin Quick Connect with this code](https://jellyfin.example/web/"
            "index.html#/quickconnect?code=123456) and approve Music Assistant. "
            "The setup will continue automatically."
        ),
        "expires_in": setup_flow.QUICK_CONNECT_TIMEOUT,
    }
    assert session.finished_values == {
        CONF_URL: "https://jellyfin.example/",
        CONF_AUTH_METHOD: AUTH_QUICK_CONNECT,
        CONF_VERIFY_SSL: True,
        CONF_USERNAME: "music-user",
        CONF_USER_ID: "user-id",
        CONF_ACCESS_TOKEN: "access-token",
        CONF_DEVICE_ID: initiated_device_id,
    }


@pytest.mark.asyncio
async def test_legacy_credentials_default_to_password_authentication() -> None:
    """Existing username/password setup data must remain on the password path."""
    session = FakeSetupSession(
        setup_data={
            CONF_URL: "https://jellyfin.example",
            CONF_USERNAME: "legacy-user",
            CONF_PASSWORD: "legacy-password",
            CONF_VERIFY_SSL: True,
        },
        form_results=[
            {
                CONF_URL: "https://jellyfin.example",
                CONF_AUTH_METHOD: AUTH_PASSWORD,
                CONF_VERIFY_SSL: True,
            },
            {
                CONF_USERNAME: "legacy-user",
                CONF_PASSWORD: "legacy-password",
            },
        ],
    )

    await setup_flow.run_setup(session)  # type: ignore[arg-type]

    auth_entry = next(
        entry for entry in session.forms[0]["entries"] if entry.key == CONF_AUTH_METHOD
    )
    assert auth_entry.value == AUTH_PASSWORD
    assert session.progress is None
    assert session.finished_values == {
        CONF_URL: "https://jellyfin.example",
        CONF_USERNAME: "legacy-user",
        CONF_PASSWORD: "legacy-password",
        CONF_VERIFY_SSL: True,
        CONF_AUTH_METHOD: AUTH_PASSWORD,
    }
