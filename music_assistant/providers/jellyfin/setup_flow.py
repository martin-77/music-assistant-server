"""Setup flow for the Jellyfin provider."""

from __future__ import annotations

import socket
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from aiohttp import ClientError
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType

from music_assistant.models.setup_flow import SetupFlowError, StepExpiredError
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

from .client import JellyfinClient
from .const import USER_APP_NAME
from .quick_connect import initiate_quick_connect, wait_for_quick_connect

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession

QUICK_CONNECT_TIMEOUT = 9 * 60

_CONNECTION_ENTRIES = (
    ConfigEntry(key=CONF_URL, type=ConfigEntryType.STRING, required=True),
    ConfigEntry(
        key=CONF_AUTH_METHOD,
        type=ConfigEntryType.STRING,
        required=True,
        default_value=AUTH_QUICK_CONNECT,
        options=[
            ConfigValueOption(AUTH_QUICK_CONNECT),
            ConfigValueOption(AUTH_PASSWORD),
        ],
    ),
    ConfigEntry(
        key=CONF_VERIFY_SSL,
        type=ConfigEntryType.BOOLEAN,
        required=False,
        advanced=True,
        default_value=True,
    ),
)

_PASSWORD_ENTRIES = (
    ConfigEntry(key=CONF_USERNAME, type=ConfigEntryType.STRING, required=True),
    ConfigEntry(key=CONF_PASSWORD, type=ConfigEntryType.SECURE_STRING, required=False),
)


def _prefill(entries: tuple[ConfigEntry, ...], setup_data: dict[str, Any]) -> list[ConfigEntry]:
    """Return setup entries prefilled from existing setup data."""
    return [replace(entry, value=setup_data.get(entry.key, entry.value)) for entry in entries]


async def run_setup(session: SetupSession) -> None:
    """Collect Jellyfin connection details and authenticate the provider."""
    errors: dict[str, str] | None = None
    setup_data = dict(session.context.setup_data)

    if CONF_AUTH_METHOD not in setup_data and setup_data.get(CONF_USERNAME):
        setup_data[CONF_AUTH_METHOD] = AUTH_PASSWORD

    while True:
        submitted = await session.form(
            _prefill(_CONNECTION_ENTRIES, setup_data),
            step_id="user",
            errors=errors,
        )
        setup_data.update(submitted)
        errors = None

        if setup_data[CONF_AUTH_METHOD] == AUTH_PASSWORD:
            credentials = await session.form(
                _prefill(_PASSWORD_ENTRIES, setup_data),
                step_id="password",
                last_step=True,
            )
            setup_data.update(credentials)
            setup_data.pop(CONF_ACCESS_TOKEN, None)
            setup_data.pop(CONF_USER_ID, None)
            setup_data.pop(CONF_DEVICE_ID, None)
        else:
            try:
                await _quick_connect(session, setup_data)
            except StepExpiredError:
                errors = {"base": "quick_connect_expired"}
                continue
            except (ClientError, KeyError, TypeError, ValueError):
                errors = {"base": "quick_connect_failed"}
                continue

        try:
            await session.finish(setup_data)
            return
        except SetupFlowError as err:
            errors = {"base": err.translation_key or str(err)}


async def _quick_connect(session: SetupSession, setup_data: dict[str, Any]) -> None:
    """Run Jellyfin Quick Connect and update setup data with the resulting token."""
    base_url = str(setup_data[CONF_URL]).rstrip("/")
    verify_ssl = bool(setup_data.get(CONF_VERIFY_SSL, True))
    http_session = session.mass.http_session if verify_ssl else session.mass.http_session_no_ssl
    device_id = str(setup_data.get(CONF_DEVICE_ID) or uuid4().hex)

    client = JellyfinClient(
        session=http_session,
        base_url=base_url,
        app_name=USER_APP_NAME,
        app_version=session.mass.version,
        device_name=socket.gethostname(),
        device_id=device_id,
        user_id="",
        access_token="",
        verify_ssl=verify_ssl,
    )

    request = await initiate_quick_connect(client)
    quick_connect_url = f"{base_url}/web/index.html#/quickconnect"
    progress_text = (
        f"**Quick Connect code: `{request.code}`**\n\n"
        f"Open [Jellyfin Quick Connect]({quick_connect_url}) and enter this code. "
        "Music Assistant will continue automatically after approval."
    )
    auth = await session.progress_until(
        wait_for_quick_connect(client, request.secret),
        step_id="quick_connect",
        text=progress_text,
        expires_in=QUICK_CONNECT_TIMEOUT,
    )

    setup_data[CONF_USERNAME] = auth.username
    setup_data[CONF_USER_ID] = auth.user_id
    setup_data[CONF_ACCESS_TOKEN] = auth.access_token
    setup_data[CONF_DEVICE_ID] = device_id
    setup_data.pop(CONF_PASSWORD, None)
