"""Jellyfin Quick Connect authentication helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .client import JellyfinClient

QUICK_CONNECT_POLL_INTERVAL = 2.0


@dataclass(frozen=True, slots=True)
class QuickConnectRequest:
    """Pending Jellyfin Quick Connect request."""

    secret: str
    code: str


@dataclass(frozen=True, slots=True)
class QuickConnectAuth:
    """Credentials returned after a Quick Connect request is approved."""

    user_id: str
    username: str
    access_token: str


async def initiate_quick_connect(client: JellyfinClient) -> QuickConnectRequest:
    """Create a Quick Connect request on the Jellyfin server."""
    async with client.session.post(
        f"{client.base_url}/QuickConnect/Initiate",
        json={},
        headers=client.headers,
        ssl=client.verify_ssl,
        raise_for_status=True,
    ) as response:
        payload: dict[str, Any] = await response.json()

    return QuickConnectRequest(
        secret=str(payload["Secret"]),
        code=str(payload["Code"]),
    )


async def wait_for_quick_connect(
    client: JellyfinClient,
    secret: str,
    *,
    poll_interval: float = QUICK_CONNECT_POLL_INTERVAL,
) -> QuickConnectAuth:
    """Wait until a Quick Connect request is approved and exchange it for a token."""
    while True:
        async with client.session.get(
            f"{client.base_url}/QuickConnect/Connect",
            params={"secret": secret},
            headers=client.headers,
            ssl=client.verify_ssl,
            raise_for_status=True,
        ) as response:
            state: dict[str, Any] = await response.json()

        if bool(state.get("Authenticated")):
            break
        await asyncio.sleep(poll_interval)

    async with client.session.post(
        f"{client.base_url}/Users/AuthenticateWithQuickConnect",
        json={"Secret": secret},
        headers=client.headers,
        ssl=client.verify_ssl,
        raise_for_status=True,
    ) as response:
        payload: dict[str, Any] = await response.json()

    user = payload["User"]
    return QuickConnectAuth(
        user_id=str(user["Id"]),
        username=str(user["Name"]),
        access_token=str(payload["AccessToken"]),
    )
