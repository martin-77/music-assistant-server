"""Jellyfin user play-state helpers."""

from __future__ import annotations

from urllib.parse import quote

from .client import JellyfinClient


async def mark_played(client: JellyfinClient, item_id: str) -> None:
    """Mark an item as played for the authenticated Jellyfin user."""
    encoded_item_id = quote(item_id, safe="")
    async with client.session.post(
        f"{client.base_url}/UserPlayedItems/{encoded_item_id}",
        params={"userId": client.user_id},
        headers=client.headers,
        ssl=client.verify_ssl,
        raise_for_status=True,
    ):
        return
