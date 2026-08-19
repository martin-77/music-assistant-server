"""Internal Jellyfin authentication and streaming helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiojellyfin import Connection

if TYPE_CHECKING:
    from aiojellyfin.session import SessionConfiguration

STREAM_CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_STREAMING_BITRATE = 140_000_000
TICKS_PER_SECOND = 10_000_000


@dataclass(slots=True)
class JellyfinSession:
    """Authenticated Jellyfin session used for requests not handled by aiojellyfin."""

    config: SessionConfiguration
    user_id: str
    access_token: str

    @property
    def base_url(self) -> str:
        """Return normalized Jellyfin base URL."""
        return self.config.url.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        """Return current Jellyfin authorization headers."""
        return {
            "User-Agent": self.config.user_agent,
            "Authorization": self.config.authentication_header(self.access_token),
        }

    async def stream_audio(
        self,
        item_id: str,
        *,
        container: str,
        seek_position: int = 0,
    ) -> AsyncGenerator[bytes]:
        """Stream an audio item using modern header-based Jellyfin authentication."""
        params: dict[str, str] = {
            "userId": self.user_id,
            "deviceId": self.config.device_id,
            "maxStreamingBitrate": str(DEFAULT_MAX_STREAMING_BITRATE),
            "container": container,
        }

        if seek_position > 0:
            params["startTimeTicks"] = str(seek_position * TICKS_PER_SECOND)

        async with self.config.session.get(
            f"{self.base_url}/Audio/{item_id}/universal",
            params=params,
            headers=self.headers,
            ssl=self.config.verify_ssl,
            raise_for_status=True,
        ) as response:
            async for chunk in response.content.iter_chunked(STREAM_CHUNK_SIZE):
                yield chunk


async def authenticate(
    session_config: SessionConfiguration,
    username: str,
    password: str = "",
) -> tuple[Connection, JellyfinSession]:
    """Authenticate with Jellyfin and create legacy and internal clients."""
    response = await session_config.session.post(
        f"{session_config.url.rstrip('/')}/Users/AuthenticateByName",
        json={"Username": username, "Pw": password},
        headers={
            "Content-Type": "application/json",
            "User-Agent": session_config.user_agent,
            "Authorization": session_config.authentication_header(),
        },
        ssl=session_config.verify_ssl,
        raise_for_status=True,
    )
    payload: dict[str, Any] = await response.json()

    user_id = str(payload["User"]["Id"])
    access_token = str(payload["AccessToken"])

    return (
        Connection(session_config, user_id, access_token),
        JellyfinSession(
            config=session_config,
            user_id=user_id,
            access_token=access_token,
        ),
    )
