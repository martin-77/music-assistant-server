"""Internal Jellyfin API client."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote, urlparse

from aiohttp import ClientResponseError, ClientSession

STREAM_CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_STREAMING_BITRATE = 140_000_000
TICKS_PER_SECOND = 10_000_000

DEFAULT_FIELDS = (
    "Path,Genres,SortName,Studios,Writer,Taglines,LocalTrailerCount,"
    "OfficialRating,CumulativeRunTimeTicks,ItemCounts,"
    "Metascore,AirTime,DateCreated,People,Overview,"
    "CriticRating,CriticRatingSummary,Etag,ShortOverview,ProductionLocations,"
    "Tags,ProviderIds,ParentId,RemoteTrailers,SpecialEpisodeNumbers,"
    "MediaSources,VoteCount,RecursiveItemCount,PrimaryImageAspectRatio"
)


class NotFound(Exception):
    """Raised when a Jellyfin item does not exist."""


@dataclass(slots=True)
class JellyfinClient:
    """Authenticated Jellyfin API client."""

    session: ClientSession
    base_url: str
    app_name: str
    app_version: str
    device_name: str
    device_id: str
    user_id: str
    access_token: str
    verify_ssl: bool = True

    @property
    def user_agent(self) -> str:
        """Return the client user agent."""
        return f"{self.app_name}/{self.app_version}"

    @property
    def headers(self) -> dict[str, str]:
        """Return authenticated Jellyfin request headers."""
        return {
            "User-Agent": self.user_agent,
            "Authorization": self._authentication_header(self.access_token),
        }

    async def get_media_folders(self) -> dict[str, Any]:
        """Return media folders visible to the authenticated user."""
        return await self._get_json("/Items")

    async def get_artist(self, item_id: str) -> dict[str, Any]:
        """Return one artist."""
        return await self._get_item(item_id, "MusicArtist")

    async def get_album(self, item_id: str) -> dict[str, Any]:
        """Return one album."""
        return await self._get_item(item_id, "MusicAlbum")

    async def get_track(self, item_id: str) -> dict[str, Any]:
        """Return one audio track."""
        return await self._get_item(item_id, "Audio")

    async def get_playlist(self, item_id: str) -> dict[str, Any]:
        """Return one playlist."""
        return await self._get_item(item_id, "Playlist")

    async def search_artists(
        self, query: str, limit: int, fields: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Search artists."""
        response = await self._items_request(
            endpoint="/Artists",
            search_term=query,
            limit=limit,
            fields=fields,
            enable_user_data=True,
        )
        return self._items_from_response(response)

    async def search_albums(
        self, query: str, limit: int, fields: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Search albums."""
        response = await self._items_request(
            item_type="MusicAlbum",
            search_term=query,
            limit=limit,
            fields=fields,
            enable_user_data=True,
        )
        return self._items_from_response(response)

    async def search_tracks(
        self, query: str, limit: int, fields: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Search tracks."""
        response = await self._items_request(
            item_type="Audio",
            search_term=query,
            limit=limit,
            fields=fields,
            enable_user_data=True,
        )
        return self._items_from_response(response)

    async def search_playlists(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search playlists."""
        response = await self._items_request(
            item_type="Playlist",
            search_term=query,
            limit=limit,
            enable_user_data=True,
        )
        return self._items_from_response(response)

    async def iter_artists(
        self, parent_id: str, fields: Sequence[str]
    ) -> AsyncGenerator[dict[str, Any]]:
        """Iterate artists below a parent."""
        async for item in self._iter_items(
            endpoint="/Artists",
            parent_id=parent_id,
            fields=fields,
            enable_user_data=True,
        ):
            yield item

    async def iter_albums(
        self, parent_id: str, fields: Sequence[str]
    ) -> AsyncGenerator[dict[str, Any]]:
        """Iterate albums below a parent."""
        async for item in self._iter_items(
            item_type="MusicAlbum",
            parent_id=parent_id,
            fields=fields,
            enable_user_data=True,
        ):
            yield item

    async def iter_tracks(
        self, parent_id: str, fields: Sequence[str]
    ) -> AsyncGenerator[dict[str, Any]]:
        """Iterate tracks below a parent."""
        async for item in self._iter_items(
            item_type="Audio",
            parent_id=parent_id,
            fields=fields,
            enable_user_data=True,
        ):
            yield item

    async def iter_playlists(self, parent_id: str) -> AsyncGenerator[dict[str, Any]]:
        """Iterate playlists below a parent."""
        async for item in self._iter_items(
            item_type="Playlist",
            parent_id=parent_id,
            enable_user_data=True,
        ):
            yield item

    async def get_album_tracks(self, album_id: str, fields: Sequence[str]) -> list[dict[str, Any]]:
        """Return tracks belonging to an album."""
        response = await self._items_request(
            item_type="Audio",
            parent_id=album_id,
            fields=fields,
            enable_user_data=True,
        )
        return self._items_from_response(response)

    async def get_artist_albums(
        self, artist_id: str, fields: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Return albums belonging to an artist."""
        response = await self._items_request(
            item_type="MusicAlbum",
            parent_id=artist_id,
            fields=fields,
            enable_user_data=True,
        )
        return self._items_from_response(response)

    async def get_playlist_tracks(
        self,
        playlist_id: str,
        fields: Sequence[str],
        *,
        start_index: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return one page of playlist tracks."""
        response = await self._items_request(
            endpoint=f"/Playlists/{playlist_id}/Items",
            fields=fields,
            enable_user_data=True,
            start_index=start_index,
            limit=limit,
            recursive=False,
        )
        return self._items_from_response(response)

    async def get_similar_tracks(
        self,
        track_id: str,
        *,
        limit: int,
        fields: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return tracks similar to the supplied track."""
        params = {
            "limit": str(limit),
            "fields": ",".join(fields),
        }
        response = await self._get_json(f"/Items/{track_id}/Similar", params)
        return self._items_from_response(response)

    async def stream_audio(
        self,
        item_id: str,
        *,
        container: str,
        seek_position: int = 0,
    ) -> AsyncGenerator[bytes]:
        """Stream an audio item using header-based authentication."""
        params = {
            "userId": self.user_id,
            "deviceId": self.device_id,
            "maxStreamingBitrate": str(DEFAULT_MAX_STREAMING_BITRATE),
            "container": container,
        }

        if seek_position > 0:
            params["startTimeTicks"] = str(seek_position * TICKS_PER_SECOND)

        async with self.session.get(
            f"{self.base_url}/Audio/{item_id}/universal",
            params=params,
            headers=self.headers,
            ssl=self.verify_ssl,
            raise_for_status=True,
        ) as response:
            async for chunk in response.content.iter_chunked(STREAM_CHUNK_SIZE):
                yield chunk

    def artwork(
        self,
        item_id: str,
        image_type: str,
        *,
        index: int | None = None,
    ) -> str:
        """Return an opaque Music Assistant path for Jellyfin artwork."""
        encoded_item = quote(item_id, safe="")
        encoded_type = quote(image_type, safe="")
        encoded_index = "" if index is None else f"/{index}"
        return f"jellyfin://image/{encoded_item}/{encoded_type}{encoded_index}"

    async def resolve_image(self, path: str) -> bytes:
        """Resolve Jellyfin artwork into image bytes."""
        item_id, image_type, index = self._parse_image_path(path)

        endpoint = f"/Items/{item_id}/Images/{image_type}"
        if index is not None:
            endpoint = f"{endpoint}/{index}"

        try:
            async with self.session.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                ssl=self.verify_ssl,
                raise_for_status=True,
            ) as response:
                return await response.read()
        except ClientResponseError as err:
            if err.status == 404:
                raise FileNotFoundError(path) from err
            raise

    def _parse_image_path(self, path: str) -> tuple[str, str, str | None]:
        """Parse current and legacy Jellyfin artwork paths."""
        parsed = urlparse(path)

        if parsed.scheme == "jellyfin" and parsed.netloc == "image":
            parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
        else:
            base = urlparse(self.base_url)

            if (
                parsed.scheme not in {"http", "https"}
                or parsed.scheme != base.scheme
                or parsed.netloc != base.netloc
            ):
                raise FileNotFoundError(f"Invalid Jellyfin image path: {path}")

            base_path = base.path.rstrip("/")
            image_prefix = f"{base_path}/Items/"
            if not parsed.path.startswith(image_prefix):
                raise FileNotFoundError(f"Invalid Jellyfin image path: {path}")

            relative_path = parsed.path[len(base_path) :].strip("/")
            legacy_parts = [unquote(part) for part in relative_path.split("/") if part]

            if (
                len(legacy_parts) not in (4, 5)
                or legacy_parts[0] != "Items"
                or legacy_parts[2] != "Images"
            ):
                raise FileNotFoundError(f"Invalid Jellyfin image path: {path}")

            parts = [legacy_parts[1], legacy_parts[3]]
            if len(legacy_parts) == 5:
                parts.append(legacy_parts[4])

        if len(parts) not in (2, 3):
            raise FileNotFoundError(f"Invalid Jellyfin image path: {path}")

        item_id, image_type = parts[:2]
        index = parts[2] if len(parts) == 3 else None
        return item_id, image_type, index

    @staticmethod
    def _items_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
        """Return typed item dictionaries from a Jellyfin response."""
        items = response.get("Items", [])
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    async def _get_item(self, item_id: str, expected_type: str) -> dict[str, Any]:
        item = await self._get_json(
            f"/Users/{self.user_id}/Items/{item_id}",
            {"Fields": DEFAULT_FIELDS},
        )
        if item.get("Type") != expected_type:
            raise NotFound(item_id)
        return item

    async def _get_json(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            async with self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers=self.headers,
                ssl=self.verify_ssl,
                raise_for_status=True,
            ) as response:
                result: dict[str, Any] = await response.json()
                return result
        except ClientResponseError as err:
            if err.status == 404:
                raise NotFound(endpoint) from err
            raise

    async def _items_request(
        self,
        *,
        endpoint: str = "/Items",
        item_type: str | None = None,
        parent_id: str | None = None,
        search_term: str | None = None,
        fields: Sequence[str] = (),
        enable_user_data: bool = False,
        start_index: int | None = None,
        limit: int | None = None,
        recursive: bool = True,
    ) -> dict[str, Any]:
        params: dict[str, str] = {
            "recursive": "true" if recursive else "false",
        }

        if item_type is not None:
            params["includeItemTypes"] = item_type
        if parent_id is not None:
            params["parentId"] = parent_id
        if search_term is not None:
            params["searchTerm"] = search_term
        if fields:
            params["fields"] = ",".join(fields)
        if enable_user_data:
            params["enableUserData"] = "true"
        if start_index is not None:
            params["startIndex"] = str(start_index)
        if limit is not None:
            params["limit"] = str(limit)

        return await self._get_json(endpoint, params)

    async def _iter_items(
        self,
        *,
        endpoint: str = "/Items",
        item_type: str | None = None,
        parent_id: str | None = None,
        fields: Sequence[str] = (),
        enable_user_data: bool = False,
        page_size: int = 100,
    ) -> AsyncGenerator[dict[str, Any]]:
        start_index = 0

        while True:
            response = await self._items_request(
                endpoint=endpoint,
                item_type=item_type,
                parent_id=parent_id,
                fields=fields,
                enable_user_data=enable_user_data,
                start_index=start_index,
                limit=page_size,
            )

            items: list[dict[str, Any]] = response.get("Items", [])
            for item in items:
                yield item

            start_index += len(items)
            total = int(response.get("TotalRecordCount", start_index))

            if not items or start_index >= total:
                return

    def _authentication_header(self, api_token: str | None = None) -> str:
        params = {
            "Client": self.app_name,
            "Device": self.device_name,
            "DeviceId": self.device_id,
            "Version": self.app_version,
        }
        if api_token:
            params["Token"] = api_token

        values = ", ".join(f'{key}="{value}"' for key, value in params.items())
        return f"MediaBrowser {values}"


async def authenticate(
    *,
    session: ClientSession,
    url: str,
    username: str,
    password: str,
    app_name: str,
    app_version: str,
    device_name: str,
    device_id: str,
    verify_ssl: bool,
) -> JellyfinClient:
    """Authenticate with Jellyfin and return an API client."""
    base_url = url.rstrip("/")

    temporary = JellyfinClient(
        session=session,
        base_url=base_url,
        app_name=app_name,
        app_version=app_version,
        device_name=device_name,
        device_id=device_id,
        user_id="",
        access_token="",
        verify_ssl=verify_ssl,
    )

    async with session.post(
        f"{base_url}/Users/AuthenticateByName",
        json={"Username": username, "Pw": password},
        headers={
            "Content-Type": "application/json",
            "User-Agent": temporary.user_agent,
            "Authorization": temporary._authentication_header(),
        },
        ssl=verify_ssl,
        raise_for_status=True,
    ) as response:
        payload: dict[str, Any] = await response.json()

    return JellyfinClient(
        session=session,
        base_url=base_url,
        app_name=app_name,
        app_version=app_version,
        device_name=device_name,
        device_id=device_id,
        user_id=str(payload["User"]["Id"]),
        access_token=str(payload["AccessToken"]),
        verify_ssl=verify_ssl,
    )
