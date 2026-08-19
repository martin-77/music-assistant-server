"""Jellyfin support for MusicAssistant."""

from __future__ import annotations

import hashlib
import socket
from asyncio import TaskGroup
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import MediaType, ProviderFeature, StreamType
from music_assistant_models.errors import LoginFailed, MediaNotFoundError
from music_assistant_models.media_items import (
    Album,
    Artist,
    Playlist,
    ProviderMapping,
    SearchResults,
    Track,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import UNKNOWN_ARTIST, UNKNOWN_ARTIST_ID_MBID
from music_assistant.controllers.cache import use_cache
from music_assistant.mass import MusicAssistant
from music_assistant.models import ProviderInstanceType
from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.jellyfin.parsers import (
    audio_format,
    parse_album,
    parse_artist,
    parse_playlist,
    parse_track,
)

from .client import JellyfinClient, NotFound, authenticate
from .const import (
    ALBUM_FIELDS,
    ARTIST_FIELDS,
    COLLECTION_TYPE_MUSIC,
    COLLECTION_TYPE_PLAYLISTS,
    ITEM_KEY_COLLECTION_TYPE,
    ITEM_KEY_ID,
    ITEM_KEY_MEDIA_STREAMS,
    ITEM_KEY_MEDIA_TYPE,
    ITEM_KEY_NAME,
    ITEM_KEY_RUNTIME_TICKS,
    MEDIA_TYPE_AUDIO,
    SUPPORTED_CONTAINER_FORMATS,
    TRACK_FIELDS,
    USER_APP_NAME,
)
from .playstate import mark_played

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.media_items import MediaItemType
    from music_assistant_models.provider import ProviderManifest

CONF_URL = "url"
CONF_USERNAME = "username"
CONF_PASSWORD = "password"
CONF_VERIFY_SSL = "verify_ssl"
CONF_AUTH_METHOD = "auth_method"
CONF_ACCESS_TOKEN = "access_token"
CONF_USER_ID = "user_id"
CONF_DEVICE_ID = "device_id"
AUTH_QUICK_CONNECT = "quick_connect"
AUTH_PASSWORD = "password"
SUPPORTED_FEATURES = {
    ProviderFeature.LIBRARY_ARTISTS,
    ProviderFeature.LIBRARY_ALBUMS,
    ProviderFeature.LIBRARY_TRACKS,
    ProviderFeature.LIBRARY_PLAYLISTS,
    ProviderFeature.BROWSE,
    ProviderFeature.SEARCH,
    ProviderFeature.ARTIST_ALBUMS,
    ProviderFeature.SIMILAR_TRACKS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return JellyfinProvider(mass, manifest, config, SUPPORTED_FEATURES)


class JellyfinProvider(MusicProvider):
    """Provider for a jellyfin music library."""

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to setup this provider."""
        return ()

    async def handle_async_init(self) -> None:
        """Initialize provider(instance) with given configuration."""
        username = str(self.get_setup_value(CONF_USERNAME) or "")
        verify_ssl = bool(self.get_setup_value(CONF_VERIFY_SSL))
        http_session = self.mass.http_session if verify_ssl else self.mass.http_session_no_ssl
        url = str(self.get_setup_value(CONF_URL)).rstrip("/")

        try:
            access_token = self.get_setup_value(CONF_ACCESS_TOKEN)
            user_id = self.get_setup_value(CONF_USER_ID)
            saved_device_id = self.get_setup_value(CONF_DEVICE_ID)

            if access_token and user_id and saved_device_id:
                self._client = JellyfinClient(
                    session=http_session,
                    base_url=url,
                    app_name=USER_APP_NAME,
                    app_version=self.mass.version,
                    device_name=socket.gethostname(),
                    device_id=str(saved_device_id),
                    user_id=str(user_id),
                    access_token=str(access_token),
                    verify_ssl=verify_ssl,
                )
                # Quick Connect tokens are persistent until revoked. Validate the saved
                # token during provider startup so a revoked session surfaces as LoginFailed.
                await self._client.get_media_folders()
                return

            # Legacy/password configurations continue to authenticate on startup. Keep
            # their deterministic device id so existing installations do not accumulate
            # a new Jellyfin device on every restart or re-add.
            device_id = hashlib.sha256(f"{self.mass.server_id}+{username}".encode()).hexdigest()
            self._client = await authenticate(
                session=http_session,
                url=url,
                username=username,
                password=str(self.get_setup_value(CONF_PASSWORD) or ""),
                app_name=USER_APP_NAME,
                app_version=self.mass.version,
                device_name=socket.gethostname(),
                device_id=device_id,
                verify_ssl=verify_ssl,
            )
        except Exception as err:
            raise LoginFailed(f"Authentication failed: {err}") from err

    @property
    def is_streaming_provider(self) -> bool:
        """Return True if the provider is a streaming provider."""
        return False

    @use_cache(60 * 15)  # Cache for 15 minutes
    async def search(
        self,
        search_query: str,
        media_types: list[MediaType],
        limit: int = 20,
    ) -> SearchResults:
        """
        Perform search on the Jellyfin library.

        :param search_query: Search query.
        :param media_types: A list of media_types to include. All types if None.
        :param limit: Number of items to return in the search (per type).
        """
        artists = None
        albums = None
        tracks = None
        playlists = None

        async with TaskGroup() as tg:
            if MediaType.ARTIST in media_types:
                artists = tg.create_task(self._search_artist(search_query, limit))
            if MediaType.ALBUM in media_types:
                albums = tg.create_task(self._search_album(search_query, limit))
            if MediaType.TRACK in media_types:
                tracks = tg.create_task(self._search_track(search_query, limit))
            if MediaType.PLAYLIST in media_types:
                playlists = tg.create_task(self._search_playlist(search_query, limit))

        search_results = SearchResults()

        if artists:
            search_results.artists = artists.result()
        if albums:
            search_results.albums = albums.result()
        if tracks:
            search_results.tracks = tracks.result()
        if playlists:
            search_results.playlists = playlists.result()

        return search_results

    async def get_library_artists(self) -> AsyncGenerator[Artist]:
        """Retrieve all library artists from Jellyfin Music."""
        jellyfin_libraries = await self._get_music_libraries()
        for jellyfin_library in jellyfin_libraries:
            async for artist in self._client.iter_artists(
                jellyfin_library[ITEM_KEY_ID], ARTIST_FIELDS
            ):
                yield parse_artist(self.logger, self.instance_id, self._client, artist)

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """Retrieve all library albums from Jellyfin Music."""
        jellyfin_libraries = await self._get_music_libraries()
        for jellyfin_library in jellyfin_libraries:
            async for album in self._client.iter_albums(
                jellyfin_library[ITEM_KEY_ID], ALBUM_FIELDS
            ):
                yield parse_album(self.logger, self.instance_id, self._client, album)

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """Retrieve library tracks from Jellyfin Music."""
        jellyfin_libraries = await self._get_music_libraries()
        for jellyfin_library in jellyfin_libraries:
            async for track in self._client.iter_tracks(
                jellyfin_library[ITEM_KEY_ID], TRACK_FIELDS
            ):
                if not len(track[ITEM_KEY_MEDIA_STREAMS]):
                    self.logger.warning(
                        "Invalid track %s: Does not have any media streams", track[ITEM_KEY_NAME]
                    )
                    continue
                yield parse_track(self.logger, self.instance_id, self._client, track)

    async def get_library_playlists(self) -> AsyncGenerator[Playlist]:
        """Retrieve all library playlists from the provider."""
        playlist_libraries = await self._get_playlists()
        for playlist_library in playlist_libraries:
            async for playlist in self._client.iter_playlists(playlist_library[ITEM_KEY_ID]):
                if ITEM_KEY_MEDIA_TYPE in playlist:  # Only jellyfin has this property
                    if playlist[ITEM_KEY_MEDIA_TYPE] == MEDIA_TYPE_AUDIO:
                        yield parse_playlist(self.instance_id, self._client, playlist)
                else:  # emby playlists are only audio type
                    yield parse_playlist(self.instance_id, self._client, playlist)

    async def get_album(self, prov_album_id: str) -> Album:
        """Get full album details by id."""
        try:
            album = await self._client.get_album(prov_album_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {prov_album_id} not found")
        return parse_album(self.logger, self.instance_id, self._client, album)

    @use_cache(3600)  # Cache for 1 hour
    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """Get album tracks for given album id."""
        jellyfin_album_tracks = await self._client.get_album_tracks(prov_album_id, TRACK_FIELDS)
        return [
            parse_track(self.logger, self.instance_id, self._client, jellyfin_album_track)
            for jellyfin_album_track in jellyfin_album_tracks
        ]

    @use_cache(60 * 15)  # Cache for 15 minutes
    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Get full artist details by id."""
        if prov_artist_id == UNKNOWN_ARTIST:
            artist = Artist(
                item_id=UNKNOWN_ARTIST,
                name=UNKNOWN_ARTIST,
                provider=self.instance_id,
                provider_mappings={
                    ProviderMapping(
                        item_id=UNKNOWN_ARTIST,
                        provider_domain=self.domain,
                        provider_instance=self.instance_id,
                    )
                },
            )
            artist.mbid = UNKNOWN_ARTIST_ID_MBID
            return artist

        try:
            jellyfin_artist = await self._client.get_artist(prov_artist_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {prov_artist_id} not found")
        return parse_artist(self.logger, self.instance_id, self._client, jellyfin_artist)

    @use_cache(60 * 15)  # Cache for 15 minutes
    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id."""
        try:
            track = await self._client.get_track(prov_track_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {prov_track_id} not found")
        return parse_track(self.logger, self.instance_id, self._client, track)

    @use_cache(60 * 15)  # Cache for 15 minutes
    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Get full playlist details by id."""
        try:
            playlist = await self._client.get_playlist(prov_playlist_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {prov_playlist_id} not found")
        return parse_playlist(self.instance_id, self._client, playlist)

    @use_cache(3600)  # Cache for 1 hour
    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> list[Track]:
        """Get playlist tracks."""
        result: list[Track] = []
        playlist_items = await self._client.get_playlist_tracks(
            prov_playlist_id,
            TRACK_FIELDS,
            start_index=page * 100,
            limit=100,
        )
        for index, jellyfin_track in enumerate(playlist_items, 1):
            pos = (page * 100) + index
            try:
                if track := parse_track(
                    self.logger, self.instance_id, self._client, jellyfin_track
                ):
                    track.position = pos
                    result.append(track)
            except (KeyError, ValueError) as err:
                self.logger.error(
                    "Skipping track %s: %s", jellyfin_track.get(ITEM_KEY_NAME, index), str(err)
                )
        return result

    @use_cache(3600)  # Cache for 1 hour
    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """Get a list of albums for the given artist."""
        albums = await self._client.get_artist_albums(prov_artist_id, ALBUM_FIELDS)
        return [parse_album(self.logger, self.instance_id, self._client, album) for album in albums]

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return the content details for the given track when it will be streamed."""
        try:
            jellyfin_track = await self._client.get_track(item_id)
        except NotFound:
            raise MediaNotFoundError(f"Item {item_id} not found")
        runtime_ticks = jellyfin_track.get(ITEM_KEY_RUNTIME_TICKS)
        return StreamDetails(
            item_id=jellyfin_track[ITEM_KEY_ID],
            provider=self.instance_id,
            audio_format=audio_format(jellyfin_track),
            stream_type=StreamType.CUSTOM,
            duration=(
                int(runtime_ticks / 10000000)  # 10000000 ticks per second
                if runtime_ticks is not None
                else None
            ),
            can_seek=True,
            allow_seek=True,
        )

    async def get_audio_stream(
        self,
        streamdetails: StreamDetails,
        seek_position: int = 0,
    ) -> AsyncGenerator[bytes]:
        """Return an authenticated audio byte stream from Jellyfin."""
        async for chunk in self._client.stream_audio(
            streamdetails.item_id,
            container=SUPPORTED_CONTAINER_FORMATS,
            seek_position=seek_position,
        ):
            yield chunk

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: int,
        media_item: MediaItemType,
        is_playing: bool = False,
    ) -> None:
        """Sync completed track plays to Jellyfin user data."""
        if media_type != MediaType.TRACK or not fully_played or is_playing:
            return
        await mark_played(self._client, prov_item_id)

    @use_cache(3600)  # Cache for 1 hour
    async def get_similar_tracks(self, prov_track_id: str, limit: int = 25) -> list[Track]:
        """Retrieve a dynamic list of tracks based on the provided item."""
        tracks = await self._client.get_similar_tracks(
            prov_track_id, limit=limit, fields=TRACK_FIELDS
        )
        return [parse_track(self.logger, self.instance_id, self._client, track) for track in tracks]

    async def resolve_image(self, path: str) -> bytes:
        """Resolve Jellyfin artwork through the authenticated API client."""
        return await self._client.resolve_image(path)

    async def _search_track(self, search_query: str, limit: int) -> list[Track]:
        resultset = await self._client.search_tracks(search_query, limit, TRACK_FIELDS)
        tracks = []
        for item in resultset:
            tracks.append(parse_track(self.logger, self.instance_id, self._client, item))
        return tracks

    async def _search_album(self, search_query: str, limit: int) -> list[Album]:
        # an "Artist - Album" style query: search on the album part only
        albumname = search_query.split(" - ", 1)[1] if " - " in search_query else search_query
        resultset = await self._client.search_albums(albumname, limit, ALBUM_FIELDS)
        albums = []
        for item in resultset:
            albums.append(parse_album(self.logger, self.instance_id, self._client, item))
        return albums

    async def _search_artist(self, search_query: str, limit: int) -> list[Artist]:
        resultset = await self._client.search_artists(search_query, limit, ARTIST_FIELDS)
        artists = []
        for item in resultset:
            artists.append(parse_artist(self.logger, self.instance_id, self._client, item))
        return artists

    async def _search_playlist(self, search_query: str, limit: int) -> list[Playlist]:
        resultset = await self._client.search_playlists(search_query, limit)
        playlists = []
        for item in resultset:
            playlists.append(parse_playlist(self.instance_id, self._client, item))
        return playlists

    async def _get_music_libraries(self) -> list[dict[str, Any]]:
        """Return all supported libraries a user has access to."""
        response = await self._client.get_media_folders()
        libraries = response["Items"]
        result = []
        for library in libraries:
            if library.get(ITEM_KEY_COLLECTION_TYPE) == COLLECTION_TYPE_MUSIC:
                result.append(library)
        return result

    async def _get_playlists(self) -> list[dict[str, Any]]:
        """Return all supported libraries a user has access to."""
        response = await self._client.get_media_folders()
        libraries = response["Items"]
        result = []
        for library in libraries:
            if library.get(ITEM_KEY_COLLECTION_TYPE) == COLLECTION_TYPE_PLAYLISTS:
                result.append(library)
        return result
