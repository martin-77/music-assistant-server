"""Tests for the Jellyfin provider."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest

from music_assistant.mass import MusicAssistant
from music_assistant.providers.jellyfin.client import JellyfinClient, NotFound
from tests.common import get_fixtures_dir, wait_for_sync_completion

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig


MUSIC_FOLDER = "8aeb9430-b1d5-420e-9847-3217ac2120c3"


class FixtureJellyfinClient:
    """In-memory Jellyfin client backed by JSON fixtures."""

    def __init__(self) -> None:
        """Initialize the fixture store."""
        self.objects: dict[str, dict[str, Any]] = {}
        self.parents: dict[str, set[str]] = defaultdict(set)

    def add_json_bytes(self, data: str | bytes) -> None:
        """Add one Jellyfin JSON fixture."""
        item: dict[str, Any] = json.loads(data)
        item_id = str(item["Id"])

        self.objects[item_id] = item
        self.parents[item_id].add(MUSIC_FOLDER)

        for parent in item.get("AlbumArtists", []):
            self.parents[item_id].add(str(parent["Id"]))

        for parent in item.get("ArtistItems", []):
            self.parents[item_id].add(str(parent["Id"]))

        if album_id := item.get("AlbumId"):
            self.parents[item_id].add(str(album_id))

    async def get_media_folders(self) -> dict[str, Any]:
        """Return one fake music library."""
        return {
            "Items": [
                {
                    "Id": MUSIC_FOLDER,
                    "Name": "Music",
                    "CollectionType": "music",
                }
            ],
            "TotalRecordCount": 1,
            "StartIndex": 0,
        }

    async def get_artist(self, item_id: str) -> dict[str, Any]:
        """Return an artist by id."""
        return self._get_item(item_id, "MusicArtist")

    async def get_album(self, item_id: str) -> dict[str, Any]:
        """Return an album by id."""
        return self._get_item(item_id, "MusicAlbum")

    async def get_track(self, item_id: str) -> dict[str, Any]:
        """Return a track by id."""
        return self._get_item(item_id, "Audio")

    async def get_playlist(self, item_id: str) -> dict[str, Any]:
        """Return a playlist by id."""
        return self._get_item(item_id, "Playlist")

    async def search_artists(
        self,
        query: str,
        limit: int,
        fields: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Search artists."""
        return self._search("MusicArtist", query, limit)

    async def search_albums(
        self,
        query: str,
        limit: int,
        fields: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Search albums."""
        return self._search("MusicAlbum", query, limit)

    async def search_tracks(
        self,
        query: str,
        limit: int,
        fields: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Search tracks."""
        return self._search("Audio", query, limit)

    async def search_playlists(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search playlists."""
        return self._search("Playlist", query, limit)

    async def iter_artists(
        self,
        parent_id: str,
        fields: Sequence[str],
    ) -> AsyncGenerator[dict[str, Any]]:
        """Iterate artists below a parent."""
        for item in self._children(parent_id, "MusicArtist"):
            yield item

    async def iter_albums(
        self,
        parent_id: str,
        fields: Sequence[str],
    ) -> AsyncGenerator[dict[str, Any]]:
        """Iterate albums below a parent."""
        for item in self._children(parent_id, "MusicAlbum"):
            yield item

    async def iter_tracks(
        self,
        parent_id: str,
        fields: Sequence[str],
    ) -> AsyncGenerator[dict[str, Any]]:
        """Iterate tracks below a parent."""
        for item in self._children(parent_id, "Audio"):
            yield item

    async def iter_playlists(
        self,
        parent_id: str,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Iterate playlists below a parent."""
        for item in self._children(parent_id, "Playlist"):
            yield item

    async def get_album_tracks(
        self,
        album_id: str,
        fields: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return tracks belonging to an album."""
        return self._children(album_id, "Audio")

    async def get_artist_albums(
        self,
        artist_id: str,
        fields: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return albums belonging to an artist."""
        return self._children(artist_id, "MusicAlbum")

    async def get_playlist_tracks(
        self,
        playlist_id: str,
        fields: Sequence[str],
        *,
        start_index: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return one page of playlist tracks."""
        items = self._children(playlist_id, "Audio")
        return items[start_index : start_index + limit]

    async def get_similar_tracks(
        self,
        track_id: str,
        *,
        limit: int,
        fields: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return deterministic similar tracks for tests."""
        return self._search("Audio", "thrown", limit)

    def artwork(
        self,
        item_id: str,
        image_type: str,
        *,
        index: int | None = None,
    ) -> str:
        """Return a deterministic artwork path."""
        suffix = "" if index is None else f"/{index}"
        return f"jellyfin://image/{item_id}/{image_type}{suffix}"

    def _get_item(self, item_id: str, expected_type: str) -> dict[str, Any]:
        item = self.objects.get(item_id)
        if item is None or item.get("Type") != expected_type:
            raise NotFound(item_id)
        return item

    def _search(
        self,
        item_type: str,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = query.casefold()
        return [
            item
            for item in self.objects.values()
            if item.get("Type") == item_type and query in str(item.get("Name", "")).casefold()
        ][:limit]

    def _children(
        self,
        parent_id: str,
        item_type: str,
    ) -> list[dict[str, Any]]:
        return [
            item
            for item_id, item in self.objects.items()
            if item.get("Type") == item_type and parent_id in self.parents[item_id]
        ]


@pytest.fixture
async def jellyfin_provider(mass: MusicAssistant) -> AsyncGenerator[ProviderConfig]:
    """Configure a fixture-backed Jellyfin provider."""
    client = FixtureJellyfinClient()

    async for _, artist in get_fixtures_dir("artists", "jellyfin"):
        client.add_json_bytes(artist)

    async for _, album in get_fixtures_dir("albums", "jellyfin"):
        client.add_json_bytes(album)

    async for _, track in get_fixtures_dir("tracks", "jellyfin"):
        client.add_json_bytes(track)

    async def authenticate(**_kwargs: Any) -> JellyfinClient:
        return client  # type: ignore[return-value]

    with mock.patch("music_assistant.providers.jellyfin.authenticate", authenticate):
        async with wait_for_sync_completion(mass):
            config = await mass.config._create_provider_instance(
                "jellyfin",
                {},
                setup_data=mass.config._encrypt_values(
                    {
                        "url": "http://localhost",
                        "username": "username",
                        "password": "password",
                    }
                ),
            )
            await mass.music.start_sync()

        yield config


@pytest.mark.usefixtures("jellyfin_provider")
async def test_get_artist_albums(mass: MusicAssistant) -> None:
    """Test that get_artist_albums returns albums for a real artist ID."""
    artists = await mass.music.artists.library_items(search="Ash", summary=False)
    ash = artists[0]
    prov_mapping = next(
        mapping for mapping in ash.provider_mappings if mapping.provider_domain == "jellyfin"
    )
    albums = await mass.music.artists.get_provider_artist_albums(
        prov_mapping.item_id,
        prov_mapping.provider_instance,
    )
    assert any(album.name == "Nu-Clear Sounds" for album in albums)


@pytest.mark.usefixtures("jellyfin_provider")
async def test_initial_sync(mass: MusicAssistant) -> None:
    """Test that initial sync worked."""
    artists = await mass.music.artists.library_items(search="Ash")
    assert artists[0].name == "Ash"

    albums = await mass.music.albums.library_items(search="christmas")
    assert albums[0].name == "This Is Christmas"

    tracks = await mass.music.tracks.library_items(search="where the bands are")
    assert tracks[0].name == "Where the Bands Are"
    assert tracks[0].version == "2018 Version"
