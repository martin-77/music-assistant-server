"""Test we can parse Jellyfin models into Music Assistant models."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any

import pytest
from music_assistant_models.enums import ContentType

from music_assistant.providers.jellyfin.const import (
    ITEM_KEY_CONTAINER,
    ITEM_KEY_MEDIA_CHANNELS,
    ITEM_KEY_MEDIA_CODEC,
    ITEM_KEY_MEDIA_SOURCES,
    ITEM_KEY_MEDIA_STREAM_TYPE,
    ITEM_KEY_MEDIA_STREAMS,
)
from music_assistant.providers.jellyfin.parsers import (
    audio_format,
    parse_album,
    parse_artist,
    parse_track,
)

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
ARTIST_FIXTURES = list(FIXTURES_DIR.glob("artists/*.json"))
ALBUM_FIXTURES = list(FIXTURES_DIR.glob("albums/*.json"))
TRACK_FIXTURES = list(FIXTURES_DIR.glob("tracks/*.json"))

_LOGGER = logging.getLogger(__name__)


class FakeJellyfinClient:
    """Minimal Jellyfin client used by parser tests."""

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


@pytest.fixture
def client() -> FakeJellyfinClient:
    """Return a fake Jellyfin client."""
    return FakeJellyfinClient()


def load_fixture(path: pathlib.Path) -> dict[str, Any]:
    """Load a Jellyfin JSON fixture."""
    with path.open(encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)
    return data


@pytest.mark.parametrize("example", ARTIST_FIXTURES, ids=lambda val: str(val.stem))
def test_parse_artists(
    example: pathlib.Path,
    client: FakeJellyfinClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test we can parse artists."""
    raw_data = load_fixture(example)
    parsed = parse_artist(_LOGGER, "xx-instance-id-xx", client, raw_data).to_dict()
    parsed["external_ids"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", ALBUM_FIXTURES, ids=lambda val: str(val.stem))
def test_parse_albums(
    example: pathlib.Path,
    client: FakeJellyfinClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test we can parse albums."""
    raw_data = load_fixture(example)
    parsed = parse_album(_LOGGER, "xx-instance-id-xx", client, raw_data).to_dict()
    parsed["external_ids"].sort()
    assert snapshot == parsed


@pytest.mark.parametrize("example", TRACK_FIXTURES, ids=lambda val: str(val.stem))
def test_parse_tracks(
    example: pathlib.Path,
    client: FakeJellyfinClient,
    snapshot: SnapshotAssertion,
) -> None:
    """Test we can parse tracks."""
    raw_data = load_fixture(example)
    parsed = parse_track(_LOGGER, "xx-instance-id-xx", client, raw_data).to_dict()
    parsed["external_ids"].sort()
    assert snapshot == parsed


def test_audio_format_empty_mediastreams() -> None:
    """Test audio_format handles empty MediaStreams array."""
    track: dict[str, Any] = {
        ITEM_KEY_MEDIA_STREAMS: [],
    }

    result = audio_format(track)

    assert result is not None
    assert hasattr(result, "content_type")


def test_audio_format_missing_channels() -> None:
    """Test audio_format applies default when Channels field is missing."""
    track: dict[str, Any] = {
        ITEM_KEY_MEDIA_SOURCES: [{ITEM_KEY_CONTAINER: "mp3"}],
        ITEM_KEY_MEDIA_STREAMS: [
            {
                ITEM_KEY_MEDIA_STREAM_TYPE: "Audio",
                ITEM_KEY_MEDIA_CODEC: "mp3",
                "SampleRate": 48000,
                "BitDepth": 16,
                "BitRate": 320000,
            }
        ],
    }

    result = audio_format(track)

    assert result is not None
    assert result.channels == 2
    assert result.sample_rate == 48000
    assert result.bit_depth == 16
    assert result.bit_rate == 320


def test_audio_format_wav_container_not_treated_as_raw_pcm() -> None:
    """A WAV/PCM source must report the container rather than raw PCM."""
    track: dict[str, Any] = {
        ITEM_KEY_MEDIA_SOURCES: [{ITEM_KEY_CONTAINER: "wav"}],
        ITEM_KEY_MEDIA_STREAMS: [
            {
                ITEM_KEY_MEDIA_STREAM_TYPE: "Video",
                ITEM_KEY_MEDIA_CODEC: "mjpeg",
            },
            {
                ITEM_KEY_MEDIA_STREAM_TYPE: "Audio",
                ITEM_KEY_MEDIA_CODEC: "pcm_s24le",
                ITEM_KEY_MEDIA_CHANNELS: 2,
                "SampleRate": 48000,
                "BitDepth": 24,
            },
        ],
    }

    result = audio_format(track)

    assert result.content_type == ContentType.WAV
    assert not result.content_type.is_pcm()
    assert result.codec_type == ContentType.PCM_S24LE
    assert result.sample_rate == 48000
    assert result.bit_depth == 24
    assert result.channels == 2
