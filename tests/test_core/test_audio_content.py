"""Unit tests for AudioContent class."""

import pytest
from batch_router.core.content import AudioContent
from batch_router.core.enums import Modality


def test_audio_content_creation_base64_wav():
    """Test creating AudioContent with base64 WAV data."""
    audio = AudioContent(
        type="audio",
        source_type="base64",
        media_type="audio/wav",
        data="UklGRiQAAABXQVZF"
    )
    assert audio.type == "audio"
    assert audio.source_type == "base64"
    assert audio.media_type == "audio/wav"
    assert audio.get_modality() == Modality.AUDIO


def test_audio_content_creation_base64_mp3():
    """Test creating AudioContent with base64 MP3 data."""
    audio = AudioContent(
        type="audio",
        source_type="base64",
        media_type="audio/mp3",
        data="SUQzBAAAAAAAI1RTU0UAAAA"
    )
    assert audio.type == "audio"
    assert audio.media_type == "audio/mp3"
    assert audio.get_modality() == Modality.AUDIO


def test_audio_content_wave_mime_type():
    """Test AudioContent with audio/wave MIME type."""
    audio = AudioContent(
        type="audio",
        source_type="base64",
        media_type="audio/wave",
        data="UklGRiQAAABXQVZF"
    )
    assert audio.media_type == "audio/wave"


def test_audio_content_mpeg_mime_type():
    """Test AudioContent with audio/mpeg MIME type."""
    audio = AudioContent(
        type="audio",
        source_type="base64",
        media_type="audio/mpeg",
        data="SUQzBAAAAAAAI1RTU0UAAAA"
    )
    assert audio.media_type == "audio/mpeg"


def test_audio_content_validation_missing_data():
    """Test that AudioContent validates missing data."""
    with pytest.raises(ValueError, match="base64 source_type requires data"):
        AudioContent(
            type="audio",
            source_type="base64",
            media_type="audio/wav",
            data=""
        )


def test_audio_content_validation_invalid_format_aac():
    """Test that AudioContent rejects AAC format."""
    with pytest.raises(ValueError, match="Unsupported audio format"):
        AudioContent(
            type="audio",
            source_type="base64",
            media_type="audio/aac",
            data="UklGRiQAAABXQVZF"
        )


def test_audio_content_validation_invalid_format_flac():
    """Test that AudioContent rejects FLAC format."""
    with pytest.raises(ValueError, match="Unsupported audio format"):
        AudioContent(
            type="audio",
            source_type="base64",
            media_type="audio/flac",
            data="UklGRiQAAABXQVZF"
        )


def test_audio_content_validation_invalid_format_ogg():
    """Test that AudioContent rejects OGG format."""
    with pytest.raises(ValueError, match="Unsupported audio format"):
        AudioContent(
            type="audio",
            source_type="base64",
            media_type="audio/ogg",
            data="UklGRiQAAABXQVZF"
        )


def test_audio_content_url_source():
    """Test AudioContent with URL source."""
    audio = AudioContent(
        type="audio",
        source_type="url",
        media_type="audio/mp3",
        data="https://example.com/audio.mp3"
    )
    assert audio.source_type == "url"
    assert audio.data.startswith("https://")


def test_audio_content_url_validation_invalid():
    """Test that AudioContent validates URL format."""
    with pytest.raises(ValueError, match="url source_type requires valid HTTP"):
        AudioContent(
            type="audio",
            source_type="url",
            media_type="audio/mp3",
            data="ftp://example.com/audio.mp3"
        )


def test_audio_content_file_uri_source():
    """Test AudioContent with file URI source."""
    audio = AudioContent(
        type="audio",
        source_type="file_uri",
        media_type="audio/wav",
        data="gs://bucket/audio.wav"
    )
    assert audio.source_type == "file_uri"
    assert "gs://" in audio.data


def test_audio_content_file_uri_validation():
    """Test that AudioContent validates file_uri has data."""
    with pytest.raises(ValueError, match="file_uri source_type requires data"):
        AudioContent(
            type="audio",
            source_type="file_uri",
            media_type="audio/wav",
            data=""
        )


def test_audio_content_with_duration():
    """Test AudioContent with duration metadata."""
    audio = AudioContent(
        type="audio",
        source_type="base64",
        media_type="audio/wav",
        data="UklGRiQAAABXQVZF",
        duration_seconds=60.0
    )
    assert audio.duration_seconds == 60.0


def test_audio_content_without_duration():
    """Test AudioContent without duration metadata (default None)."""
    audio = AudioContent(
        type="audio",
        source_type="base64",
        media_type="audio/wav",
        data="UklGRiQAAABXQVZF"
    )
    assert audio.duration_seconds is None


def test_audio_content_default_values():
    """Test AudioContent default values."""
    audio = AudioContent(data="UklGRiQAAABXQVZF")
    assert audio.type == "audio"
    assert audio.source_type == "base64"
    assert audio.media_type == "audio/wav"
    assert audio.duration_seconds is None


def test_audio_content_error_message_includes_valid_types():
    """Test that error message includes list of valid MIME types."""
    with pytest.raises(ValueError) as exc_info:
        AudioContent(
            type="audio",
            source_type="base64",
            media_type="audio/aac",
            data="data"
        )
    error_msg = str(exc_info.value)
    assert "audio/wav" in error_msg
    assert "audio/wave" in error_msg
    assert "audio/mp3" in error_msg
    assert "audio/mpeg" in error_msg


