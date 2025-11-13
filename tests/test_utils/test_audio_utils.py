"""Unit tests for audio utility functions."""

import base64
import pytest
from pathlib import Path
from batch_router.utilities.audio import (
    encode_audio_file,
    decode_audio_content,
    validate_audio_format,
    _get_media_type_from_extension,
)
from batch_router.core.content import AudioContent


# Test data - minimal valid WAV header
MINIMAL_WAV_DATA = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
# Test data - MP3 ID3 header
MINIMAL_MP3_DATA = b'ID3\x04\x00\x00\x00\x00\x00#TSSE\x00\x00\x00'


@pytest.fixture
def temp_audio_files(tmp_path):
    """Create temporary audio files for testing."""
    wav_file = tmp_path / "test_audio.wav"
    wav_file.write_bytes(MINIMAL_WAV_DATA)
    
    mp3_file = tmp_path / "test_audio.mp3"
    mp3_file.write_bytes(MINIMAL_MP3_DATA)
    
    return {"wav": wav_file, "mp3": mp3_file}


def test_encode_audio_file_wav(temp_audio_files):
    """Test encoding a WAV file."""
    audio = encode_audio_file(temp_audio_files["wav"])
    
    assert isinstance(audio, AudioContent)
    assert audio.type == "audio"
    assert audio.source_type == "base64"
    assert audio.media_type == "audio/wav"
    assert len(audio.data) > 0
    # Verify it's valid base64
    base64.b64decode(audio.data)


def test_encode_audio_file_mp3(temp_audio_files):
    """Test encoding an MP3 file."""
    audio = encode_audio_file(temp_audio_files["mp3"])
    
    assert isinstance(audio, AudioContent)
    assert audio.type == "audio"
    assert audio.source_type == "base64"
    assert audio.media_type == "audio/mp3"
    assert len(audio.data) > 0
    # Verify it's valid base64
    base64.b64decode(audio.data)


def test_encode_audio_file_with_explicit_media_type(temp_audio_files):
    """Test encoding with explicit media_type parameter."""
    audio = encode_audio_file(temp_audio_files["wav"], media_type="audio/wave")
    assert audio.media_type == "audio/wave"


def test_encode_audio_file_not_found():
    """Test encoding a non-existent file."""
    with pytest.raises(FileNotFoundError):
        encode_audio_file("nonexistent.wav")


def test_encode_audio_file_unsupported_format(tmp_path):
    """Test encoding an unsupported audio format."""
    aac_file = tmp_path / "test.aac"
    aac_file.write_bytes(b"dummy data")
    
    with pytest.raises(ValueError, match="Unsupported audio format"):
        encode_audio_file(aac_file)


def test_encode_audio_file_wrong_extension(tmp_path):
    """Test that only .wav and .mp3 extensions are accepted."""
    flac_file = tmp_path / "test.flac"
    flac_file.write_bytes(b"dummy data")
    
    with pytest.raises(ValueError, match="Only .wav and .mp3"):
        encode_audio_file(flac_file)


def test_get_media_type_from_extension_wav():
    """Test media type detection for WAV files."""
    assert _get_media_type_from_extension(".wav") == "audio/wav"
    assert _get_media_type_from_extension("wav") == "audio/wav"
    assert _get_media_type_from_extension(".WAV") == "audio/wav"


def test_get_media_type_from_extension_mp3():
    """Test media type detection for MP3 files."""
    assert _get_media_type_from_extension(".mp3") == "audio/mp3"
    assert _get_media_type_from_extension("mp3") == "audio/mp3"
    assert _get_media_type_from_extension(".MP3") == "audio/mp3"


def test_get_media_type_from_extension_unsupported():
    """Test media type detection for unsupported formats."""
    with pytest.raises(ValueError, match="Unsupported extension"):
        _get_media_type_from_extension(".aac")


def test_decode_audio_content():
    """Test decoding AudioContent back to bytes."""
    original_data = MINIMAL_WAV_DATA
    base64_data = base64.b64encode(original_data).decode("utf-8")
    
    audio = AudioContent(
        source_type="base64",
        media_type="audio/wav",
        data=base64_data
    )
    
    decoded = decode_audio_content(audio)
    assert decoded == original_data


def test_decode_audio_content_round_trip(temp_audio_files):
    """Test encoding and decoding round trip."""
    original_data = temp_audio_files["wav"].read_bytes()
    
    # Encode
    audio = encode_audio_file(temp_audio_files["wav"])
    
    # Decode
    decoded = decode_audio_content(audio)
    
    assert decoded == original_data


def test_decode_audio_content_url_source_fails():
    """Test that decoding URL source fails with clear error."""
    audio = AudioContent(
        source_type="url",
        media_type="audio/mp3",
        data="https://example.com/audio.mp3"
    )
    
    with pytest.raises(ValueError, match="Can only decode base64 audio"):
        decode_audio_content(audio)


def test_decode_audio_content_file_uri_source_fails():
    """Test that decoding file_uri source fails with clear error."""
    audio = AudioContent(
        source_type="file_uri",
        media_type="audio/wav",
        data="gs://bucket/audio.wav"
    )
    
    with pytest.raises(ValueError, match="Can only decode base64 audio"):
        decode_audio_content(audio)

def test_validate_audio_format_wav():
    """Test validating WAV format."""
    assert validate_audio_format("audio.wav") is True
    assert validate_audio_format("audio.WAV") is True
    assert validate_audio_format(Path("path/to/audio.wav")) is True


def test_validate_audio_format_mp3():
    """Test validating MP3 format."""
    assert validate_audio_format("audio.mp3") is True
    assert validate_audio_format("audio.MP3") is True
    assert validate_audio_format(Path("path/to/audio.mp3")) is True


def test_validate_audio_format_unsupported():
    """Test validating unsupported formats."""
    assert validate_audio_format("audio.aac") is False
    assert validate_audio_format("audio.flac") is False
    assert validate_audio_format("audio.ogg") is False
    assert validate_audio_format("audio.m4a") is False


def test_validate_audio_format_no_extension():
    """Test validating file with no extension."""
    assert validate_audio_format("audio") is False
    assert validate_audio_format("audio.") is False


def test_encode_with_path_string(temp_audio_files):
    """Test encoding with path as string."""
    audio = encode_audio_file(str(temp_audio_files["wav"]))
    assert audio.media_type == "audio/wav"


def test_encode_with_path_object(temp_audio_files):
    """Test encoding with Path object."""
    audio = encode_audio_file(temp_audio_files["wav"])
    assert audio.media_type == "audio/wav"


