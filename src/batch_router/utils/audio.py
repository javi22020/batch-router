"""Utilities for audio content handling."""

import base64
from pathlib import Path
from typing import Optional
from ..core.content import AudioContent


def encode_audio_file(
    file_path: str | Path,
    media_type: Optional[str] = None
) -> AudioContent:
    """
    Read an audio file and encode it as base64.
    
    Only WAV and MP3 formats are supported.
    
    Args:
        file_path: Path to audio file (must be .wav or .mp3)
        media_type: MIME type (auto-detected from extension if not provided)
    
    Returns:
        AudioContent with base64-encoded data
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If file extension is not .wav or .mp3
        
    Example:
        audio = encode_audio_file("speech.wav")
        message = UnifiedMessage(
            role="user",
            content=[
                TextContent(text="Transcribe this audio"),
                audio
            ]
        )
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Validate file extension
    extension = file_path.suffix.lower()
    if extension not in ['.wav', '.mp3']:
        raise ValueError(
            f"Unsupported audio format: {extension}. "
            "Only .wav and .mp3 files are supported."
        )
    
    # Auto-detect media type from extension if not provided
    if media_type is None:
        media_type = _get_media_type_from_extension(extension)
    
    # Read and encode file
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    
    base64_data = base64.b64encode(audio_bytes).decode("utf-8")
    
    return AudioContent(
        type="audio",
        source_type="base64",
        media_type=media_type,
        data=base64_data
    )


def _get_media_type_from_extension(extension: str) -> str:
    """
    Map file extension to MIME type.
    
    Only supports WAV and MP3 formats.
    """
    extension = extension.lower().lstrip(".")
    
    mime_map = {
        "wav": "audio/wav",
        "mp3": "audio/mp3"
    }
    
    if extension not in mime_map:
        raise ValueError(
            f"Unsupported extension: .{extension}. "
            "Only .wav and .mp3 are supported."
        )
    
    return mime_map[extension]


def decode_audio_content(audio: AudioContent) -> bytes:
    """
    Decode AudioContent back to raw bytes.
    
    Only works for base64-encoded audio.
    
    Args:
        audio: AudioContent with base64 data
        
    Returns:
        Raw audio bytes
        
    Raises:
        ValueError: If source_type is not base64
    """
    if audio.source_type != "base64":
        raise ValueError(
            f"Can only decode base64 audio, got source_type={audio.source_type}"
        )
    
    return base64.b64decode(audio.data)


def estimate_audio_file_size(audio: AudioContent) -> int | None:
    """
    Estimate the size of audio file in bytes.
    
    For base64 data, calculates actual size.
    For URLs/URIs, returns None.
    
    Args:
        audio: AudioContent to estimate
        
    Returns:
        Size in bytes, or None if cannot determine
    """
    if audio.source_type == "base64":
        # Base64 encoding increases size by ~33%
        # So original size = len(base64) * 3/4
        return len(audio.data) * 3 // 4
    
    return None


def validate_audio_format(file_path: str | Path) -> bool:
    """
    Validate that an audio file has a supported format.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if format is supported (WAV or MP3), False otherwise
    """
    extension = Path(file_path).suffix.lower()
    return extension in ['.wav', '.mp3']


