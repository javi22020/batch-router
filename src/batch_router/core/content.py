"""Multimodal content types for messages."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TextContent:
    """Plain text content in a message."""
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content (base64, URL, or file URI)."""
    type: Literal["image"] = "image"
    source_type: Literal["base64", "url", "file_uri"] = "base64"
    media_type: str = "image/jpeg"  # "image/jpeg", "image/png", etc.
    data: str = ""  # base64 string, URL, or gs:// URI


@dataclass
class DocumentContent:
    """PDF/document content (base64, URL, or file URI)."""
    type: Literal["document"] = "document"
    source_type: Literal["base64", "url", "file_uri"] = "base64"
    media_type: str = "application/pdf"  # "application/pdf", etc.
    data: str = ""


@dataclass
class AudioContent:
    """
    Audio content in a message.
    
    Supports audio in WAV or MP3 format with various source types:
    - base64: Direct base64-encoded audio data
    - url: Public URL to audio file
    - file_uri: Provider-specific file URI (e.g., gs:// for Google)
    
    Supported audio formats:
    - WAV: audio/wav, audio/wave
    - MP3: audio/mp3, audio/mpeg
    
    Examples:
        # Base64 encoded WAV audio
        audio = AudioContent(
            type="audio",
            source_type="base64",
            media_type="audio/wav",
            data="UklGRiQAAABXQVZFZm10..."
        )
        
        # MP3 from URL
        audio = AudioContent(
            type="audio",
            source_type="url",
            media_type="audio/mp3",
            data="https://example.com/audio.mp3"
        )
        
        # Provider-specific file URI (Google)
        audio = AudioContent(
            type="audio",
            source_type="file_uri",
            media_type="audio/wav",
            data="gs://bucket-name/audio.wav"
        )
    """
    type: Literal["audio"] = "audio"
    source_type: Literal["base64", "url", "file_uri"] = "base64"
    media_type: str = "audio/wav"  # Must be audio/wav, audio/wave, audio/mp3, or audio/mpeg
    data: str = ""  # base64 string, URL, or file URI
    
    # Optional metadata
    duration_seconds: Optional[float] = None  # Audio duration in seconds
    
    def __post_init__(self):
        """Validate audio content."""
        # Validate source_type and data relationship
        if self.source_type == "base64" and not self.data:
            raise ValueError("base64 source_type requires data")
        if self.source_type == "url" and not self.data.startswith(("http://", "https://")):
            raise ValueError("url source_type requires valid HTTP(S) URL")
        if self.source_type == "file_uri" and not self.data:
            raise ValueError("file_uri source_type requires data")
        
        # Validate media_type - only WAV and MP3 are supported
        valid_audio_types = {
            "audio/wav", "audio/wave",
            "audio/mp3", "audio/mpeg"
        }
        if self.media_type not in valid_audio_types:
            raise ValueError(
                f"Unsupported audio format: {self.media_type}. "
                f"Only WAV and MP3 formats are supported. "
                f"Valid MIME types: {', '.join(sorted(valid_audio_types))}"
            )
    
    def get_modality(self):
        """Return the modality for this content."""
        from .enums import Modality
        return Modality.AUDIO
