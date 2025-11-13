"""Multimodal content types for messages."""

from pydantic import BaseModel
from typing import Literal, Optional


class TextContent(BaseModel):
    """Plain text content in a message."""
    type: Literal["text"] = "text"
    text: str = ""


class ImageContent(BaseModel):
    """Image content (base64, URL, or file URI)."""
    type: Literal["image"] = "image"
    source_type: Literal["base64", "url", "file_uri"] = "base64"
    media_type: str = "image/jpeg"  # "image/jpeg", "image/png", etc.
    data: str = ""  # base64 string, URL, or gs:// URI


class DocumentContent(BaseModel):
    """PDF/document content (base64, URL, or file URI)."""
    type: Literal["document"] = "document"
    source_type: Literal["base64", "url", "file_uri"] = "base64"
    media_type: str = "application/pdf"  # "application/pdf", etc.
    data: str = ""


class AudioContent(BaseModel):
    """
    Audio content in a message.
    
    Supports audio in WAV or MP3 format with various source types:
    - base64: Direct base64-encoded audio data
    - url: Public URL to audio file
    - file_uri: Provider-specific file URI (e.g., gs:// for Google)
    
    Supported audio formats:
    - WAV: audio/wav, audio/wave
    - MP3: audio/mp3, audio/mpeg
    """
    type: Literal["audio"] = "audio"
    source_type: Literal["base64", "url", "file_uri"] = "base64"
    media_type: str = "audio/wav"  # Must be audio/wav, audio/wave, audio/mp3, or audio/mpeg
    data: str = ""  # base64 string, URL, or file URI
    
    # Optional metadata
    duration_seconds: Optional[float] = None  # Audio duration in seconds

MessageContent = TextContent | ImageContent | DocumentContent | AudioContent
