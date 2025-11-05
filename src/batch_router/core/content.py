"""Multimodal content types for messages."""

from dataclasses import dataclass
from typing import Literal


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
