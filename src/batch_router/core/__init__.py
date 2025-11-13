"""Core data models and base classes for batch processing."""

from .base import BaseProvider
from .config import GenerationConfig
from .content import TextContent, ImageContent, DocumentContent, AudioContent, MessageContent
from .enums import BatchStatus, ResultStatus, Modality
from .messages import UnifiedMessage
from .requests import UnifiedRequest, UnifiedBatchMetadata
from .responses import BatchStatusResponse, UnifiedResult, RequestCounts, OutputPaths

__all__ = [
    # Base class
    "BaseProvider",
    # Configuration
    "GenerationConfig",
    # Content types
    "TextContent",
    "ImageContent",
    "DocumentContent",
    "AudioContent",
    "MessageContent",
    # Enums
    "BatchStatus",
    "ResultStatus",
    "Modality",
    # Messages
    "UnifiedMessage",
    # Requests
    "UnifiedRequest",
    "UnifiedBatchMetadata",
    # Responses
    "BatchStatusResponse",
    "UnifiedResult",
    "RequestCounts",
    "OutputPaths"
]
