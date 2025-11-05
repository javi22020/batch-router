"""Core data models and base classes for batch processing."""

from .base import BaseProvider
from .config import GenerationConfig
from .content import TextContent, ImageContent, DocumentContent
from .enums import BatchStatus, ResultStatus
from .messages import UnifiedMessage
from .requests import UnifiedRequest, UnifiedBatchMetadata
from .responses import BatchStatusResponse, UnifiedResult, RequestCounts
from .types import MessageContent, SystemPrompt

__all__ = [
    # Base class
    "BaseProvider",
    # Configuration
    "GenerationConfig",
    # Content types
    "TextContent",
    "ImageContent",
    "DocumentContent",
    # Enums
    "BatchStatus",
    "ResultStatus",
    # Messages
    "UnifiedMessage",
    # Requests
    "UnifiedRequest",
    "UnifiedBatchMetadata",
    # Responses
    "BatchStatusResponse",
    "UnifiedResult",
    "RequestCounts",
    # Type aliases
    "MessageContent",
    "SystemPrompt",
]
