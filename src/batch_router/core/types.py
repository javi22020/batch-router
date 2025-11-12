"""Type aliases and protocols for better IDE support."""

from typing import Union
from .content import TextContent, ImageContent, DocumentContent, AudioContent

# Type alias for any message content
MessageContent = Union[TextContent, ImageContent, DocumentContent, AudioContent]

# Type alias for system prompt (can be string or list of strings)
SystemPrompt = Union[str, list[str], None]
