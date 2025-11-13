"""Unified message representation."""

from pydantic import BaseModel, Field
from typing import Literal, Any
from .content import MessageContent

class UnifiedMessage(BaseModel):
    """
    Unified message format across all providers.

    Important: System messages should NOT be in the messages array.
    Use UnifiedRequest.system_prompt instead.
    Only 'user' and 'assistant' roles are allowed here.
    """
    role: Literal["user", "assistant"]
    content: list[MessageContent]
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)
