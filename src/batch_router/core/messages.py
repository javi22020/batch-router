"""Unified message representation."""

from dataclasses import dataclass, field, asdict
from typing import Literal, Any
from .types import MessageContent
from .content import TextContent


@dataclass
class UnifiedMessage:
    """
    Unified message format across all providers.

    Important: System messages should NOT be in the messages array.
    Use UnifiedRequest.system_prompt instead.
    Only 'user' and 'assistant' roles are allowed here.
    """
    role: Literal["user", "assistant"]
    content: list[MessageContent]  # List to support multimodal
    provider_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate message structure."""
        if self.role not in ["user", "assistant"]:
            raise ValueError(
                f"Invalid role '{self.role}'. "
                "Only 'user' and 'assistant' roles are allowed. "
                "Use system_prompt field in UnifiedRequest for system instructions."
            )

        if not self.content:
            raise ValueError("content list cannot be empty")

    @classmethod
    def from_text(cls, role: str, text: str, **kwargs) -> "UnifiedMessage":
        """Convenience constructor for text-only messages."""
        return cls(
            role=role,  # type: ignore
            content=[TextContent(text=text)],
            provider_kwargs=kwargs
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": [
                asdict(c) if hasattr(c, "__dataclass_fields__") else c
                for c in self.content
            ],
            "provider_kwargs": self.provider_kwargs
        }
