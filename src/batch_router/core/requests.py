"""Request structures for batch operations."""

from dataclasses import dataclass, field
from typing import Optional, Any
from .messages import UnifiedMessage
from .config import GenerationConfig


@dataclass
class UnifiedRequest:
    """
    A single request in unified format.

    System prompt is at REQUEST level, not in messages array.
    This design choice allows proper handling across all providers:
    - OpenAI: Converts to message with role="system"
    - Anthropic: Uses 'system' parameter
    - Google: Uses 'systemInstruction' in config
    """
    custom_id: str  # REQUIRED: User must provide unique ID
    model: str  # Provider-specific model ID (e.g., "gpt-4o", "claude-sonnet-4-5")
    messages: list[UnifiedMessage]

    # System prompt at request level
    system_prompt: Optional[str | list[str]] = None

    # Generation parameters
    generation_config: Optional[GenerationConfig] = None

    # Provider-specific advanced features
    provider_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate request structure."""
        if not self.custom_id:
            raise ValueError("custom_id is required and cannot be empty")

        if not self.messages:
            raise ValueError("messages list cannot be empty")

        # Ensure no system messages in messages array
        for msg in self.messages:
            if msg.role not in ["user", "assistant"]:
                raise ValueError(
                    f"Invalid role '{msg.role}' in messages. "
                    "Use system_prompt field for system instructions."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "custom_id": self.custom_id,
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
        }

        if self.system_prompt is not None:
            result["system_prompt"] = self.system_prompt

        if self.generation_config is not None:
            result["generation_config"] = self.generation_config.to_dict()

        if self.provider_kwargs:
            result["provider_kwargs"] = self.provider_kwargs

        return result


@dataclass
class UnifiedBatchMetadata:
    """
    Batch metadata before sending to provider.
    This is NOT the response - just the input specification.
    name is an optional custom name for generated files, useful for test tracking. If provided, the name will be used as a prefix for any output files related to this batch.
    """
    provider: str  # "openai", "anthropic", "google", "vllm"
    requests: list[UnifiedRequest]
    metadata: dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None  # Optional custom name for generated files

    def __post_init__(self):
        """Validate batch metadata."""
        valid_providers = ["openai", "anthropic", "google", "mistral", "vllm"]
        if self.provider not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}")

        if not self.requests:
            raise ValueError("requests list cannot be empty")

        # Check for duplicate custom_ids
        custom_ids = [req.custom_id for req in self.requests]
        if len(custom_ids) != len(set(custom_ids)):
            raise ValueError("Duplicate custom_id values found in requests")
