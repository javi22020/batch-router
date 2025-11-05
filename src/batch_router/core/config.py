"""Common generation parameters across providers."""

from dataclasses import dataclass, asdict
from typing import Optional, Any


@dataclass
class GenerationConfig:
    """
    Common generation parameters that map to provider-specific params.

    Providers will convert these to their native parameter names:
    - OpenAI: max_tokens, temperature, top_p, presence_penalty, frequency_penalty
    - Anthropic: max_tokens, temperature, top_p, top_k
    - Google: maxOutputTokens, temperature, topP, topK
    - vLLM: max_tokens, temperature, top_p, top_k
    """
    # Core parameters (supported by all)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Optional parameters (not all providers support)
    top_k: Optional[int] = None  # Not in OpenAI
    stop_sequences: Optional[list[str]] = None
    presence_penalty: Optional[float] = None  # OpenAI, Google only
    frequency_penalty: Optional[float] = None  # OpenAI only

    def __post_init__(self):
        """Validate parameter ranges."""
        if self.temperature is not None:
            if not (0 <= self.temperature <= 2):
                raise ValueError("temperature must be between 0 and 2")

        if self.max_tokens is not None:
            if self.max_tokens < 1:
                raise ValueError("max_tokens must be positive")

        if self.top_p is not None:
            if not (0 <= self.top_p <= 1):
                raise ValueError("top_p must be between 0 and 1")

        if self.top_k is not None:
            if self.top_k < 1:
                raise ValueError("top_k must be positive")

        if self.presence_penalty is not None:
            if not (-2 <= self.presence_penalty <= 2):
                raise ValueError("presence_penalty must be between -2 and 2")

        if self.frequency_penalty is not None:
            if not (-2 <= self.frequency_penalty <= 2):
                raise ValueError("frequency_penalty must be between -2 and 2")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
