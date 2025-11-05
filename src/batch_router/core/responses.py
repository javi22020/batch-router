"""Response structures from batch operations."""

from dataclasses import dataclass, field
from typing import Optional, Any
from .enums import BatchStatus, ResultStatus


@dataclass
class RequestCounts:
    """
    Breakdown of request statuses within a batch.
    Used to show progress and completion statistics.
    """
    total: int
    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    cancelled: int = 0
    expired: int = 0

    def is_complete(self) -> bool:
        """Check if all requests have finished processing."""
        return self.processing == 0

    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.succeeded / self.total) * 100


@dataclass
class BatchStatusResponse:
    """
    Response from checking batch status.
    Does NOT contain actual results - only status info.
    """
    batch_id: str
    provider: str
    status: BatchStatus
    request_counts: RequestCounts

    # Timestamps (ISO 8601 format)
    created_at: str
    completed_at: Optional[str] = None
    expires_at: Optional[str] = None

    # Provider-specific additional data
    provider_data: dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if batch has finished processing."""
        return self.status in [
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED
        ]


@dataclass
class UnifiedResult:
    """
    Individual request result within a batch.

    Results from all providers are converted to this unified format.
    """
    custom_id: str
    status: ResultStatus

    # If succeeded: full response object
    response: Optional[dict[str, Any]] = None

    # If errored: error details
    error: Optional[dict[str, Any]] = None

    # Provider-specific raw data (for debugging)
    provider_data: dict[str, Any] = field(default_factory=dict)

    def get_text_response(self) -> Optional[str]:
        """
        Extract text response from successful result.
        Handles different provider response formats.
        """
        if self.status != ResultStatus.SUCCEEDED or not self.response:
            return None

        # Try common response structures
        try:
            # OpenAI/vLLM format: response.choices[0].message.content
            if "choices" in self.response:
                choices = self.response.get("choices", [])
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    return message.get("content")

            # Anthropic format: response.content[0].text
            if "content" in self.response:
                content = self.response.get("content", [])
                if content and len(content) > 0:
                    if isinstance(content[0], dict):
                        return content[0].get("text")

            # Google format: response.candidates[0].content.parts[0].text
            if "candidates" in self.response:
                candidates = self.response.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and len(parts) > 0:
                        return parts[0].get("text")

            # Direct text field (fallback)
            if "text" in self.response:
                return self.response.get("text")

        except (KeyError, IndexError, TypeError):
            pass

        return None
