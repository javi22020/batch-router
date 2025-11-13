"""Response structures from batch operations."""

from pydantic import BaseModel
from typing import Optional, Any
from .enums import BatchStatus, ResultStatus
from datetime import datetime
from pathlib import Path
from .messages import UnifiedMessage

class OutputPaths(BaseModel):
    raw_output_batch_jsonl: str | Path
    unified_output_jsonl: str | Path

class RequestCounts(BaseModel):
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


class BatchStatusResponse(BaseModel):
    """
    Response from checking batch status.
    Does NOT contain actual results - only status info.
    """
    batch_id: str
    provider: str
    status: BatchStatus
    request_counts: RequestCounts

    # Timestamps
    created_at: datetime
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Provider-specific additional data
    provider_data: dict[str, Any] = {}

    def is_complete(self) -> bool:
        """Check if batch has finished processing."""
        return self.status in [
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED
        ]


class UnifiedResult(BaseModel):
    """
    Individual request result within a batch.

    Results from all providers are converted to this unified format.
    """
    custom_id: str
    status: ResultStatus
    response: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    provider_data: dict[str, Any] = {}

    def get_text_response(self) -> Optional[str]:
        """
        Extract text response from provider-specific response format.
        
        Supports multiple provider formats:
        - OpenAI: response["choices"][0]["message"]["content"]
        - Anthropic: response["content"][0]["text"]
        - Google: response["candidates"][0]["content"]["parts"][0]["text"]
        - Direct: response["text"]
        
        Returns:
            Text response if available, None otherwise
        """
        if self.status != ResultStatus.SUCCEEDED or not self.response:
            return None
        
        # Try OpenAI format
        if "choices" in self.response:
            try:
                return self.response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass
        
        # Try Anthropic format
        if "content" in self.response:
            try:
                content = self.response["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text")
            except (KeyError, IndexError, TypeError):
                pass
        
        # Try Google format
        if "candidates" in self.response:
            try:
                return self.response["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                pass
        
        # Try direct text field
        if "text" in self.response:
            return self.response["text"]
        
        return None
