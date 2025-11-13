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
    messages: list[UnifiedMessage]
