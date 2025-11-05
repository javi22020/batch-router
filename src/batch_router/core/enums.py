"""Status enumerations for batch processing."""

from enum import Enum


class BatchStatus(Enum):
    """
    Unified batch status across all providers.

    Provider mapping:
    - OpenAI: validating, in_progress, completed, failed, expired, cancelled
    - Anthropic: in_progress, ended (then check request_counts)
    - Google: JOB_STATE_PENDING, JOB_STATE_RUNNING, JOB_STATE_SUCCEEDED,
              JOB_STATE_FAILED, JOB_STATE_CANCELLED
    - vLLM: File-based (processing if file not ready, completed when exists)
    """
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ResultStatus(Enum):
    """
    Status of individual request within a batch.
    All providers support these statuses.
    """
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
