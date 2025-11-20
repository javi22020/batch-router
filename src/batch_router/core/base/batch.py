from enum import Enum

class BatchStatus(Enum):
    """
    Enumeration representing the possible statuses of a batch process.

    Attributes:
        PENDING (str): The batch is queued and waiting to process.
        RUNNING (str): The batch is currently being processed.
        COMPLETED (str): The batch has successfully finished processing.
        FAILED (str): The batch processing encountered an error and failed.
        CANCELLED (str): The batch was manually cancelled.
        EXPIRED (str): The batch was not processed within the allowed time frame.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
