"""Custom exception hierarchy for batch router."""

from typing import Optional


class BatchRouterError(Exception):
    """Base exception for all batch router errors."""
    pass


class ProviderNotFoundError(BatchRouterError):
    """Raised when provider is not registered."""
    pass


class ValidationError(BatchRouterError):
    """Raised when request validation fails."""
    pass


class BatchTimeoutError(BatchRouterError):
    """Raised when batch doesn't complete within timeout."""
    pass


class BatchNotFoundError(BatchRouterError):
    """Raised when batch_id doesn't exist."""
    pass


class BatchNotCompleteError(BatchRouterError):
    """Raised when trying to get results from incomplete batch."""
    pass


class FileOperationError(BatchRouterError):
    """Raised when file operations fail."""
    pass


class ProviderError(BatchRouterError):
    """Raised when provider API call fails."""

    def __init__(self, provider: str, message: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
