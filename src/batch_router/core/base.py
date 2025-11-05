"""Abstract base class defining provider interface."""

from abc import ABC, abstractmethod
from typing import Optional, Any, AsyncIterator
from pathlib import Path
from .requests import UnifiedRequest, UnifiedBatchMetadata
from .responses import BatchStatusResponse, UnifiedResult


class BaseProvider(ABC):
    """
    Abstract base class for all batch providers.

    Each provider must implement:
    1. Conversion from unified format to provider-specific format
    2. Sending batch requests to the provider API
    3. Polling for batch status
    4. Retrieving and converting results back to unified format
    5. File management for JSONL inputs/outputs

    File Management:
    - All providers MUST save JSONL files to .batch_router/generated/<provider>/
    - Format: batch_<batch_id>_input_<format>.jsonl
      - _unified.jsonl: Unified format (for reference)
      - _provider.jsonl: Provider-specific format (what gets sent)
      - _output.jsonl: Raw provider output
      - _results.jsonl: Converted to unified format
    """

    def __init__(self, name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize provider.

        Args:
            name: Provider name (e.g., "openai")
            api_key: API key for the provider (if needed)
            **kwargs: Provider-specific configuration
        """
        self.name = name
        self.api_key = api_key
        self.config = kwargs
        self._validate_configuration()

    @abstractmethod
    def _validate_configuration(self) -> None:
        """
        Validate provider configuration.
        Should check for required credentials, tools, etc.
        Raise ValueError if configuration is invalid.
        """
        pass

    # ========================================================================
    # FORMAT CONVERSION (must be implemented by each provider)
    # ========================================================================

    @abstractmethod
    def _convert_to_provider_format(
        self,
        requests: list[UnifiedRequest]
    ) -> list[dict[str, Any]]:
        """
        Convert unified requests to provider-specific format.

        This is where system_prompt gets converted to provider format:
        - OpenAI: Add as message with role="system"
        - Anthropic: Add as 'system' field in params
        - Google: Add as 'systemInstruction' in config
        - vLLM: Add as message with role="system" (OpenAI-compatible)

        Args:
            requests: List of unified requests

        Returns:
            List of provider-specific request dictionaries
        """
        pass

    @abstractmethod
    def _convert_from_provider_format(
        self,
        provider_results: list[dict[str, Any]]
    ) -> list[UnifiedResult]:
        """
        Convert provider-specific results to unified format.

        Args:
            provider_results: Raw results from provider

        Returns:
            List of unified results
        """
        pass

    # ========================================================================
    # BATCH OPERATIONS (must be implemented by each provider)
    # ========================================================================

    @abstractmethod
    async def send_batch(
        self,
        batch: UnifiedBatchMetadata
    ) -> str:
        """
        Send batch to provider.

        Implementation steps:
        1. Convert requests to provider format
        2. Save unified format JSONL to .batch_router/generated/<provider>/
        3. Save provider format JSONL
        4. Upload/send to provider API
        5. Return batch_id for tracking

        Args:
            batch: Batch metadata with unified requests

        Returns:
            batch_id: Unique identifier for tracking

        Raises:
            ValidationError: If requests are invalid
            ProviderError: If API call fails
        """
        pass

    @abstractmethod
    async def get_status(
        self,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a batch.

        Does NOT retrieve results - only status information.

        Args:
            batch_id: Batch identifier

        Returns:
            Status information including request counts

        Raises:
            BatchNotFoundError: If batch_id doesn't exist
        """
        pass

    @abstractmethod
    async def get_results(
        self,
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed batch.

        Implementation steps:
        1. Download/fetch results from provider
        2. Save raw results to .batch_router/generated/<provider>/
        3. Convert to unified format
        4. Save unified results JSONL
        5. Yield each result

        Args:
            batch_id: Batch identifier

        Yields:
            UnifiedResult objects (order NOT guaranteed)

        Raises:
            BatchNotCompleteError: If batch is still processing
            BatchNotFoundError: If batch_id doesn't exist
        """
        pass

    @abstractmethod
    async def cancel_batch(
        self,
        batch_id: str
    ) -> bool:
        """
        Cancel a running batch.

        Args:
            batch_id: Batch identifier

        Returns:
            True if cancelled successfully, False if already complete

        Raises:
            BatchNotFoundError: If batch_id doesn't exist
        """
        pass

    async def list_batches(
        self,
        limit: int = 20
    ) -> list[BatchStatusResponse]:
        """
        List recent batches.

        Optional method - providers may not implement if API doesn't support.

        Args:
            limit: Maximum number of batches to return

        Returns:
            List of batch status responses
        """
        raise NotImplementedError(f"{self.name} provider does not support listing batches")

    # ========================================================================
    # HELPER METHODS (can be overridden if needed)
    # ========================================================================

    def get_batch_file_path(
        self,
        batch_id: str,
        file_type: str
    ) -> Path:
        """
        Get path for batch file.

        Args:
            batch_id: Batch identifier
            file_type: One of "unified", "provider", "output", "results"

        Returns:
            Path to the file
        """
        # Import here to avoid circular dependency
        from pathlib import Path

        # Base directory for batch files
        batch_dir_path = ".batch_router/generated"
        base_dir = Path(batch_dir_path) / self.name
        base_dir.mkdir(parents=True, exist_ok=True)

        return base_dir / f"batch_{batch_id}_{file_type}.jsonl"
