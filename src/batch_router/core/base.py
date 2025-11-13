"""Abstract base class defining provider interface."""

from abc import ABC, abstractmethod
from typing import Optional, Any, AsyncIterator
from pathlib import Path
import json
import asyncio
import aiofiles
from .requests import UnifiedRequest, UnifiedBatchMetadata
from .responses import BatchStatusResponse, UnifiedResult
from .enums import Modality
from .content import TextContent, ImageContent, DocumentContent, AudioContent
from ..exceptions import UnsupportedModalityError


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
    
    # Class attribute - must be overridden by each provider
    supported_modalities: set[Modality] = set()

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
        
        # Validate that subclass defined supported_modalities
        if not self.supported_modalities:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define supported_modalities"
            )
        
        self._validate_configuration()

    @abstractmethod
    def _validate_configuration(self) -> None:
        """
        Validate provider configuration.
        Should check for required credentials, tools, etc.
        Raise ValueError if configuration is invalid.
        """
        pass
    
    def validate_request_modalities(
        self,
        requests: list[UnifiedRequest]
    ) -> None:
        """
        Validate that all content modalities in requests are supported.
        
        Args:
            requests: List of unified requests to validate
            
        Raises:
            UnsupportedModalityError: If any request contains unsupported modality
        """
        for req in requests:
            for message in req.messages:
                for content in message.content:
                    # Get modality from content
                    if isinstance(content, TextContent):
                        modality = Modality.TEXT
                    elif isinstance(content, ImageContent):
                        modality = Modality.IMAGE
                    elif isinstance(content, DocumentContent):
                        modality = Modality.DOCUMENT
                    elif isinstance(content, AudioContent):
                        modality = Modality.AUDIO
                    else:
                        raise ValueError(f"Unknown content type: {type(content)}")
                    
                    # Check if supported
                    if modality not in self.supported_modalities:
                        raise UnsupportedModalityError(
                            f"Provider '{self.name}' does not support {modality.value} "
                            f"content in batch API. Supported modalities: "
                            f"{', '.join(m.value for m in self.supported_modalities)}"
                        )

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

    async def send_batch_and_wait(
        self,
        batch: UnifiedBatchMetadata,
        poll_rate: int = 15
    ) -> list[UnifiedResult]:
        """
        Send a batch and wait for it to complete, returning all results.

        This is a convenience method that combines send_batch, polling via get_status,
        and collecting results from get_results into a single operation.

        Args:
            batch: Batch metadata containing requests and configuration
            poll_rate: Seconds to wait between status checks (default: 15)

        Returns:
            List of all UnifiedResult objects from the completed batch.
            Note: Order is NOT guaranteed to match input order.

        Raises:
            BatchNotFoundError: If batch_id doesn't exist during polling
            Any exceptions raised by send_batch, get_status, or get_results
        """
        batch_id = await self.send_batch(batch=batch)
        status = await self.get_status(batch_id=batch_id)

        while not status.is_complete():
            await asyncio.sleep(poll_rate)
            status = await self.get_status(batch_id=batch_id)
        
        results_iterator = await self.get_results(batch_id=batch_id)

        results: list[UnifiedResult] = []
        async for result in results_iterator:
            results.append(result)
        
        return results

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

    def _save_batch_metadata(
        self,
        batch_id: str,
        custom_name: Optional[str],
        model: Optional[str]
    ) -> None:
        """
        Save batch metadata for later retrieval.

        This allows get_results to use the same file naming scheme as send_batch.

        Args:
            batch_id: Batch identifier
            custom_name: Custom name from UnifiedBatchMetadata.name
            model: Model name from the first request
        """
        if not custom_name or not model:
            return  # No metadata to save

        metadata = {
            "custom_name": custom_name,
            "model": model
        }

        batch_dir_path = ".batch_router/generated"
        base_dir = Path(batch_dir_path) / self.name
        base_dir.mkdir(parents=True, exist_ok=True)

        meta_file = base_dir / f"batch_{batch_id}.meta.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)

    def _load_batch_metadata(
        self,
        batch_id: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Load batch metadata if it exists.

        Args:
            batch_id: Batch identifier

        Returns:
            Tuple of (custom_name, model) or (None, None) if no metadata exists
        """
        batch_dir_path = ".batch_router/generated"
        base_dir = Path(batch_dir_path) / self.name
        meta_file = base_dir / f"batch_{batch_id}.meta.json"

        if not meta_file.exists():
            return None, None

        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            return metadata.get("custom_name"), metadata.get("model")
        except (json.JSONDecodeError, IOError):
            return None, None

    def get_batch_file_path(
        self,
        batch_id: str,
        file_type: str,
        custom_name: Optional[str] = None,
        model: Optional[str] = None
    ) -> Path:
        """
        Get path for batch file.

        Args:
            batch_id: Batch identifier
            file_type: One of "unified", "provider", "output", "results"
            custom_name: Optional custom name for the file (from UnifiedBatchMetadata.name)
            model: Optional model name (used with custom_name)

        Returns:
            Path to the file

        Note:
            If custom_name is provided, the filename format will be:
            {name}_{model}_{provider}_{file_type}.jsonl

            All components are sanitized (alphanumeric + dashes only, underscores replaced).
        """
        # Import here to avoid circular dependency
        from pathlib import Path
        from ..utilities import sanitize_filename_component

        # Base directory for batch files
        batch_dir_path = ".batch_router/generated"
        base_dir = Path(batch_dir_path) / self.name
        base_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on whether custom_name is provided
        if custom_name and model:
            # Use custom naming format: {name}_{model}_{provider}_{file_type}.jsonl
            sanitized_name = sanitize_filename_component(custom_name)
            sanitized_model = sanitize_filename_component(model)
            sanitized_provider = sanitize_filename_component(self.name)
            sanitized_file_type = sanitize_filename_component(file_type)

            filename = f"{sanitized_name}_{sanitized_model}_{sanitized_provider}_{sanitized_file_type}.jsonl"
        else:
            # Use default naming format: batch_{batch_id}_{file_type}.jsonl
            filename = f"batch_{batch_id}_{file_type}.jsonl"

        return base_dir / filename

    async def _read_jsonl(
        self,
        file_path: str | Path
    ) -> list[dict[str, Any]]:
        """Read JSONL file."""
        if aiofiles:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                lines = content.strip().split("\n")
                return [json.loads(line) for line in lines if line.strip()]
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                return [json.loads(line) for line in lines if line.strip()]
    
    def _write_jsonl_sync(
        self,
        file_path: str | Path,
        data: list[dict[str, Any]]
    ):
        """Write data to JSONL file. Sync version."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        text = "\n".join([json.dumps(item, ensure_ascii=False) for item in data])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    async def _write_jsonl_async(
        self,
        file_path: str | Path,
        data: list[dict[str, Any]]
    ) -> None:
        """Write data to JSONL file. Async version."""
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        if aiofiles:
            # Use async file operations if available
            async with aiofiles.open(file_path, "w") as f:
                for item in data:
                    await f.write(json.dumps(item) + "\n")
        else:
            # Fallback to synchronous write
            with open(file_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
