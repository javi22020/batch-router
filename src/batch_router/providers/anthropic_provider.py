"""Anthropic Message Batches API provider implementation."""

import json
import os
from pathlib import Path
from typing import Any, Optional
import anthropic
import aiofiles

from ..core.base import BaseProvider
from ..core.requests import UnifiedRequest, UnifiedBatchMetadata
from ..core.responses import BatchStatusResponse, UnifiedResult, RequestCounts
from ..core.enums import BatchStatus, ResultStatus, Modality
from ..core.content import TextContent, ImageContent, DocumentContent


class AnthropicProvider(BaseProvider):
    """
    Provider implementation for Anthropic Message Batches API.

    Uses the Anthropic SDK to interact with the Message Batches API:
    - Converts unified format to Anthropic batch format
    - Handles system prompts as separate 'system' field
    - Manages batch lifecycle (create, monitor, retrieve results)
    - Saves JSONL files for transparency
    
    Note: Anthropic batch API does NOT support audio currently.
    """
    
    # Anthropic batch API does NOT support audio
    supported_modalities = {Modality.TEXT, Modality.IMAGE}

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration (e.g., base_url, timeout)
        """
        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        super().__init__(name="anthropic", api_key=api_key, **kwargs)

        # Initialize Anthropic client
        client_kwargs = {"api_key": self.api_key}

        # Add optional kwargs
        if "base_url" in kwargs:
            client_kwargs["base_url"] = kwargs["base_url"]
        if "timeout" in kwargs:
            client_kwargs["timeout"] = kwargs["timeout"]

        self.client = anthropic.Anthropic(**client_kwargs)

    def _validate_configuration(self) -> None:
        """Validate that API key is provided."""
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Provide via api_key parameter or ANTHROPIC_API_KEY environment variable."
            )

    def _convert_content_to_anthropic(self, content: list) -> list[dict[str, Any]]:
        """
        Convert unified content format to Anthropic format.

        Args:
            content: List of content objects (TextContent, ImageContent, etc.)

        Returns:
            List of Anthropic-formatted content blocks
        """
        anthropic_content = []

        for item in content:
            if isinstance(item, TextContent):
                anthropic_content.append({
                    "type": "text",
                    "text": item.text
                })
            elif isinstance(item, ImageContent):
                # Anthropic image format
                if item.source_type == "base64":
                    anthropic_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.media_type,
                            "data": item.data
                        }
                    })
                elif item.source_type == "url":
                    anthropic_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": item.data
                        }
                    })
            elif isinstance(item, DocumentContent):
                # Anthropic document format (for PDF support)
                if item.source_type == "base64":
                    anthropic_content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": item.media_type,
                            "data": item.data
                        }
                    })

        return anthropic_content

    def _convert_to_provider_format(
        self,
        requests: list[UnifiedRequest]
    ) -> list[dict[str, Any]]:
        """
        Convert unified requests to Anthropic batch format.

        Anthropic batch format:
        {
            "custom_id": "unique-id",
            "params": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [...],
                "system": "optional system prompt",
                ...other params...
            }
        }

        Args:
            requests: List of unified requests

        Returns:
            List of Anthropic batch request dictionaries
        """
        anthropic_requests = []

        for request in requests:
            # Build messages array
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg.role,
                    "content": self._convert_content_to_anthropic(msg.content)
                })

            # Build params object
            params: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
            }

            # Add system prompt if provided
            if request.system_prompt:
                # Anthropic supports both string and list of strings for system
                params["system"] = request.system_prompt

            # Add generation config if provided
            if request.generation_config:
                config = request.generation_config

                if config.max_tokens is not None:
                    params["max_tokens"] = config.max_tokens
                if config.temperature is not None:
                    params["temperature"] = config.temperature
                if config.top_p is not None:
                    params["top_p"] = config.top_p
                if config.top_k is not None:
                    params["top_k"] = config.top_k
                if config.stop_sequences is not None:
                    params["stop_sequences"] = config.stop_sequences

            # Add provider-specific kwargs
            if request.provider_kwargs:
                params.update(request.provider_kwargs)

            # Ensure max_tokens is set (required by Anthropic)
            if "max_tokens" not in params:
                params["max_tokens"] = 1024  # Default value

            # Build the batch request
            anthropic_requests.append({
                "custom_id": request.custom_id,
                "params": params
            })

        return anthropic_requests

    def _convert_from_provider_format(
        self,
        provider_results: list[dict[str, Any]]
    ) -> list[UnifiedResult]:
        """
        Convert Anthropic batch results to unified format.

        Anthropic result format:
        {
            "custom_id": "...",
            "result": {
                "type": "succeeded" | "errored" | "expired" | "canceled",
                "message": {...} (if succeeded),
                "error": {...} (if errored)
            }
        }

        Args:
            provider_results: Raw results from Anthropic

        Returns:
            List of unified results
        """
        unified_results = []

        for result in provider_results:
            custom_id = result.get("custom_id", "")
            result_data = result.get("result", {})
            result_type = result_data.get("type", "errored")

            # Map result type to unified status
            status_map = {
                "succeeded": ResultStatus.SUCCEEDED,
                "errored": ResultStatus.ERRORED,
                "expired": ResultStatus.EXPIRED,
                "canceled": ResultStatus.CANCELLED
            }
            status = status_map.get(result_type, ResultStatus.ERRORED)

            # Extract response or error
            response = None
            error = None

            if status == ResultStatus.SUCCEEDED:
                # Extract the message object
                message = result_data.get("message")
                if message:
                    response = message
            elif status == ResultStatus.ERRORED:
                # Extract error details
                error = result_data.get("error")

            unified_results.append(UnifiedResult(
                custom_id=custom_id,
                status=status,
                response=response,
                error=error,
                provider_data=result
            ))

        return unified_results

    async def send_batch(
        self,
        batch: UnifiedBatchMetadata
    ) -> str:
        """
        Send batch to Anthropic Message Batches API.

        Steps:
        1. Validate modalities (will raise error if audio is present)
        2. Convert requests to Anthropic format
        3. Save unified and provider JSONL files
        4. Create batch via API
        5. Return batch ID

        Args:
            batch: Batch metadata with unified requests

        Returns:
            batch_id: Anthropic batch ID (e.g., "msgbatch_...")

        Raises:
            UnsupportedModalityError: If audio content is present
            ValueError: If requests are invalid
            Exception: If API call fails
        """
        # Validate modalities FIRST (will raise error if audio is present)
        self.validate_request_modalities(batch.requests)
        
        # Convert to Anthropic format
        anthropic_requests = self._convert_to_provider_format(batch.requests)

        # Create batch via API
        try:
            message_batch = self.client.messages.batches.create(
                requests=anthropic_requests
            )
            batch_id = message_batch.id
        except Exception as e:
            raise Exception(f"Failed to create Anthropic batch: {str(e)}") from e

        # Extract custom naming parameters
        custom_name = batch.name
        model = batch.requests[0].model if batch.requests else None

        # Save metadata for later use in get_results
        self._save_batch_metadata(batch_id, custom_name, model)

        # Save files for transparency
        try:
            # Save unified format
            unified_path = self.get_batch_file_path(batch_id, "unified", custom_name, model)
            self._write_jsonl_sync(
                unified_path,
                [req.to_dict() for req in batch.requests]
            )

            # Save provider format
            provider_path = self.get_batch_file_path(batch_id, "provider", custom_name, model)
            self._write_jsonl_sync(provider_path, anthropic_requests)
        except Exception as e:
            # Log but don't fail the batch
            print(f"Warning: Failed to save batch files: {str(e)}")

        return batch_id

    async def get_status(
        self,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of an Anthropic batch.

        Args:
            batch_id: Anthropic batch ID

        Returns:
            BatchStatusResponse with current status and counts

        Raises:
            Exception: If batch not found or API call fails
        """
        try:
            message_batch = self.client.messages.batches.retrieve(batch_id)
        except Exception as e:
            raise Exception(f"Failed to retrieve batch status: {str(e)}") from e

        # Map Anthropic processing_status to unified BatchStatus
        processing_status = message_batch.processing_status

        if processing_status == "in_progress":
            status = BatchStatus.IN_PROGRESS
        elif processing_status == "ended":
            # Check if it completed successfully or failed
            request_counts = message_batch.request_counts
            if request_counts.errored == request_counts.processing + request_counts.succeeded + request_counts.errored:
                # All requests errored
                status = BatchStatus.FAILED
            else:
                status = BatchStatus.COMPLETED
        elif processing_status == "canceling":
            status = BatchStatus.IN_PROGRESS
        elif processing_status == "canceled":
            status = BatchStatus.CANCELLED
        else:
            status = BatchStatus.IN_PROGRESS

        counts_data = message_batch.request_counts
        request_counts = RequestCounts(
            total=counts_data.processing + counts_data.succeeded + counts_data.errored + counts_data.canceled + counts_data.expired,
            processing=counts_data.processing,
            succeeded=counts_data.succeeded,
            errored=counts_data.errored,
            cancelled=counts_data.canceled,
            expired=counts_data.expired
        )

        return BatchStatusResponse(
            batch_id=batch_id,
            provider="anthropic",
            status=status,
            request_counts=request_counts,
            created_at=message_batch.created_at,
            completed_at=message_batch.ended_at,
            expires_at=message_batch.expires_at,
            provider_data={
                "processing_status": processing_status,
                "results_url": message_batch.results_url
            }
        )

    async def get_results(
        self,
        batch_id: str
    ) -> list[UnifiedResult]:
        """
        Get results from a completed Anthropic batch.

        Steps:
        1. Check if batch is complete
        2. Download results from Anthropic
        3. Save raw results to file
        4. Convert to unified format
        5. Save unified results
        6. Return all results

        Args:
            batch_id: Anthropic batch ID

        Returns:
            List of UnifiedResult objects

        Raises:
            Exception: If batch not complete or results unavailable
        """
        

    async def cancel_batch(
        self,
        batch_id: str
    ) -> bool:
        """
        Cancel a running Anthropic batch.

        Args:
            batch_id: Anthropic batch ID

        Returns:
            True if cancellation initiated, False if already complete

        Raises:
            Exception: If batch not found or API call fails
        """
        try:
            # Check current status
            status = await self.get_status(batch_id)

            # If already complete, return False
            if status.is_complete():
                return False

            # Cancel the batch
            self.client.messages.batches.cancel(batch_id)
            return True

        except Exception as e:
            raise Exception(f"Failed to cancel batch: {str(e)}") from e

    async def list_batches(
        self,
        limit: int = 20
    ) -> list[BatchStatusResponse]:
        """
        List recent Anthropic batches.

        Args:
            limit: Maximum number of batches to return

        Returns:
            List of batch status responses
        """
        try:
            batches = self.client.messages.batches.list(limit=limit)

            results = []
            for batch in batches.data:
                # Map to BatchStatusResponse
                processing_status = batch.processing_status

                if processing_status == "in_progress":
                    status = BatchStatus.IN_PROGRESS
                elif processing_status == "ended":
                    status = BatchStatus.COMPLETED
                elif processing_status == "canceled":
                    status = BatchStatus.CANCELLED
                else:
                    status = BatchStatus.IN_PROGRESS

                counts_data = batch.request_counts
                request_counts = RequestCounts(
                    total=counts_data.processing + counts_data.succeeded + counts_data.errored + counts_data.canceled + counts_data.expired,
                    processing=counts_data.processing,
                    succeeded=counts_data.succeeded,
                    errored=counts_data.errored,
                    cancelled=counts_data.canceled,
                    expired=counts_data.expired
                )

                results.append(BatchStatusResponse(
                    batch_id=batch.id,
                    provider="anthropic",
                    status=status,
                    request_counts=request_counts,
                    created_at=batch.created_at,
                    completed_at=batch.ended_at,
                    expires_at=batch.expires_at,
                    provider_data={
                        "processing_status": processing_status,
                        "results_url": batch.results_url
                    }
                ))

            return results

        except Exception as e:
            raise Exception(f"Failed to list batches: {str(e)}") from e

    # Helper methods for file I/O

    async def _write_jsonl(
        self,
        file_path: Path,
        data: list[dict[str, Any]]
    ) -> None:
        """Write data to JSONL file asynchronously."""
        async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
            for item in data:
                await f.write(json.dumps(item) + '\n')

    async def _read_jsonl(
        self,
        file_path: Path
    ) -> list[dict[str, Any]]:
        """Read JSONL file asynchronously."""
        results = []
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            async for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
