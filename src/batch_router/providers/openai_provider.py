"""OpenAI Batch API provider implementation."""

import json
import os
from datetime import datetime
from typing import Optional, Any, AsyncIterator
from pathlib import Path

try:
    import aiofiles
except ImportError:
    aiofiles = None  # type: ignore

from openai import AsyncOpenAI, OpenAI

from ..core.base import BaseProvider
from ..core.requests import UnifiedRequest, UnifiedBatchMetadata
from ..core.responses import BatchStatusResponse, UnifiedResult, RequestCounts
from ..core.enums import BatchStatus, ResultStatus
from ..core.content import TextContent, ImageContent, DocumentContent


class OpenAIProvider(BaseProvider):
    """
    OpenAI Batch API provider implementation.

    Supports:
    - Chat Completions API via batch processing
    - Text and multimodal content (images via URL or base64)
    - System prompt conversion to system message
    - 50% cost reduction compared to synchronous API
    - 24-hour completion window

    Usage:
        provider = OpenAIProvider(api_key="sk-...")
        batch_id = await provider.send_batch(batch_metadata)
        status = await provider.get_status(batch_id)
        async for result in provider.get_results(batch_id):
            print(result.custom_id, result.status)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Optional custom base URL for API
            **kwargs: Additional configuration options
        """
        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY")

        super().__init__(name="openai", api_key=api_key, **kwargs)

        # Initialize both sync and async clients
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def _validate_configuration(self) -> None:
        """Validate that API key is present."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Provide via api_key parameter or OPENAI_API_KEY environment variable."
            )

    def _convert_to_provider_format(
        self,
        requests: list[UnifiedRequest]
    ) -> list[dict[str, Any]]:
        """
        Convert unified requests to OpenAI Batch API format.

        OpenAI Batch format:
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [...],
                "max_tokens": 1000,
                ...
            }
        }

        System prompt conversion:
        - Unified: request.system_prompt
        - OpenAI: Prepended as message with role="system"
        """
        provider_requests = []

        for request in requests:
            # Convert messages
            messages = []

            # Add system prompt as system message if present
            if request.system_prompt:
                system_content = (
                    request.system_prompt
                    if isinstance(request.system_prompt, str)
                    else "\n".join(request.system_prompt)
                )
                messages.append({
                    "role": "system",
                    "content": system_content
                })

            # Convert unified messages to OpenAI format
            for msg in request.messages:
                openai_message = self._convert_message_to_openai(msg)
                messages.append(openai_message)

            # Build request body
            body: dict[str, Any] = {
                "model": request.model,
                "messages": messages
            }

            # Add generation config parameters
            if request.generation_config:
                config = request.generation_config

                if config.max_tokens is not None:
                    body["max_tokens"] = config.max_tokens
                if config.temperature is not None:
                    body["temperature"] = config.temperature
                if config.top_p is not None:
                    body["top_p"] = config.top_p
                if config.presence_penalty is not None:
                    body["presence_penalty"] = config.presence_penalty
                if config.frequency_penalty is not None:
                    body["frequency_penalty"] = config.frequency_penalty
                if config.stop_sequences is not None:
                    body["stop"] = config.stop_sequences

            # Add provider-specific kwargs
            body.update(request.provider_kwargs)

            # Create OpenAI batch request format
            provider_request = {
                "custom_id": request.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }

            provider_requests.append(provider_request)

        return provider_requests

    def _convert_message_to_openai(self, message) -> dict[str, Any]:
        """Convert unified message to OpenAI message format."""
        # Handle text-only messages (most common case)
        if len(message.content) == 1 and isinstance(message.content[0], TextContent):
            return {
                "role": message.role,
                "content": message.content[0].text
            }

        # Handle multimodal messages
        content_parts = []
        for content_item in message.content:
            if isinstance(content_item, TextContent):
                content_parts.append({
                    "type": "text",
                    "text": content_item.text
                })
            elif isinstance(content_item, ImageContent):
                image_part = self._convert_image_to_openai(content_item)
                content_parts.append(image_part)
            elif isinstance(content_item, DocumentContent):
                # OpenAI doesn't support documents in chat completions
                # Skip or raise error
                pass

        return {
            "role": message.role,
            "content": content_parts
        }

    def _convert_image_to_openai(self, image: ImageContent) -> dict[str, Any]:
        """Convert unified image content to OpenAI format."""
        if image.source_type == "url":
            return {
                "type": "image_url",
                "image_url": {
                    "url": image.data
                }
            }
        elif image.source_type == "base64":
            # OpenAI expects: data:image/jpeg;base64,<base64_string>
            data_url = f"data:{image.media_type};base64,{image.data}"
            return {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        else:
            # file_uri not directly supported, treat as URL
            return {
                "type": "image_url",
                "image_url": {
                    "url": image.data
                }
            }

    def _convert_from_provider_format(
        self,
        provider_results: list[dict[str, Any]]
    ) -> list[UnifiedResult]:
        """
        Convert OpenAI batch results to unified format.

        OpenAI result format:
        {
            "id": "batch_req_123",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "request_id": "req_123",
                "body": {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1711652795,
                    "model": "gpt-3.5-turbo-0125",
                    "choices": [...],
                    "usage": {...}
                }
            },
            "error": null
        }
        """
        unified_results = []

        for result in provider_results:
            custom_id = result.get("custom_id", "")

            # Check if request errored
            if result.get("error"):
                error = result["error"]
                unified_result = UnifiedResult(
                    custom_id=custom_id,
                    status=ResultStatus.ERRORED,
                    error={
                        "code": error.get("code", "unknown"),
                        "message": error.get("message", "Unknown error")
                    },
                    provider_data=result
                )
            else:
                # Successful response
                response = result.get("response", {})
                status_code = response.get("status_code", 0)

                if status_code == 200:
                    body = response.get("body", {})
                    unified_result = UnifiedResult(
                        custom_id=custom_id,
                        status=ResultStatus.SUCCEEDED,
                        response=body,
                        provider_data=result
                    )
                else:
                    # Non-200 status code
                    unified_result = UnifiedResult(
                        custom_id=custom_id,
                        status=ResultStatus.ERRORED,
                        error={
                            "code": f"http_{status_code}",
                            "message": f"Request failed with status {status_code}"
                        },
                        provider_data=result
                    )

            unified_results.append(unified_result)

        return unified_results

    async def send_batch(
        self,
        batch: UnifiedBatchMetadata
    ) -> str:
        """
        Send batch to OpenAI.

        Steps:
        1. Convert requests to OpenAI format
        2. Save unified format JSONL
        3. Save provider format JSONL
        4. Upload file to OpenAI
        5. Create batch job
        6. Return batch ID
        """
        # Convert to provider format
        provider_requests = self._convert_to_provider_format(batch.requests)

        # We'll use a temporary file for uploading
        # Since we don't have a batch_id yet, we'll generate one after creation
        # For now, save with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = f".batch_router/temp/batch_{timestamp}_provider.jsonl"

        # Ensure temp directory exists
        temp_dir = Path(".batch_router/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Write provider format JSONL to temp file
        await self._write_jsonl(temp_file, provider_requests)

        # Upload file to OpenAI
        with open(temp_file, "rb") as f:
            file_response = await self.async_client.files.create(
                file=f,
                purpose="batch"
            )

        file_id = file_response.id

        # Create batch
        batch_response = await self.async_client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=batch.metadata
        )

        batch_id = batch_response.id

        # Now save the files with proper batch_id
        unified_path = self.get_batch_file_path(batch_id, "unified")
        provider_path = self.get_batch_file_path(batch_id, "provider")

        # Save unified format
        unified_data = [req.to_dict() for req in batch.requests]
        await self._write_jsonl(str(unified_path), unified_data)

        # Save provider format
        await self._write_jsonl(str(provider_path), provider_requests)

        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass

        return batch_id

    async def get_status(
        self,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a batch.

        OpenAI status values:
        - validating
        - failed
        - in_progress
        - finalizing
        - completed
        - expired
        - cancelling
        - cancelled
        """
        # Retrieve batch from OpenAI
        batch = await self.async_client.batches.retrieve(batch_id)

        # Map OpenAI status to unified status
        status_map = {
            "validating": BatchStatus.VALIDATING,
            "failed": BatchStatus.FAILED,
            "in_progress": BatchStatus.IN_PROGRESS,
            "finalizing": BatchStatus.IN_PROGRESS,  # Still processing
            "completed": BatchStatus.COMPLETED,
            "expired": BatchStatus.EXPIRED,
            "cancelling": BatchStatus.IN_PROGRESS,  # Still processing
            "cancelled": BatchStatus.CANCELLED,
        }

        unified_status = status_map.get(batch.status, BatchStatus.IN_PROGRESS)

        # Build request counts
        request_counts = RequestCounts(
            total=batch.request_counts.total,
            processing=batch.request_counts.total -
                      batch.request_counts.completed -
                      batch.request_counts.failed,
            succeeded=batch.request_counts.completed,
            errored=batch.request_counts.failed,
            cancelled=0,  # OpenAI doesn't track this separately
            expired=0  # OpenAI doesn't track this separately
        )

        # Convert timestamps
        created_at = datetime.fromtimestamp(batch.created_at).isoformat()
        completed_at = (
            datetime.fromtimestamp(batch.completed_at).isoformat()
            if batch.completed_at else None
        )
        expires_at = (
            datetime.fromtimestamp(batch.expires_at).isoformat()
            if batch.expires_at else None
        )

        return BatchStatusResponse(
            batch_id=batch_id,
            provider="openai",
            status=unified_status,
            request_counts=request_counts,
            created_at=created_at,
            completed_at=completed_at,
            expires_at=expires_at,
            provider_data={
                "input_file_id": batch.input_file_id,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "raw_status": batch.status
            }
        )

    async def get_results(
        self,
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed batch.

        Steps:
        1. Check batch is complete
        2. Download output file
        3. Save raw output
        4. Convert to unified format
        5. Save unified results
        6. Yield each result
        """
        # Get batch status
        batch = await self.async_client.batches.retrieve(batch_id)

        if batch.status not in ["completed", "failed", "expired", "cancelled"]:
            raise ValueError(
                f"Batch {batch_id} is not complete. "
                f"Current status: {batch.status}"
            )

        if not batch.output_file_id:
            # No results available (all failed or expired)
            return

        # Download output file
        file_response = await self.async_client.files.content(batch.output_file_id)
        output_content = file_response.text

        # Parse JSONL
        output_lines = output_content.strip().split("\n")
        provider_results = [json.loads(line) for line in output_lines if line.strip()]

        # Save raw output
        output_path = self.get_batch_file_path(batch_id, "output")
        await self._write_jsonl(str(output_path), provider_results)

        # Convert to unified format
        unified_results = self._convert_from_provider_format(provider_results)

        # Save unified results
        results_path = self.get_batch_file_path(batch_id, "results")
        unified_dicts = [
            {
                "custom_id": r.custom_id,
                "status": r.status.value,
                "response": r.response,
                "error": r.error
            }
            for r in unified_results
        ]
        await self._write_jsonl(str(results_path), unified_dicts)

        # Yield each result
        for result in unified_results:
            yield result

    async def cancel_batch(
        self,
        batch_id: str
    ) -> bool:
        """
        Cancel a running batch.

        Returns:
            True if cancelled, False if already complete
        """
        try:
            batch = await self.async_client.batches.cancel(batch_id)
            return batch.status in ["cancelling", "cancelled"]
        except Exception:
            # Batch might already be complete
            return False

    async def list_batches(
        self,
        limit: int = 20
    ) -> list[BatchStatusResponse]:
        """
        List recent batches.

        OpenAI supports listing batches with pagination.
        """
        batches_page = await self.async_client.batches.list(limit=limit)

        results = []
        for batch in batches_page.data:
            # Convert each batch to BatchStatusResponse
            status_map = {
                "validating": BatchStatus.VALIDATING,
                "failed": BatchStatus.FAILED,
                "in_progress": BatchStatus.IN_PROGRESS,
                "finalizing": BatchStatus.IN_PROGRESS,
                "completed": BatchStatus.COMPLETED,
                "expired": BatchStatus.EXPIRED,
                "cancelling": BatchStatus.IN_PROGRESS,
                "cancelled": BatchStatus.CANCELLED,
            }

            unified_status = status_map.get(batch.status, BatchStatus.IN_PROGRESS)

            request_counts = RequestCounts(
                total=batch.request_counts.total,
                processing=batch.request_counts.total -
                          batch.request_counts.completed -
                          batch.request_counts.failed,
                succeeded=batch.request_counts.completed,
                errored=batch.request_counts.failed,
                cancelled=0,
                expired=0
            )

            created_at = datetime.fromtimestamp(batch.created_at).isoformat()
            completed_at = (
                datetime.fromtimestamp(batch.completed_at).isoformat()
                if batch.completed_at else None
            )
            expires_at = (
                datetime.fromtimestamp(batch.expires_at).isoformat()
                if batch.expires_at else None
            )

            status_response = BatchStatusResponse(
                batch_id=batch.id,
                provider="openai",
                status=unified_status,
                request_counts=request_counts,
                created_at=created_at,
                completed_at=completed_at,
                expires_at=expires_at,
                provider_data={
                    "input_file_id": batch.input_file_id,
                    "output_file_id": batch.output_file_id,
                    "error_file_id": batch.error_file_id,
                    "raw_status": batch.status
                }
            )

            results.append(status_response)

        return results

    # ========================================================================
    # Helper methods
    # ========================================================================

    async def _write_jsonl(
        self,
        file_path: str,
        data: list[dict[str, Any]]
    ) -> None:
        """Write data to JSONL file."""
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

    async def _read_jsonl(
        self,
        file_path: str
    ) -> list[dict[str, Any]]:
        """Read JSONL file."""
        if aiofiles:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                lines = content.strip().split("\n")
                return [json.loads(line) for line in lines if line.strip()]
        else:
            with open(file_path, "r") as f:
                lines = f.read().strip().split("\n")
                return [json.loads(line) for line in lines if line.strip()]
