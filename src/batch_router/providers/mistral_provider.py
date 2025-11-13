"""Mistral Batch API provider implementation."""

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional
import aiofiles

try:
    from mistralai import Mistral
except ImportError:
    raise ImportError(
        "Mistral SDK is required for MistralProvider. "
        "Install it with: pip install mistralai"
    )

from ..core.base import BaseProvider
from ..core.requests import UnifiedRequest, UnifiedBatchMetadata
from ..core.responses import BatchStatusResponse, UnifiedResult, RequestCounts
from ..core.enums import BatchStatus, ResultStatus, Modality
from ..core.content import TextContent, ImageContent, DocumentContent


class MistralProvider(BaseProvider):
    """
    Provider implementation for Mistral Batch API.

    Uses the Mistral SDK to interact with the Batch API:
    - Converts unified format to Mistral batch format
    - Handles system prompts as messages with role="system"
    - Manages batch lifecycle (create, monitor, retrieve results)
    - Saves JSONL files for transparency
    
    Note: Mistral batch API currently does not support audio.
    """
    
    # Mistral batch API does not support audio currently
    supported_modalities = {Modality.TEXT, Modality.IMAGE}

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Mistral provider.

        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            **kwargs: Additional configuration (e.g., endpoint, server_url, timeout)
        """
        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")

        super().__init__(name="mistral", api_key=api_key, **kwargs)

        # Initialize Mistral client
        client_kwargs = {"api_key": self.api_key}

        # Add optional kwargs
        if "server_url" in kwargs:
            client_kwargs["server_url"] = kwargs["server_url"]

        self.client = Mistral(**client_kwargs)

        # Store the endpoint for batch processing (default to chat completions)
        self.endpoint = kwargs.get("endpoint", "/v1/chat/completions")

    def _validate_configuration(self) -> None:
        """Validate that API key is provided."""
        if not self.api_key:
            raise ValueError(
                "Mistral API key is required. "
                "Provide via api_key parameter or MISTRAL_API_KEY environment variable."
            )

    def _convert_content_to_mistral(self, content: list) -> list[dict[str, Any]]:
        """
        Convert unified content format to Mistral format.

        Args:
            content: List of content objects (TextContent, ImageContent, etc.)

        Returns:
            List of Mistral-formatted content blocks
        """
        mistral_content = []

        for item in content:
            if isinstance(item, TextContent):
                mistral_content.append({
                    "type": "text",
                    "text": item.text
                })
            elif isinstance(item, ImageContent):
                # Mistral image format (supports base64 and URL)
                if item.source_type == "base64":
                    mistral_content.append({
                        "type": "image_url",
                        "image_url": f"data:{item.media_type};base64,{item.data}"
                    })
                elif item.source_type == "url":
                    mistral_content.append({
                        "type": "image_url",
                        "image_url": item.data
                    })

        return mistral_content

    def _convert_to_provider_format(
        self,
        requests: list[UnifiedRequest]
    ) -> list[dict[str, Any]]:
        """
        Convert unified requests to Mistral batch format.

        Mistral batch format:
        {
            "custom_id": "unique-id",
            "body": {
                "model": "mistral-small-latest",
                "max_tokens": 100,
                "messages": [
                    {"role": "system", "content": "..."},  # Optional
                    {"role": "user", "content": "..."}
                ],
                ...other params...
            }
        }

        Args:
            requests: List of unified requests

        Returns:
            List of Mistral batch request dictionaries
        """
        mistral_requests = []

        for request in requests:
            # Build messages array
            messages = []

            # Add system prompt as first message if provided
            if request.system_prompt:
                if isinstance(request.system_prompt, list):
                    system_text = "\n".join(request.system_prompt)
                else:
                    system_text = request.system_prompt

                messages.append({
                    "role": "system",
                    "content": system_text
                })

            # Add conversation messages
            for msg in request.messages:
                # Convert content to appropriate format
                content = self._convert_content_to_mistral(msg.content)

                # If there's only one text content, use string format
                if len(content) == 1 and content[0].get("type") == "text":
                    messages.append({
                        "role": msg.role,
                        "content": content[0]["text"]
                    })
                else:
                    # Use array format for multimodal content
                    messages.append({
                        "role": msg.role,
                        "content": content
                    })

            # Build body object
            body: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
            }

            # Add generation config if provided
            if request.generation_config:
                config = request.generation_config

                if config.max_tokens is not None:
                    body["max_tokens"] = config.max_tokens
                if config.temperature is not None:
                    body["temperature"] = config.temperature
                if config.top_p is not None:
                    body["top_p"] = config.top_p
                if config.stop_sequences is not None:
                    body["stop"] = config.stop_sequences

            # Add provider-specific kwargs
            if request.provider_kwargs:
                body.update(request.provider_kwargs)

            # Build the batch request
            mistral_requests.append({
                "custom_id": request.custom_id,
                "body": body
            })

        return mistral_requests

    def _convert_from_provider_format(
        self,
        provider_results: list[dict[str, Any]]
    ) -> list[UnifiedResult]:
        """
        Convert Mistral batch results to unified format.

        Mistral result format:
        {
            "custom_id": "...",
            "status": 200 | 4xx | 5xx,
            "body": {
                "id": "...",
                "object": "chat.completion",
                "model": "...",
                "choices": [...],
                "usage": {...}
            } (if status 200)
            or
            "body": {
                "error": {...}
            } (if error)
        }

        Args:
            provider_results: Raw results from Mistral

        Returns:
            List of unified results
        """
        unified_results = []

        for result in provider_results:
            custom_id = result.get("custom_id", "")
            status_code = result.get("status", 500)
            body = result.get("body", {})

            # Determine status based on HTTP status code
            if status_code == 200:
                status = ResultStatus.SUCCEEDED
                response = body
                error = None
            else:
                status = ResultStatus.ERRORED
                response = None
                error = body.get("error", {"message": f"HTTP {status_code}", "type": "api_error"})

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
        Send batch to Mistral Batch API.

        Steps:
        1. Validate modalities (will raise error if audio is present)
        2. Convert requests to Mistral format
        3. Save unified and provider JSONL files
        4. Upload JSONL file to Mistral
        5. Create batch job via API
        6. Return batch job ID

        Args:
            batch: Batch metadata with unified requests

        Returns:
            batch_id: Mistral batch job ID

        Raises:
            UnsupportedModalityError: If audio content is present
            ValueError: If requests are invalid
            Exception: If API call fails
        """
        # Validate modalities FIRST (will raise error if audio is present)
        self.validate_request_modalities(batch.requests)
        
        # Convert to Mistral format
        mistral_requests = self._convert_to_provider_format(batch.requests)

        # Create temporary JSONL file for upload
        temp_jsonl_path = Path(f"/tmp/mistral_batch_temp_{id(batch)}.jsonl")

        try:
            # Write requests to temporary JSONL file
            async with aiofiles.open(temp_jsonl_path, mode='w', encoding='utf-8') as f:
                for item in mistral_requests:
                    await f.write(json.dumps(item) + '\n')

            # Upload file to Mistral
            with open(temp_jsonl_path, 'rb') as f:
                batch_data = self.client.files.upload(
                    file={
                        "file_name": temp_jsonl_path.name,
                        "content": f
                    },
                    purpose="batch"
                )

            # Get model from first request (all requests should use same model for Mistral batch)
            model = batch.requests[0].model if batch.requests else "mistral-small-latest"

            # Create batch job
            created_job = self.client.batch.jobs.create(
                input_files=[batch_data.id],
                model=model,
                endpoint=self.endpoint,
                metadata=batch.metadata or {}
            )

            batch_id = created_job.id

        except Exception as e:
            raise Exception(f"Failed to create Mistral batch: {str(e)}") from e
        finally:
            # Clean up temporary file
            if temp_jsonl_path.exists():
                temp_jsonl_path.unlink()

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
            self._write_jsonl_sync(provider_path, mistral_requests)
        except Exception as e:
            # Log but don't fail the batch
            print(f"Warning: Failed to save batch files: {str(e)}")

        return batch_id

    async def get_status(
        self,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a Mistral batch.

        Args:
            batch_id: Mistral batch job ID

        Returns:
            BatchStatusResponse with current status and counts

        Raises:
            Exception: If batch not found or API call fails
        """
        try:
            retrieved_job = self.client.batch.jobs.get(job_id=batch_id)
        except Exception as e:
            raise Exception(f"Failed to retrieve batch status: {str(e)}") from e

        # Map Mistral status to unified BatchStatus
        job_status = retrieved_job.status

        status_map = {
            "QUEUED": BatchStatus.VALIDATING,
            "RUNNING": BatchStatus.IN_PROGRESS,
            "SUCCESS": BatchStatus.COMPLETED,
            "FAILED": BatchStatus.FAILED,
            "TIMEOUT_EXCEEDED": BatchStatus.FAILED,
            "CANCELLATION_REQUESTED": BatchStatus.IN_PROGRESS,
            "CANCELLED": BatchStatus.CANCELLED
        }
        status = status_map.get(job_status, BatchStatus.IN_PROGRESS)

        # Calculate request counts based on job status
        # Mistral doesn't provide detailed per-request counts in the job status
        # We estimate based on total_requests if available
        total_requests = getattr(retrieved_job, 'total_requests', 0)

        if status == BatchStatus.COMPLETED:
            request_counts = RequestCounts(
                total=total_requests,
                processing=0,
                succeeded=total_requests,  # Assume all succeeded if job completed
                errored=0,
                cancelled=0,
                expired=0
            )
        elif status == BatchStatus.FAILED:
            request_counts = RequestCounts(
                total=total_requests,
                processing=0,
                succeeded=0,
                errored=total_requests,  # Assume all failed if job failed
                cancelled=0,
                expired=0
            )
        elif status == BatchStatus.CANCELLED:
            request_counts = RequestCounts(
                total=total_requests,
                processing=0,
                succeeded=0,
                errored=0,
                cancelled=total_requests,
                expired=0
            )
        else:
            # Still processing
            request_counts = RequestCounts(
                total=total_requests,
                processing=total_requests,
                succeeded=0,
                errored=0,
                cancelled=0,
                expired=0
            )

        # Build response
        return BatchStatusResponse(
            batch_id=batch_id,
            provider="mistral",
            status=status,
            request_counts=request_counts,
            created_at=getattr(retrieved_job, 'created_at', ''),
            completed_at=getattr(retrieved_job, 'completed_at', None),
            expires_at=None,  # Mistral doesn't provide expiration
            provider_data={
                "status": job_status,
                "model": getattr(retrieved_job, 'model', None),
                "endpoint": getattr(retrieved_job, 'endpoint', None),
                "input_files": getattr(retrieved_job, 'input_files', []),
                "output_file": getattr(retrieved_job, 'output_file', None),
                "error_file": getattr(retrieved_job, 'error_file', None)
            }
        )

    async def get_results(
        self,
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed Mistral batch.

        Steps:
        1. Check if batch is complete
        2. Download results from Mistral
        3. Save raw results to file
        4. Convert to unified format
        5. Save unified results
        6. Yield each result

        Args:
            batch_id: Mistral batch job ID

        Yields:
            UnifiedResult objects

        Raises:
            Exception: If batch not complete or results unavailable
        """
        # Check batch status first
        status = await self.get_status(batch_id)
        if not status.is_complete():
            raise Exception(
                f"Batch {batch_id} is not complete yet. "
                f"Status: {status.status.value}"
            )

        # Get output file ID
        output_file = status.provider_data.get("output_file")
        if not output_file:
            raise Exception(f"No output file available for batch {batch_id}")

        # Download results from Mistral
        try:
            output_file_stream = self.client.files.download(file_id=output_file)

            # Read and parse JSONL results
            raw_results = []
            content = output_file_stream.read().decode('utf-8')

            for line in content.strip().split('\n'):
                if line:
                    result = json.loads(line)
                    raw_results.append(result)

                    # Convert and yield immediately
                    unified_result = self._convert_from_provider_format([result])[0]
                    yield unified_result

            # Load batch metadata for consistent file naming
            custom_name, model = self._load_batch_metadata(batch_id)

            # Save raw results
            output_path = self.get_batch_file_path(batch_id, "output", custom_name, model)
            self._write_jsonl_sync(output_path, raw_results)

            # Save unified results
            unified_results = self._convert_from_provider_format(raw_results)
            results_path = self.get_batch_file_path(batch_id, "results", custom_name, model)
            self._write_jsonl_sync(
                results_path,
                [
                    {
                        "custom_id": r.custom_id,
                        "status": r.status.value,
                        "response": r.response,
                        "error": r.error
                    }
                    for r in unified_results
                ]
            )

        except Exception as e:
            raise Exception(f"Failed to retrieve batch results: {str(e)}") from e

    async def cancel_batch(
        self,
        batch_id: str
    ) -> bool:
        """
        Cancel a running Mistral batch.

        Args:
            batch_id: Mistral batch job ID

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
            self.client.batch.jobs.cancel(job_id=batch_id)
            return True

        except Exception as e:
            raise Exception(f"Failed to cancel batch: {str(e)}") from e

    async def list_batches(
        self,
        limit: int = 20
    ) -> list[BatchStatusResponse]:
        """
        List recent Mistral batches.

        Args:
            limit: Maximum number of batches to return

        Returns:
            List of batch status responses
        """
        try:
            # Mistral list_job returns a paginated response
            list_response = self.client.batch.jobs.list()

            results = []
            count = 0

            # Iterate through jobs
            for job in list_response.data:
                if count >= limit:
                    break

                # Map status
                job_status = job.status
                status_map = {
                    "QUEUED": BatchStatus.VALIDATING,
                    "RUNNING": BatchStatus.IN_PROGRESS,
                    "SUCCESS": BatchStatus.COMPLETED,
                    "FAILED": BatchStatus.FAILED,
                    "TIMEOUT_EXCEEDED": BatchStatus.FAILED,
                    "CANCELLATION_REQUESTED": BatchStatus.IN_PROGRESS,
                    "CANCELLED": BatchStatus.CANCELLED
                }
                status = status_map.get(job_status, BatchStatus.IN_PROGRESS)

                # Estimate counts
                total_requests = getattr(job, 'total_requests', 0)

                if status == BatchStatus.COMPLETED:
                    request_counts = RequestCounts(
                        total=total_requests,
                        processing=0,
                        succeeded=total_requests,
                        errored=0,
                        cancelled=0,
                        expired=0
                    )
                elif status == BatchStatus.FAILED:
                    request_counts = RequestCounts(
                        total=total_requests,
                        processing=0,
                        succeeded=0,
                        errored=total_requests,
                        cancelled=0,
                        expired=0
                    )
                elif status == BatchStatus.CANCELLED:
                    request_counts = RequestCounts(
                        total=total_requests,
                        processing=0,
                        succeeded=0,
                        errored=0,
                        cancelled=total_requests,
                        expired=0
                    )
                else:
                    request_counts = RequestCounts(
                        total=total_requests,
                        processing=total_requests,
                        succeeded=0,
                        errored=0,
                        cancelled=0,
                        expired=0
                    )

                results.append(BatchStatusResponse(
                    batch_id=job.id,
                    provider="mistral",
                    status=status,
                    request_counts=request_counts,
                    created_at=getattr(job, 'created_at', ''),
                    completed_at=getattr(job, 'completed_at', None),
                    expires_at=None,
                    provider_data={
                        "status": job_status,
                        "model": getattr(job, 'model', None),
                        "endpoint": getattr(job, 'endpoint', None)
                    }
                ))

                count += 1

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
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

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
