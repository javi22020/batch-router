"""Google GenAI Batch API provider implementation."""

import json
import os
from typing import Optional, Any, AsyncIterator
from pathlib import Path
from datetime import datetime

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise ImportError(
        "google-genai package is required for GoogleProvider. "
        "Install with: pip install google-genai>=1.0.0"
    )

from ..core.base import BaseProvider
from ..core.requests import UnifiedRequest, UnifiedBatchMetadata
from ..core.responses import BatchStatusResponse, UnifiedResult, RequestCounts
from ..core.enums import BatchStatus, ResultStatus, Modality
from ..core.content import TextContent, ImageContent, DocumentContent, AudioContent
from ..exceptions import (
    ProviderError,
    BatchNotFoundError,
    BatchNotCompleteError,
)
from ..utilities.file_manager import FileManager


class GoogleProvider(BaseProvider):
    """
    Google GenAI Batch API provider.

    Uses the google-genai SDK to process batch requests through Google's
    Gemini batch processing API. Supports text, images, documents, and audio.
    """
    
    # Declare supported modalities
    supported_modalities = {
        Modality.TEXT,
        Modality.IMAGE,
        Modality.DOCUMENT,
        Modality.AUDIO
    }

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Google provider.

        Args:
            api_key: Google API key (if not provided, will use GOOGLE_API_KEY env var)
            **kwargs: Additional provider configuration
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("GOOGLE_API_KEY")

        super().__init__(name="google", api_key=api_key, **kwargs)

        # Initialize Google GenAI client
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ProviderError(
                "google",
                f"Failed to initialize Google GenAI client: {str(e)}",
                e
            )

    def _validate_configuration(self) -> None:
        """Validate Google provider configuration."""
        if not self.api_key:
            raise ValueError(
                "Google API key is required. "
                "Provide via api_key parameter or GOOGLE_API_KEY environment variable."
            )

    def _convert_content_to_google_format(
        self,
        content_item: TextContent | ImageContent | DocumentContent | AudioContent
    ) -> dict[str, Any]:
        """
        Convert a single content item to Google format (updated for audio).

        Args:
            content_item: Unified content object

        Returns:
            Google-formatted content part
        """
        if isinstance(content_item, TextContent):
            return {"text": content_item.text}

        elif isinstance(content_item, ImageContent):
            if content_item.source_type == "base64":
                return {
                    "inline_data": {
                        "mime_type": content_item.media_type,
                        "data": content_item.data
                    }
                }
            elif content_item.source_type == "file_uri":
                # Google Cloud Storage URI (gs://)
                return {
                    "file_data": {
                        "mime_type": content_item.media_type,
                        "file_uri": content_item.data
                    }
                }
            elif content_item.source_type == "url":
                # For URLs, we'd need to download and convert to base64
                # For now, raise an error as it's not directly supported
                raise ValueError(
                    "URL source type for images is not directly supported by Google. "
                    "Convert to base64 or upload to Google Cloud Storage."
                )

        elif isinstance(content_item, DocumentContent):
            if content_item.source_type == "base64":
                return {
                    "inline_data": {
                        "mime_type": content_item.media_type,
                        "data": content_item.data
                    }
                }
            elif content_item.source_type == "file_uri":
                return {
                    "file_data": {
                        "mime_type": content_item.media_type,
                        "file_uri": content_item.data
                    }
                }
            elif content_item.source_type == "url":
                raise ValueError(
                    "URL source type for documents is not directly supported by Google. "
                    "Convert to base64 or upload to Google Cloud Storage."
                )
        
        elif isinstance(content_item, AudioContent):
            # Google supports two approaches for audio:
            # 1. Inline data (base64): For audio < 20MB total request size
            # 2. File URI (file_uri): For larger files or reused audio via Files API
            
            # Validate media_type - Google is strict about MIME types
            valid_google_mimes = {
                "audio/wav", "audio/wave",
                "audio/mp3", "audio/mpeg"
            }
            if content_item.media_type not in valid_google_mimes:
                raise ValueError(
                    f"Invalid audio MIME type for Google: {content_item.media_type}. "
                    f"Supported types: {', '.join(sorted(valid_google_mimes))}"
                )
            
            if content_item.source_type == "base64":
                # Inline data approach
                return {
                    "inline_data": {
                        "mime_type": content_item.media_type,
                        "data": content_item.data
                    }
                }
            elif content_item.source_type == "file_uri":
                # File URI approach (for Files API uploaded files)
                if not content_item.data.startswith("gs://"):
                    raise ValueError(
                        "Google file_uri must use gs:// URI format from Files API. "
                        f"Got: {content_item.data[:50]}..."
                    )
                return {
                    "file_data": {
                        "mime_type": content_item.media_type,
                        "file_uri": content_item.data
                    }
                }
            elif content_item.source_type == "url":
                # Google doesn't support direct URLs for audio
                raise ValueError(
                    "Google Gemini batch API does not support URL source_type for audio. "
                    "Use base64 for inline data or file_uri for uploaded files via Files API."
                )

        raise ValueError(f"Unsupported content type: {type(content_item)}")

    def _convert_to_provider_format(
        self,
        requests: list[UnifiedRequest]
    ) -> list[dict[str, Any]]:
        """
        Convert unified requests to Google Batch API format.

        Google format for JSONL input:
        {
            "key": "custom_id",
            "request": {
                "contents": [...],
                "config": {
                    "systemInstruction": {...},
                    "generationConfig": {...}
                }
            }
        }

        Args:
            requests: List of unified requests

        Returns:
            List of Google-formatted request dictionaries
        """
        google_requests = []

        for request in requests:
            # Convert messages to Google contents format
            contents = []
            for msg in request.messages:
                parts = []
                for content_item in msg.content:
                    parts.append(self._convert_content_to_google_format(content_item))

                contents.append({
                    "role": msg.role,
                    "parts": parts
                })

            # Build the request config
            request_config: dict[str, Any] = {}

            # Add system instruction if present
            if request.system_prompt:
                system_text = (
                    request.system_prompt
                    if isinstance(request.system_prompt, str)
                    else "\n".join(request.system_prompt)
                )
                request_config["systemInstruction"] = {
                    "parts": [{"text": system_text}]
                }

            # Add generation config if present
            if request.generation_config:
                gen_config = {}

                # Map unified params to Google params
                if request.generation_config.temperature is not None:
                    gen_config["temperature"] = request.generation_config.temperature

                if request.generation_config.max_tokens is not None:
                    gen_config["maxOutputTokens"] = request.generation_config.max_tokens

                if request.generation_config.top_p is not None:
                    gen_config["topP"] = request.generation_config.top_p

                if request.generation_config.top_k is not None:
                    gen_config["topK"] = request.generation_config.top_k

                if request.generation_config.stop_sequences:
                    gen_config["stopSequences"] = request.generation_config.stop_sequences

                if request.generation_config.presence_penalty is not None:
                    gen_config["presencePenalty"] = request.generation_config.presence_penalty

                if gen_config:
                    request_config["generationConfig"] = gen_config

            # Merge provider-specific kwargs
            if request.provider_kwargs:
                request_config.update(request.provider_kwargs)

            # Build final Google request
            google_request = {
                "key": request.custom_id,
                "request": {
                    "contents": contents,
                }
            }

            # Add config only if not empty
            if request_config:
                google_request["request"]["config"] = request_config

            google_requests.append(google_request)

        return google_requests

    def _convert_from_provider_format(
        self,
        provider_results: list[dict[str, Any]]
    ) -> list[UnifiedResult]:
        """
        Convert Google results to unified format.

        Google result format:
        {
            "key": "custom_id",
            "response": {...} or "error": {...}
        }

        Args:
            provider_results: Raw results from Google

        Returns:
            List of unified results
        """
        unified_results = []

        for result in provider_results:
            custom_id = result.get("key", "unknown")

            # Check if response or error
            if "response" in result:
                # Successful response
                unified_results.append(UnifiedResult(
                    custom_id=custom_id,
                    status=ResultStatus.SUCCEEDED,
                    response=result["response"],
                    provider_data=result
                ))
            elif "error" in result:
                # Error response
                unified_results.append(UnifiedResult(
                    custom_id=custom_id,
                    status=ResultStatus.ERRORED,
                    error=result["error"],
                    provider_data=result
                ))
            else:
                # Unknown format
                unified_results.append(UnifiedResult(
                    custom_id=custom_id,
                    status=ResultStatus.ERRORED,
                    error={"message": "Unknown result format", "details": result},
                    provider_data=result
                ))

        return unified_results

    async def send_batch(
        self,
        batch: UnifiedBatchMetadata
    ) -> str:
        """
        Send batch to Google GenAI Batch API with audio support.

        Implementation steps:
        1. Validate modalities
        2. Convert requests to Google format
        3. Save unified format JSONL
        4. Save provider format JSONL
        5. Upload JSONL file using File API
        6. Create batch job
        7. Return batch_id (job name)

        Args:
            batch: Batch metadata with unified requests

        Returns:
            batch_id: Google batch job name (e.g., "batches/xyz123")

        Raises:
            UnsupportedModalityError: If unsupported content modality is present
            ProviderError: If batch creation fails
        """
        # Validate modalities FIRST
        self.validate_request_modalities(batch.requests)
        
        try:
            # Generate a simple batch ID for file tracking
            # We'll use timestamp-based ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            local_batch_id = f"google_{timestamp}"

            # Extract custom naming parameters
            custom_name = batch.name
            model = batch.requests[0].model if batch.requests else None

            # Save metadata for later use in get_results
            self._save_batch_metadata(local_batch_id, custom_name, model)

            # Step 1: Convert to Google format
            google_requests = self._convert_to_provider_format(batch.requests)

            # Step 2: Save unified format
            unified_path = self.get_batch_file_path(local_batch_id, "unified", custom_name, model)
            await FileManager.write_jsonl(
                unified_path,
                [req.to_dict() for req in batch.requests]
            )

            # Step 3: Save provider format
            provider_path = self.get_batch_file_path(local_batch_id, "provider", custom_name, model)
            await FileManager.write_jsonl(provider_path, google_requests)

            # Step 4: Upload JSONL file to Google
            uploaded_file = self.client.files.upload(
                file=str(provider_path),
                config=genai_types.UploadFileConfig(
                    display_name=f"batch_{local_batch_id}",
                    mime_type="application/jsonl"
                )
            )

            # Step 5: Create batch job
            # Get model from first request (assume all use same model)
            model = batch.requests[0].model
            display_name = batch.metadata.get("display_name", f"batch_{local_batch_id}")

            batch_job = self.client.batches.create(
                model=model,
                src=uploaded_file.name,
                config={
                    "display_name": display_name,
                }
            )

            # Store mapping from Google job name to our local batch ID
            # This is useful for file management
            mapping_path = self.get_batch_file_path(local_batch_id, "metadata")
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mapping_path, 'w') as f:
                json.dump({
                    "google_job_name": batch_job.name,
                    "local_batch_id": local_batch_id,
                    "uploaded_file_name": uploaded_file.name,
                    "model": model,
                    "display_name": display_name,
                }, f)

            # Return the Google job name as batch_id
            return batch_job.name

        except Exception as e:
            raise ProviderError(
                "google",
                f"Failed to send batch: {str(e)}",
                e
            )

    def _get_local_batch_id(self, batch_id: str) -> str:
        """
        Get local batch ID from Google job name.

        This searches for the metadata file that contains the mapping.

        Args:
            batch_id: Google job name

        Returns:
            Local batch ID for file management
        """
        # Search through metadata files in the google directory
        batch_dir = Path(".batch_router/generated") / self.name
        if not batch_dir.exists():
            raise BatchNotFoundError(f"Batch directory not found")

        for metadata_file in batch_dir.glob("batch_*_metadata.jsonl"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get("google_job_name") == batch_id:
                        return metadata["local_batch_id"]
            except Exception:
                continue

        # If not found, just extract from job name or use job name directly
        # Google job names are like "batches/xyz123"
        return batch_id.replace("/", "_").replace("batches_", "google_")

    async def get_status(
        self,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a Google batch job.

        Args:
            batch_id: Google batch job name (e.g., "batches/xyz123")

        Returns:
            Batch status information

        Raises:
            BatchNotFoundError: If job doesn't exist
            ProviderError: If status check fails
        """
        try:
            # Get job status from Google
            batch_job = self.client.batches.get(name=batch_id)

            # Map Google job state to unified status
            state_mapping = {
                "JOB_STATE_PENDING": BatchStatus.IN_PROGRESS,
                "JOB_STATE_RUNNING": BatchStatus.IN_PROGRESS,
                "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
                "JOB_STATE_FAILED": BatchStatus.FAILED,
                "JOB_STATE_CANCELLED": BatchStatus.CANCELLED,
                "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
            }

            state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
            unified_status = state_mapping.get(state_name, BatchStatus.IN_PROGRESS)

            # Extract request counts from batch stats
            batch_stats = getattr(batch_job, 'batch_stats', None)
            if batch_stats:
                total = getattr(batch_stats, 'total_request_count', 0)
                succeeded = getattr(batch_stats, 'succeeded_request_count', 0)
                failed = getattr(batch_stats, 'failed_request_count', 0)

                # Calculate processing (pending + running)
                processing = total - succeeded - failed
                processing = max(0, processing)  # Ensure non-negative

                request_counts = RequestCounts(
                    total=total,
                    processing=processing,
                    succeeded=succeeded,
                    errored=failed,
                    cancelled=0,  # Google doesn't separate cancelled from failed
                    expired=0
                )
            else:
                # No stats available, estimate based on status
                if unified_status == BatchStatus.COMPLETED:
                    request_counts = RequestCounts(total=1, succeeded=1)
                elif unified_status == BatchStatus.FAILED:
                    request_counts = RequestCounts(total=1, errored=1)
                else:
                    request_counts = RequestCounts(total=1, processing=1)

            # Extract timestamps
            created_at = getattr(batch_job, 'create_time', None)
            if created_at:
                created_at = str(created_at)
            else:
                created_at = datetime.now().isoformat()

            completed_at = None
            if unified_status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED]:
                update_time = getattr(batch_job, 'update_time', None)
                if update_time:
                    completed_at = str(update_time)

            return BatchStatusResponse(
                batch_id=batch_id,
                provider="google",
                status=unified_status,
                request_counts=request_counts,
                created_at=created_at,
                completed_at=completed_at,
                expires_at=None,  # Google doesn't provide expiry timestamp upfront
                provider_data={
                    "state": state_name,
                    "job_name": batch_job.name,
                    "display_name": getattr(batch_job, 'display_name', None),
                }
            )

        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                raise BatchNotFoundError(f"Batch job '{batch_id}' not found")
            raise ProviderError(
                "google",
                f"Failed to get batch status: {str(e)}",
                e
            )

    async def get_results(
        self,
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed Google batch job.

        Implementation steps:
        1. Check if batch is complete
        2. Download results file from Google
        3. Save raw results to output.jsonl
        4. Convert to unified format
        5. Save unified results
        6. Yield each result

        Args:
            batch_id: Google batch job name

        Yields:
            UnifiedResult objects

        Raises:
            BatchNotCompleteError: If batch is still processing
            BatchNotFoundError: If batch doesn't exist
            ProviderError: If result retrieval fails
        """
        try:
            # Step 1: Check status
            status = await self.get_status(batch_id)
            if not status.is_complete():
                raise BatchNotCompleteError(
                    f"Batch '{batch_id}' is still {status.status.value}. "
                    "Wait for completion before retrieving results."
                )

            # Step 2: Get job details
            batch_job = self.client.batches.get(name=batch_id)

            # Get local batch ID for file management
            local_batch_id = self._get_local_batch_id(batch_id)

            # Step 3: Download results
            results_data = []

            # Check if results are in a file or inline
            if hasattr(batch_job, 'dest') and batch_job.dest:
                if hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                    # Results are in a file - download it
                    result_file_name = batch_job.dest.file_name
                    file_content = self.client.files.download(file=result_file_name)

                    # Parse JSONL content
                    content_str = file_content.decode('utf-8')
                    for line in content_str.strip().split('\n'):
                        if line.strip():
                            results_data.append(json.loads(line))

                elif hasattr(batch_job.dest, 'inlined_responses') and batch_job.dest.inlined_responses:
                    # Results are inline
                    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                        result_dict = {"key": f"inline_{i}"}
                        if hasattr(inline_response, 'response') and inline_response.response:
                            # Convert response object to dict
                            response_dict = self._response_to_dict(inline_response.response)
                            result_dict["response"] = response_dict
                        elif hasattr(inline_response, 'error') and inline_response.error:
                            result_dict["error"] = self._error_to_dict(inline_response.error)
                        results_data.append(result_dict)

            # Load batch metadata for consistent file naming
            custom_name, model = self._load_batch_metadata(local_batch_id)

            # Step 4: Save raw output
            if results_data:
                output_path = self.get_batch_file_path(local_batch_id, "output", custom_name, model)
                await FileManager.write_jsonl(output_path, results_data)

            # Step 5: Convert to unified format
            unified_results = self._convert_from_provider_format(results_data)

            # Step 6: Save unified results
            results_path = self.get_batch_file_path(local_batch_id, "results", custom_name, model)
            await FileManager.write_jsonl(
                results_path,
                [
                    {
                        "custom_id": r.custom_id,
                        "status": r.status.value,
                        "response": r.response,
                        "error": r.error,
                    }
                    for r in unified_results
                ]
            )

            # Step 7: Yield results
            for result in unified_results:
                yield result

        except BatchNotCompleteError:
            raise
        except BatchNotFoundError:
            raise
        except Exception as e:
            raise ProviderError(
                "google",
                f"Failed to retrieve results: {str(e)}",
                e
            )

    def _response_to_dict(self, response: Any) -> dict[str, Any]:
        """Convert Google response object to dictionary."""
        # Try to convert using common attributes
        result = {}

        if hasattr(response, 'candidates'):
            result['candidates'] = []
            for candidate in response.candidates:
                cand_dict = {}
                if hasattr(candidate, 'content'):
                    content = candidate.content
                    cand_dict['content'] = {
                        'parts': [
                            {'text': part.text} if hasattr(part, 'text') else str(part)
                            for part in (content.parts if hasattr(content, 'parts') else [])
                        ],
                        'role': getattr(content, 'role', 'model')
                    }
                result['candidates'].append(cand_dict)

        # Try text shortcut
        if hasattr(response, 'text'):
            result['text'] = response.text

        # If we couldn't extract anything, convert to string
        if not result:
            result = {'raw': str(response)}

        return result

    def _error_to_dict(self, error: Any) -> dict[str, Any]:
        """Convert Google error object to dictionary."""
        if isinstance(error, dict):
            return error

        result = {}
        if hasattr(error, 'message'):
            result['message'] = error.message
        if hasattr(error, 'code'):
            result['code'] = error.code
        if hasattr(error, 'details'):
            result['details'] = str(error.details)

        if not result:
            result = {'message': str(error)}

        return result

    async def cancel_batch(
        self,
        batch_id: str
    ) -> bool:
        """
        Cancel a running Google batch job.

        Args:
            batch_id: Google batch job name

        Returns:
            True if cancelled successfully, False if already complete

        Raises:
            BatchNotFoundError: If batch doesn't exist
            ProviderError: If cancellation fails
        """
        try:
            # Check current status
            status = await self.get_status(batch_id)

            # If already complete, return False
            if status.is_complete():
                return False

            # Cancel the job
            self.client.batches.cancel(name=batch_id)

            return True

        except BatchNotFoundError:
            raise
        except Exception as e:
            raise ProviderError(
                "google",
                f"Failed to cancel batch: {str(e)}",
                e
            )

    async def list_batches(
        self,
        limit: int = 20
    ) -> list[BatchStatusResponse]:
        """
        List recent Google batch jobs.

        Note: This requires iterating through local metadata files as Google's
        list API may not be directly accessible through the SDK.

        Args:
            limit: Maximum number of batches to return

        Returns:
            List of batch status responses
        """
        batch_statuses = []

        try:
            # Look through local metadata files
            batch_dir = Path(".batch_router/generated") / self.name
            if not batch_dir.exists():
                return []

            metadata_files = sorted(
                batch_dir.glob("batch_*_metadata.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            for metadata_file in metadata_files[:limit]:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        google_job_name = metadata.get("google_job_name")
                        if google_job_name:
                            status = await self.get_status(google_job_name)
                            batch_statuses.append(status)
                except Exception:
                    # Skip files that can't be read
                    continue

            return batch_statuses

        except Exception as e:
            raise ProviderError(
                "google",
                f"Failed to list batches: {str(e)}",
                e
            )
