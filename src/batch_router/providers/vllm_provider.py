"""vLLM Batch API provider implementation."""

import json
import os
from datetime import datetime
from typing import Optional, Any, AsyncIterator
from pathlib import Path
from pydantic import BaseModel, Field
import asyncio

from batch_router.core.messages import UnifiedMessage

from openai import (
    AsyncOpenAI as AsyncvLLM,
    OpenAI as vLLM
)

from ..core.base import BaseProvider
from ..core.requests import UnifiedRequest, UnifiedBatchMetadata
from ..core.responses import BatchStatusResponse, UnifiedResult, RequestCounts, OutputPaths
from ..core.enums import BatchStatus, ResultStatus, Modality
from ..core.content import TextContent, ImageContent, DocumentContent, AudioContent

class RunningTask(BaseModel):
    batch_id: str
    status: BatchStatus = Field(default=BatchStatus.IN_PROGRESS)
    output_paths: OutputPaths
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)
    task: asyncio.Task

class vLLMProvider(BaseProvider):
    """
    vLLM Batch API provider implementation.

    Supports:
    - Chat Completions API via batch processing
    - Text and multimodal content (images, audio via base64)
    - System prompt conversion to system message
    - 50% cost reduction compared to synchronous API
    - 24-hour completion window

    Usage:
        provider = vLLMAIProvider(api_key="sk-...")
        batch_id = await provider.send_batch(batch_metadata)
        status = await provider.get_status(batch_id)
        async for result in provider.get_results(batch_id):
            print(result.custom_id, result.status)
    """
    
    # Declare supported modalities
    supported_modalities = {Modality.TEXT, Modality.IMAGE, Modality.AUDIO}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        vllm_command: str = "vllm",
        **kwargs
    ):
        """
        Initialize vLLM provider.

        Args:
            api_key: vLLM API key (or set VLLM_API_KEY env var)
            base_url: Optional custom base URL for API
            vllm_command: Optional vLLM command
            additional_args: Optional additional CLI arguments for run-batch command
            **kwargs: Additional configuration options
        """
        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("VLLM_API_KEY") or "EMPTY"
        base_url = base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8005"
        self.api_key = api_key
        self.base_url = base_url
        self.vllm_command = vllm_command
        self.running_tasks: dict[str, RunningTask] = {}

        super().__init__(name="vllm", api_key=api_key, **kwargs)

        # Initialize both sync and async clients
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url
        }

        self.client = vLLM(**client_kwargs)
        self.async_client = AsyncvLLM(**client_kwargs)

    def _validate_configuration(self) -> None:
        pass # No validation needed

    def _generate_batch_id(self) -> str:
        return f"vllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _convert_to_provider_format(
        self,
        requests: list[UnifiedRequest]
    ) -> list[dict[str, Any]]:
        """
        Convert unified requests to vLLM Batch API format (updated for audio).

        vLLM Batch format:
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "model_name",
                "messages": [...],
                "modalities": ["text", "audio"],  # Added when audio is present
                "max_tokens": 1000,
                ...
            }
        }

        System prompt conversion:
        - Unified: request.system_prompt
        - vLLM: Prepended as message with role="system" (vLLM-compatible)
        """
        provider_requests = []

        for request in requests:
            # Check if request contains audio
            has_audio = any(
                isinstance(content, AudioContent)
                for message in request.messages
                for content in message.content
            )
            
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

            # Convert unified messages to vLLM format
            for msg in request.messages:
                vllm_message = self._convert_message_to_vllm(msg)
                messages.append(vllm_message)

            # Build request body
            body: dict[str, Any] = {
                "model": request.model,
                "messages": messages
            }
            
            # Add modalities if audio is present
            if has_audio:
                body["modalities"] = ["text", "audio"]

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

            # Create vLLM batch request format
            provider_request = {
                "custom_id": request.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }

            provider_requests.append(provider_request)

        return provider_requests

    def _convert_message_to_vllm(self, message: UnifiedMessage) -> dict[str, Any]:
        """Convert unified message to vLLM message format (updated for audio)."""
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
                image_part = self._convert_image_to_vllm(content_item)
                content_parts.append(image_part)
            elif isinstance(content_item, AudioContent):
                audio_part = self._convert_audio_to_vllm(content_item)
                content_parts.append(audio_part)
            elif isinstance(content_item, DocumentContent):
                # vLLM doesn't support documents in chat completions
                # Skip or raise error
                pass

        return {
            "role": message.role,
            "content": content_parts
        }

    def _convert_image_to_vllm(self, image: ImageContent) -> dict[str, Any]:
        """Convert unified image content to vLLM format."""
        if image.source_type == "url":
            return {
                "type": "image_url",
                "image_url": {
                    "url": image.data
                }
            }
        elif image.source_type == "base64":
            # vLLM expects: data:image/jpeg;base64,<base64_string>
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
    
    def _convert_audio_to_vllm(self, audio: AudioContent) -> dict[str, Any]:
        """
        Convert unified audio content to vLLM format.
        
        vLLM expects audio in the following format:
        {
            "type": "input_audio",
            "input_audio": {
                "data": "<base64_string>",
                "format": "wav"  # or "mp3"
            }
        }
        
        The request must also include:
        {
            "modalities": ["text", "audio"],
            ...
        }
        
        Note: vLLM uses "format" field with simple extension names,
        not full MIME types.
        """
        if audio.source_type != "base64":
            raise ValueError(
                "vLLM batch API only supports base64-encoded audio. "
                f"Got source_type={audio.source_type}. "
                "Convert URL or file_uri audio to base64 first."
            )
        
        # Extract format from media_type and normalize
        # "audio/wav" -> "wav", "audio/mp3" -> "mp3", "audio/mpeg" -> "mp3"
        if audio.media_type in ("audio/wav", "audio/wave"):
            audio_format = "wav"
        elif audio.media_type in ("audio/mp3", "audio/mpeg"):
            audio_format = "mp3"
        else:
            # Should never happen due to AudioContent validation, but be defensive
            raise ValueError(
                f"Unsupported audio format for vLLM: {audio.media_type}. "
                "Only WAV and MP3 are supported."
            )
        
        return {
            "type": "input_audio",
            "input_audio": {
                "data": audio.data,
                "format": audio_format
            }
        }

    def _convert_from_provider_format(
        self,
        provider_results: list[dict[str, Any]]
    ) -> list[UnifiedResult]:
        """
        Convert vLLM batch results to unified format.

        vLLM result format:
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
                    "model": "model_name",
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
    
    async def _run_vllm_batch(
        self,
        batch_id: str,
        input_file: str | Path,
        output_paths: OutputPaths,
        model: str | Path
    ):
        output_file = output_paths.raw_output_batch_jsonl
        cmd = [
                self.vllm_command,
                "run-batch",
                "-i", input_file if isinstance(input_file, str) else str(input_file),
                "-o", output_file if isinstance(output_file, str) else str(output_file),
                "--model", model if isinstance(model, str) else str(model)
            ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        _ = await process.communicate()

        out_vllm = await self._read_jsonl(output_file)

        return batch_id, out_vllm, output_paths
    
    def _vllm_done_callback(
        self,
        task: asyncio.Task[tuple[str, list[dict[str, Any]], OutputPaths]]
    ):
        batch_id, out_vllm, output_paths = task.result()
        output_unified_file = output_paths.unified_output_jsonl

        unified_results = self._convert_from_provider_format(out_vllm)

        unified_dicts = [
            unified_result.model_dump() for unified_result in unified_results
        ]

        self._write_jsonl_sync(output_unified_file, unified_dicts)

        self.running_tasks[batch_id].status = BatchStatus.COMPLETED
        self.running_tasks[batch_id].output_paths = output_paths
        self.running_tasks[batch_id].completed_at = datetime.now()

    async def send_batch(
        self,
        batch: UnifiedBatchMetadata
    ) -> str:
        """
        Send batch to vLLM with audio support.

        Steps:
        1. Validate modalities
        2. Convert requests to vLLM format
        3. Save unified format JSONL
        4. Save provider format JSONL
        5. Upload file to vLLM
        6. Create batch job
        7. Return batch ID
        """
        self.validate_request_modalities(batch.requests)

        batch_id = self._generate_batch_id()
        
        # Convert to provider format
        provider_requests = self._convert_to_provider_format(batch.requests)

        # Extract custom naming parameters
        custom_name = batch.name
        model = batch.requests[0].model if batch.requests else None

        # Save metadata for later use in get_results
        self._save_batch_metadata(batch_id, custom_name, model)

        # Now save the files with proper batch_id
        unified_path = self.get_batch_file_path(batch_id, "unified", custom_name, model)
        provider_path = self.get_batch_file_path(batch_id, "provider", custom_name, model)

        # Save unified format
        unified_data = [req.to_dict() for req in batch.requests]
        self._write_jsonl_sync(str(unified_path), unified_data)

        # Save provider format
        self._write_jsonl_sync(str(provider_path), provider_requests)

        # Get results path
        output_provider_file = self.get_batch_file_path(batch_id, "output", custom_name, model)
        output_unified_file = self.get_batch_file_path(batch_id, "results", custom_name, model)

        output_paths = OutputPaths(
            raw_output_batch_jsonl=output_provider_file,
            unified_output_jsonl=output_unified_file,
        )

        task = asyncio.create_task(
            self._run_vllm_batch(
                batch_id,
                provider_path,
                output_paths,
                model
            )
        )

        task.add_done_callback(self._vllm_done_callback)

        self.running_tasks[batch_id] = RunningTask(
            batch_id=batch_id,
            output_paths=output_paths,
            task=task
        )
        
        return batch_id

    async def get_status(
        self,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a batch.

        vLLM status values:
        - validating
        - failed
        - in_progress
        - finalizing
        - completed
        - expired
        - cancelling
        - cancelled
        """
        if batch_id not in self.running_tasks.keys():
            raise ValueError(f"Batch {batch_id} not found in running tasks")
        
        running_task = self.running_tasks[batch_id]
        status = running_task.status

        return BatchStatusResponse(
            batch_id=batch_id,
            provider="vllm",
            status=status,
            request_counts=RequestCounts(
                total=1,
            ),
            created_at=running_task.created_at,
            completed_at=running_task.completed_at,
            expires_at=None,
            provider_data={}
        )

    async def get_results(
        self,
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed batch.

        Steps:
        1. Check batch is complete
        2. Read output file
        3. Convert to unified format
        4. Yield each result
        """
        if batch_id not in self.running_tasks.keys():
            raise ValueError(f"Batch {batch_id} not found in running tasks")
        
        running_task = self.running_tasks[batch_id]
        status = running_task.status

        if status != BatchStatus.COMPLETED:
            raise ValueError(f"Batch {batch_id} is not complete. Current status: {status}")

        provider_results = await self._read_jsonl(running_task.output_paths.raw_output_batch_jsonl)

        unified_results = self._convert_from_provider_format(provider_results)

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
            if batch_id not in self.running_tasks.keys():
                raise ValueError(f"Batch {batch_id} not found in running tasks")
            
            running_task = self.running_tasks[batch_id]
            running_task.task.cancel()
            running_task.status = BatchStatus.CANCELLED
            running_task.completed_at = datetime.now()
            return True
        except Exception as e:
            print(f"Error cancelling batch {batch_id}: {e}")
            return False

    async def list_batches(
        self,
        limit: int = 20
    ) -> list[BatchStatusResponse]:
        """
        IMPORTANT: List batches in the current instance only.
        """
        results = []
        for batch_id in self.running_tasks.keys():
            running_task = self.running_tasks[batch_id]
            status = running_task.status

            request_counts = RequestCounts(
                total=1
            )

            created_at = running_task.created_at
            completed_at = running_task.completed_at
            expires_at = None

            status_response = BatchStatusResponse(
                batch_id=batch_id,
                provider="vllm",
                status=status,
                request_counts=request_counts,
                created_at=created_at,
                completed_at=completed_at,
                expires_at=expires_at,
                provider_data={
                    "output_file_id": running_task.output_paths.raw_output_batch_jsonl,
                    "unified_file_id": running_task.output_paths.unified_output_jsonl,
                }
            )

            results.append(status_response)

        return results
