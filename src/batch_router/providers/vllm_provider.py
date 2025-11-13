"""vLLM local batch processing provider implementation."""

import json
import os
import asyncio
import subprocess
from datetime import datetime
from typing import Optional, Any, AsyncIterator
from pathlib import Path

try:
    import aiofiles
except ImportError:
    aiofiles = None  # type: ignore

from ..core.base import BaseProvider
from ..core.requests import UnifiedRequest, UnifiedBatchMetadata
from ..core.messages import UnifiedMessage
from ..core.responses import BatchStatusResponse, UnifiedResult, RequestCounts
from ..core.enums import BatchStatus, ResultStatus, Modality
from ..core.content import TextContent, ImageContent, DocumentContent, AudioContent


class vLLMProvider(BaseProvider):
    """
    vLLM local batch processing provider implementation.

    Uses vLLM's offline batch inference with OpenAI batch file format.
    Runs locally via subprocess execution of `vllm run-batch` command.

    Supports:
    - Chat Completions API via local batch processing
    - Text and multimodal content (images, audio - depending on model support)
    - System prompt conversion to system message (OpenAI-compatible)
    - Local execution (no API costs)
    - Synchronous batch processing

    Usage:
        provider = vLLMProvider()
        batch_id = await provider.send_batch(batch_metadata)
        status = await provider.get_status(batch_id)
        async for result in provider.get_results(batch_id):
            print(result.custom_id, result.status)

    Note:
        - Requires vLLM to be installed and available in PATH
        - Batch processing runs synchronously in background
        - Models must be available locally or downloadable
        - Audio support depends on the model being used
    """
    
    # Declare supported modalities (model-dependent, but framework supports it)
    supported_modalities = {Modality.TEXT, Modality.IMAGE, Modality.AUDIO}

    def __init__(
        self,
        vllm_command: str = "vllm",
        additional_args: Optional[list[str]] = None,
        **kwargs
    ):
        """
        Initialize vLLM provider.

        Args:
            vllm_command: Path to vLLM executable (default: "vllm")
            additional_args: Additional CLI arguments to pass to vllm run-batch
            **kwargs: Additional configuration options
        """
        # Set attributes before calling super().__init__() because
        # _validate_configuration() is called from super().__init__()
        self.vllm_command = vllm_command
        self.additional_args = additional_args or []

        # Track running processes (batch_id -> process info)
        self.processes: dict[str, dict[str, Any]] = {}

        super().__init__(name="vllm", api_key=None, **kwargs)

        # Load existing batch metadata
        self._load_batch_metadata()

    def _validate_configuration(self) -> None:
        """Validate that vLLM is available."""
        # Check if vLLM is installed and available
        try:
            result = subprocess.run(
                [self.vllm_command, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise ValueError(
                    f"vLLM command '{self.vllm_command}' is not working properly. "
                    f"Error: {result.stderr}"
                )
        except FileNotFoundError:
            raise ValueError(
                f"vLLM command '{self.vllm_command}' not found in PATH. "
                "Please install vLLM: pip install vllm"
            )
        except subprocess.TimeoutExpired:
            raise ValueError(
                f"vLLM command '{self.vllm_command}' timed out. "
                "Please check your vLLM installation."
            )

    def _get_metadata_file_path(self) -> Path:
        """Get path to batch metadata file."""
        base_dir = Path(".batch_router/generated") / self.name
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "batch_metadata.json"

    def _load_batch_metadata(self) -> None:
        """Load batch metadata from file."""
        metadata_path = self._get_metadata_file_path()
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    self.processes = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.processes = {}
        else:
            self.processes = {}

    def _save_batch_metadata(self) -> None:
        """Save batch metadata to file."""
        metadata_path = self._get_metadata_file_path()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(self.processes, f, indent=2)

    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"vllm_{timestamp}"

    def _convert_to_provider_format(
        self,
        requests: list[UnifiedRequest]
    ) -> list[dict[str, Any]]:
        """
        Convert unified requests to vLLM (OpenAI-compatible) format (updated for audio).

        vLLM uses the same batch file format as OpenAI:
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "model-name",
                "messages": [...],
                "modalities": ["text", "audio"],  # Added when audio is present
                "max_completion_tokens": 1000,
                ...
            }
        }

        System prompt conversion:
        - Unified: request.system_prompt
        - vLLM: Prepended as message with role="system" (OpenAI-compatible)
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
                    body["max_completion_tokens"] = config.max_tokens
                if config.temperature is not None:
                    body["temperature"] = config.temperature
                if config.top_p is not None:
                    body["top_p"] = config.top_p
                if config.top_k is not None:
                    body["top_k"] = config.top_k
                if config.stop_sequences is not None:
                    body["stop"] = config.stop_sequences

            # Add provider-specific kwargs
            body.update(request.provider_kwargs)

            # Create vLLM batch request format (OpenAI-compatible)
            provider_request = {
                "custom_id": request.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }

            provider_requests.append(provider_request)

        return provider_requests

    def _convert_message_to_vllm(self, message: UnifiedMessage) -> dict[str, Any]:
        """Convert unified message to vLLM (OpenAI-compatible) format (updated for audio)."""
        # Handle text-only messages (most common case)
        if len(message.content) == 1 and isinstance(message.content[0], TextContent):
            return {
                "role": message.role,
                "content": message.content[0].text
            }

        # Handle multimodal messages (if model supports it)
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
                # vLLM/OpenAI doesn't support documents in chat completions
                # Skip for now
                pass

        return {
            "role": message.role,
            "content": content_parts
        }

    def _convert_image_to_vllm(self, image: ImageContent) -> dict[str, Any]:
        """Convert unified image content to vLLM (OpenAI-compatible) format."""
        if image.source_type == "url":
            return {
                "type": "image_url",
                "image_url": {
                    "url": image.data
                }
            }
        elif image.source_type == "base64":
            # vLLM expects OpenAI format: data:image/jpeg;base64,<base64_string>
            data_url = f"data:{image.media_type};base64,{image.data}"
            return {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        else:
            # file_uri - treat as URL
            return {
                "type": "image_url",
                "image_url": {
                    "url": image.data
                }
            }
    
    def _convert_audio_to_vllm(self, audio: AudioContent) -> dict[str, Any]:
        """
        Convert unified audio content to vLLM format.
        
        vLLM uses OpenAI-compatible format with audio support.
        The format is identical to OpenAI's audio input format.
        
        Format:
        {
            "type": "input_audio",
            "input_audio": {
                "data": "<base64_string>",
                "format": "wav"  # or "mp3"
            }
        }
        
        Note: Like OpenAI, vLLM uses simple format names ("wav", "mp3"),
        not full MIME types.
        """
        if audio.source_type != "base64":
            raise ValueError(
                "vLLM only supports base64-encoded audio in batch processing. "
                f"Got source_type={audio.source_type}. "
                "Convert URL or file_uri audio to base64 first."
            )
        
        # Extract and normalize format from media_type
        if audio.media_type in ("audio/wav", "audio/wave"):
            audio_format = "wav"
        elif audio.media_type in ("audio/mp3", "audio/mpeg"):
            audio_format = "mp3"
        else:
            # Should never happen due to AudioContent validation
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

        vLLM result format (OpenAI-compatible):
        {
            "id": "vllm-abc123",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "request_id": "vllm-batch-xyz",
                "body": {
                    "id": "cmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "model-name",
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
        Send batch to vLLM for local processing with audio support.

        Steps:
        1. Validate modalities
        2. Convert requests to vLLM (OpenAI-compatible) format
        3. Generate batch ID
        4. Save unified format JSONL
        5. Save provider format JSONL (input file)
        6. Start vLLM batch process in background
        7. Return batch ID

        Note: Unlike cloud providers, vLLM runs synchronously in a subprocess.
        The batch processing happens in the background.
        """
        # Validate modalities FIRST
        self.validate_request_modalities(batch.requests)
        
        # Generate batch ID
        batch_id = self._generate_batch_id()

        # Extract custom naming parameters
        custom_name = batch.name
        model = batch.requests[0].model if batch.requests else None

        # Save metadata for later use in get_results
        self._save_batch_metadata(batch_id, custom_name, model)

        # Convert to provider format
        provider_requests = self._convert_to_provider_format(batch.requests)

        # Get file paths
        unified_path = self.get_batch_file_path(batch_id, "unified", custom_name, model)
        provider_path = self.get_batch_file_path(batch_id, "provider", custom_name, model)
        output_path = self.get_batch_file_path(batch_id, "output", custom_name, model)

        # Save unified format
        unified_data = [req.to_dict() for req in batch.requests]
        await self._write_jsonl(str(unified_path), unified_data)

        # Save provider format (input file for vLLM)
        await self._write_jsonl(str(provider_path), provider_requests)

        # Extract model from first request (vLLM requires all requests use same model)
        if not batch.requests:
            raise ValueError("No requests in batch")

        model = batch.requests[0].model

        # Validate all requests use the same model
        for req in batch.requests:
            if req.model != model:
                raise ValueError(
                    "vLLM batch processing requires all requests to use the same model. "
                    f"Found models: {model} and {req.model}"
                )

        # Store batch metadata
        self.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "in_progress",
            "model": model,
            "input_file": str(provider_path),
            "output_file": str(output_path),
            "created_at": datetime.now().isoformat(),
            "total_requests": len(batch.requests),
            "completed_at": None,
            "process_info": None,
            "metadata": batch.metadata
        }
        self._save_batch_metadata()

        # Start vLLM batch process in background
        asyncio.create_task(self._run_vllm_batch(batch_id, str(provider_path), str(output_path), model))

        return batch_id

    async def _run_vllm_batch(
        self,
        batch_id: str,
        input_file: str,
        output_file: str,
        model: str
    ) -> None:
        """
        Run vLLM batch processing in background.

        This executes the vllm run-batch command asynchronously.
        """
        try:
            # Build vLLM command
            cmd = [
                self.vllm_command,
                "run-batch",
                "-i", input_file,
                "-o", output_file,
                "--model", model
            ]

            # Add any additional arguments
            cmd.extend(self.additional_args)

            # Run the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Update process info
            if batch_id in self.processes:
                self.processes[batch_id]["process_info"] = {
                    "pid": process.pid,
                    "command": " ".join(cmd)
                }
                self._save_batch_metadata()

            # Wait for process to complete
            stdout, stderr = await process.communicate()

            # Update batch status
            if batch_id in self.processes:
                if process.returncode == 0:
                    self.processes[batch_id]["status"] = "completed"
                    self.processes[batch_id]["completed_at"] = datetime.now().isoformat()
                else:
                    self.processes[batch_id]["status"] = "failed"
                    self.processes[batch_id]["error"] = {
                        "code": f"exit_{process.returncode}",
                        "message": stderr.decode() if stderr else "Process failed"
                    }

                self.processes[batch_id]["process_info"] = None
                self._save_batch_metadata()

        except Exception as e:
            # Update batch status on error
            if batch_id in self.processes:
                self.processes[batch_id]["status"] = "failed"
                self.processes[batch_id]["error"] = {
                    "code": "exception",
                    "message": str(e)
                }
                self.processes[batch_id]["process_info"] = None
                self._save_batch_metadata()

    async def get_status(
        self,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a batch.

        For vLLM, status is determined by:
        1. Check batch metadata
        2. Check if output file exists
        3. Parse output file to get request counts if available

        vLLM doesn't have API-level status like cloud providers,
        so we track status based on file existence and process state.
        """
        # Check if batch exists
        if batch_id not in self.processes:
            raise ValueError(f"Batch {batch_id} not found")

        batch_info = self.processes[batch_id]

        # Determine status
        output_file = Path(batch_info["output_file"])

        if batch_info["status"] == "failed":
            status = BatchStatus.FAILED
        elif batch_info["status"] == "completed" or output_file.exists():
            status = BatchStatus.COMPLETED
            # Update status if it was still marked as in_progress
            if batch_info["status"] == "in_progress":
                batch_info["status"] = "completed"
                batch_info["completed_at"] = datetime.now().isoformat()
                self._save_batch_metadata()
        else:
            status = BatchStatus.IN_PROGRESS

        # Count results if output file exists
        if output_file.exists():
            try:
                results = await self._read_jsonl(str(output_file))
                succeeded = sum(1 for r in results if not r.get("error"))
                errored = sum(1 for r in results if r.get("error"))
                processing = batch_info["total_requests"] - succeeded - errored
            except:
                # If we can't read the file yet, assume still processing
                succeeded = 0
                errored = 0
                processing = batch_info["total_requests"]
        else:
            succeeded = 0
            errored = 0
            processing = batch_info["total_requests"]

        request_counts = RequestCounts(
            total=batch_info["total_requests"],
            processing=processing,
            succeeded=succeeded,
            errored=errored,
            cancelled=0,
            expired=0
        )

        return BatchStatusResponse(
            batch_id=batch_id,
            provider="vllm",
            status=status,
            request_counts=request_counts,
            created_at=batch_info["created_at"],
            completed_at=batch_info.get("completed_at"),
            expires_at=None,  # vLLM batches don't expire
            provider_data={
                "model": batch_info["model"],
                "input_file": batch_info["input_file"],
                "output_file": batch_info["output_file"],
                "process_info": batch_info.get("process_info"),
                "raw_status": batch_info["status"]
            }
        )

    async def get_results(
        self,
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed batch.

        Steps:
        1. Check batch exists and is complete
        2. Read output file
        3. Convert to unified format
        4. Save unified results
        5. Yield each result
        """
        # Check if batch exists
        if batch_id not in self.processes:
            raise ValueError(f"Batch {batch_id} not found")

        batch_info = self.processes[batch_id]
        output_file = Path(batch_info["output_file"])

        # Check if output file exists
        if not output_file.exists():
            raise ValueError(
                f"Batch {batch_id} is not complete. "
                f"Output file does not exist yet."
            )

        # Read output file
        provider_results = await self._read_jsonl(str(output_file))

        # Load batch metadata for consistent file naming
        custom_name, model = self._load_batch_metadata(batch_id)

        # Convert to unified format
        unified_results = self._convert_from_provider_format(provider_results)

        # Save unified results
        results_path = self.get_batch_file_path(batch_id, "results", custom_name, model)
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

        For vLLM, this attempts to kill the running process.

        Returns:
            True if cancelled, False if already complete
        """
        # Check if batch exists
        if batch_id not in self.processes:
            raise ValueError(f"Batch {batch_id} not found")

        batch_info = self.processes[batch_id]

        # If already complete, can't cancel
        if batch_info["status"] in ["completed", "failed", "cancelled"]:
            return False

        # Try to kill the process if it's running
        process_info = batch_info.get("process_info")
        if process_info and "pid" in process_info:
            try:
                import signal
                os.kill(process_info["pid"], signal.SIGTERM)

                # Update status
                batch_info["status"] = "cancelled"
                batch_info["completed_at"] = datetime.now().isoformat()
                batch_info["process_info"] = None
                self._save_batch_metadata()

                return True
            except (OSError, ProcessLookupError):
                # Process already finished
                pass

        # Mark as cancelled anyway
        batch_info["status"] = "cancelled"
        batch_info["completed_at"] = datetime.now().isoformat()
        batch_info["process_info"] = None
        self._save_batch_metadata()

        return True

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
