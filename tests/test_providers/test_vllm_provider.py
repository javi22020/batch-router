"""Tests for vLLM provider implementation."""

import pytest
import json
import asyncio
import subprocess
from unittest.mock import AsyncMock, Mock, patch, MagicMock, mock_open
from datetime import datetime
from pathlib import Path

from batch_router.providers.vllm_provider import VLLMProvider
from batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from batch_router.core.messages import UnifiedMessage
from batch_router.core.content import TextContent, ImageContent
from batch_router.core.config import GenerationConfig
from batch_router.core.enums import BatchStatus, ResultStatus


class TestVLLMProviderConfiguration:
    """Test provider configuration and initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="")
            provider = VLLMProvider()
            assert provider.name == "vllm"
            assert provider.api_key is None
            assert provider.vllm_command == "vllm"
            assert provider.additional_args == []

    def test_init_custom_command(self):
        """Test initialization with custom vLLM command."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="")
            provider = VLLMProvider(vllm_command="/usr/local/bin/vllm")
            assert provider.vllm_command == "/usr/local/bin/vllm"

    def test_init_additional_args(self):
        """Test initialization with additional CLI arguments."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="")
            provider = VLLMProvider(additional_args=["--dtype", "float16"])
            assert provider.additional_args == ["--dtype", "float16"]

    def test_validate_configuration_vllm_not_found(self):
        """Test validation fails when vLLM is not installed."""
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            with pytest.raises(ValueError, match="vLLM command 'vllm' not found in PATH"):
                VLLMProvider()

    def test_validate_configuration_vllm_error(self):
        """Test validation fails when vLLM command returns error."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="vLLM error")
            with pytest.raises(ValueError, match="is not working properly"):
                VLLMProvider()

    def test_validate_configuration_timeout(self):
        """Test validation fails when vLLM command times out."""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("vllm", 5)):
            with pytest.raises(ValueError, match="timed out"):
                VLLMProvider()


class TestVLLMProviderFormatConversion:
    """Test conversion between unified and vLLM formats."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="")
            return VLLMProvider()

    def test_convert_simple_text_request(self, provider):
        """Test conversion of simple text-only request."""
        request = UnifiedRequest(
            custom_id="test-1",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                UnifiedMessage.from_text("user", "Hello, world!")
            ]
        )

        result = provider._convert_to_provider_format([request])

        assert len(result) == 1
        assert result[0]["custom_id"] == "test-1"
        assert result[0]["method"] == "POST"
        assert result[0]["url"] == "/v1/chat/completions"
        assert result[0]["body"]["model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert len(result[0]["body"]["messages"]) == 1
        assert result[0]["body"]["messages"][0]["role"] == "user"
        assert result[0]["body"]["messages"][0]["content"] == "Hello, world!"

    def test_convert_request_with_system_prompt(self, provider):
        """Test system prompt is converted to system message (OpenAI-compatible)."""
        request = UnifiedRequest(
            custom_id="test-1",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                UnifiedMessage.from_text("user", "Hello!")
            ],
            system_prompt="You are a helpful assistant."
        )

        result = provider._convert_to_provider_format([request])

        messages = result[0]["body"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    def test_convert_request_with_system_prompt_list(self, provider):
        """Test system prompt list is joined."""
        request = UnifiedRequest(
            custom_id="test-1",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                UnifiedMessage.from_text("user", "Hello!")
            ],
            system_prompt=["You are helpful.", "You are concise."]
        )

        result = provider._convert_to_provider_format([request])

        messages = result[0]["body"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful.\nYou are concise."

    def test_convert_request_with_generation_config(self, provider):
        """Test generation config parameters are included."""
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            stop_sequences=["END"]
        )

        request = UnifiedRequest(
            custom_id="test-1",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[UnifiedMessage.from_text("user", "Hello!")],
            generation_config=config
        )

        result = provider._convert_to_provider_format([request])

        body = result[0]["body"]
        # vLLM uses max_completion_tokens (newer OpenAI format)
        assert body["max_completion_tokens"] == 100
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.9
        assert body["top_k"] == 50
        assert body["stop"] == ["END"]
        # vLLM doesn't support presence_penalty or frequency_penalty
        assert "presence_penalty" not in body
        assert "frequency_penalty" not in body

    def test_convert_multimodal_message_text_and_image_url(self, provider):
        """Test multimodal message with text and image URL."""
        message = UnifiedMessage(
            role="user",
            content=[
                TextContent(text="What's in this image?"),
                ImageContent(
                    source_type="url",
                    media_type="image/jpeg",
                    data="https://example.com/image.jpg"
                )
            ]
        )

        request = UnifiedRequest(
            custom_id="test-1",
            model="llava-v1.5-7b",
            messages=[message]
        )

        result = provider._convert_to_provider_format([request])

        content = result[0]["body"]["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What's in this image?"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/image.jpg"

    def test_convert_multimodal_message_base64_image(self, provider):
        """Test multimodal message with base64 image."""
        message = UnifiedMessage(
            role="user",
            content=[
                TextContent(text="Analyze this"),
                ImageContent(
                    source_type="base64",
                    media_type="image/png",
                    data="iVBORw0KGgoAAAANSUhEUgA..."
                )
            ]
        )

        request = UnifiedRequest(
            custom_id="test-1",
            model="llava-v1.5-7b",
            messages=[message]
        )

        result = provider._convert_to_provider_format([request])

        content = result[0]["body"]["messages"][0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_convert_conversation(self, provider):
        """Test multi-turn conversation."""
        request = UnifiedRequest(
            custom_id="test-1",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                UnifiedMessage.from_text("user", "What is 2+2?"),
                UnifiedMessage.from_text("assistant", "2+2 equals 4."),
                UnifiedMessage.from_text("user", "What about 3+3?")
            ]
        )

        result = provider._convert_to_provider_format([request])

        messages = result[0]["body"]["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_convert_multiple_requests(self, provider):
        """Test conversion of multiple requests in batch."""
        requests = [
            UnifiedRequest(
                custom_id="req-1",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=[UnifiedMessage.from_text("user", "Hello 1")]
            ),
            UnifiedRequest(
                custom_id="req-2",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=[UnifiedMessage.from_text("user", "Hello 2")]
            )
        ]

        result = provider._convert_to_provider_format(requests)

        assert len(result) == 2
        assert result[0]["custom_id"] == "req-1"
        assert result[1]["custom_id"] == "req-2"


class TestVLLMProviderResultConversion:
    """Test conversion of vLLM results to unified format."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="")
            return VLLMProvider()

    def test_convert_successful_result(self, provider):
        """Test conversion of successful response."""
        vllm_result = {
            "id": "vllm-abc123",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "request_id": "vllm-batch-xyz",
                "body": {
                    "id": "cmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! How can I help?"
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    }
                }
            },
            "error": None
        }

        result = provider._convert_from_provider_format([vllm_result])

        assert len(result) == 1
        assert result[0].custom_id == "request-1"
        assert result[0].status == ResultStatus.SUCCEEDED
        assert result[0].response is not None
        assert result[0].response["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert result[0].error is None

    def test_convert_error_result(self, provider):
        """Test conversion of error response."""
        vllm_result = {
            "id": "vllm-abc123",
            "custom_id": "request-1",
            "response": None,
            "error": {
                "code": "invalid_request",
                "message": "Invalid model specified"
            }
        }

        result = provider._convert_from_provider_format([vllm_result])

        assert len(result) == 1
        assert result[0].custom_id == "request-1"
        assert result[0].status == ResultStatus.ERRORED
        assert result[0].response is None
        assert result[0].error["code"] == "invalid_request"
        assert result[0].error["message"] == "Invalid model specified"

    def test_convert_http_error_result(self, provider):
        """Test conversion of non-200 HTTP response."""
        vllm_result = {
            "id": "vllm-abc123",
            "custom_id": "request-1",
            "response": {
                "status_code": 500,
                "request_id": "vllm-batch-xyz",
                "body": {}
            },
            "error": None
        }

        result = provider._convert_from_provider_format([vllm_result])

        assert len(result) == 1
        assert result[0].status == ResultStatus.ERRORED
        assert result[0].error["code"] == "http_500"

    def test_get_text_response(self, provider):
        """Test extracting text from unified result."""
        vllm_result = {
            "id": "vllm-abc123",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Test response"
                            }
                        }
                    ]
                }
            },
            "error": None
        }

        result = provider._convert_from_provider_format([vllm_result])
        text = result[0].get_text_response()

        assert text == "Test response"


class TestVLLMProviderBatchOperations:
    """Test batch operations (mocked subprocess calls)."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="")
            return VLLMProvider()

    @pytest.mark.asyncio
    async def test_send_batch(self, provider):
        """Test sending a batch."""
        # Mock file operations
        provider._write_jsonl = AsyncMock()
        provider._save_batch_metadata = Mock()

        # Create batch metadata
        batch = UnifiedBatchMetadata(
            provider="vllm",
            requests=[
                UnifiedRequest(
                    custom_id="test-1",
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=[UnifiedMessage.from_text("user", "Hello!")]
                ),
                UnifiedRequest(
                    custom_id="test-2",
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=[UnifiedMessage.from_text("user", "Hi!")]
                )
            ],
            metadata={"test": "metadata"}
        )

        # Mock the background task
        with patch.object(asyncio, 'create_task') as mock_task:
            # Send batch
            batch_id = await provider.send_batch(batch)

            assert batch_id.startswith("vllm_")
            assert batch_id in provider.processes
            assert provider.processes[batch_id]["model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
            assert provider.processes[batch_id]["total_requests"] == 2
            assert provider.processes[batch_id]["status"] == "in_progress"

            # Verify background task was created
            mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_batch_different_models_raises_error(self, provider):
        """Test sending batch with different models raises error."""
        provider._write_jsonl = AsyncMock()

        batch = UnifiedBatchMetadata(
            provider="vllm",
            requests=[
                UnifiedRequest(
                    custom_id="test-1",
                    model="model-a",
                    messages=[UnifiedMessage.from_text("user", "Hello!")]
                ),
                UnifiedRequest(
                    custom_id="test-2",
                    model="model-b",
                    messages=[UnifiedMessage.from_text("user", "Hi!")]
                )
            ]
        )

        with patch.object(asyncio, 'create_task'):
            with pytest.raises(ValueError, match="requires all requests to use the same model"):
                await provider.send_batch(batch)

    @pytest.mark.asyncio
    async def test_run_vllm_batch_success(self, provider):
        """Test running vLLM batch process successfully."""
        batch_id = "vllm_test_123"
        input_file = "/tmp/input.jsonl"
        output_file = "/tmp/output.jsonl"
        model = "meta-llama/Meta-Llama-3-8B-Instruct"

        # Setup batch metadata
        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "in_progress",
            "model": model,
            "input_file": input_file,
            "output_file": output_file,
            "created_at": datetime.now().isoformat(),
            "total_requests": 1,
            "completed_at": None,
            "process_info": None
        }

        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await provider._run_vllm_batch(batch_id, input_file, output_file, model)

            # Check status was updated
            assert provider.processes[batch_id]["status"] == "completed"
            assert provider.processes[batch_id]["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_run_vllm_batch_failure(self, provider):
        """Test running vLLM batch process with failure."""
        batch_id = "vllm_test_123"
        input_file = "/tmp/input.jsonl"
        output_file = "/tmp/output.jsonl"
        model = "meta-llama/Meta-Llama-3-8B-Instruct"

        # Setup batch metadata
        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "in_progress",
            "model": model,
            "input_file": input_file,
            "output_file": output_file,
            "created_at": datetime.now().isoformat(),
            "total_requests": 1,
            "completed_at": None,
            "process_info": None
        }

        # Mock the subprocess with failure
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error occurred"))

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await provider._run_vllm_batch(batch_id, input_file, output_file, model)

            # Check status was updated to failed
            assert provider.processes[batch_id]["status"] == "failed"
            assert "error" in provider.processes[batch_id]

    @pytest.mark.asyncio
    async def test_get_status_in_progress(self, provider):
        """Test getting status of in-progress batch."""
        batch_id = "vllm_test_123"

        # Setup batch metadata
        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "in_progress",
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "input_file": "/tmp/input.jsonl",
            "output_file": "/tmp/output.jsonl",
            "created_at": "2024-01-01T00:00:00",
            "total_requests": 10,
            "completed_at": None,
            "process_info": {"pid": 12345}
        }

        # Mock output file doesn't exist
        with patch('pathlib.Path.exists', return_value=False):
            status = await provider.get_status(batch_id)

            assert status.batch_id == batch_id
            assert status.provider == "vllm"
            assert status.status == BatchStatus.IN_PROGRESS
            assert status.request_counts.total == 10
            assert status.request_counts.processing == 10
            assert status.request_counts.succeeded == 0

    @pytest.mark.asyncio
    async def test_get_status_completed(self, provider):
        """Test getting status of completed batch."""
        batch_id = "vllm_test_123"

        # Setup batch metadata
        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "completed",
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "input_file": "/tmp/input.jsonl",
            "output_file": "/tmp/output.jsonl",
            "created_at": "2024-01-01T00:00:00",
            "total_requests": 2,
            "completed_at": "2024-01-01T00:05:00",
            "process_info": None
        }

        # Mock output file exists and has results
        output_data = [
            {"custom_id": "req-1", "response": {"status_code": 200}, "error": None},
            {"custom_id": "req-2", "response": {"status_code": 200}, "error": None}
        ]

        provider._read_jsonl = AsyncMock(return_value=output_data)

        with patch('pathlib.Path.exists', return_value=True):
            status = await provider.get_status(batch_id)

            assert status.status == BatchStatus.COMPLETED
            assert status.request_counts.total == 2
            assert status.request_counts.succeeded == 2
            assert status.request_counts.processing == 0
            assert status.completed_at == "2024-01-01T00:05:00"

    @pytest.mark.asyncio
    async def test_get_status_failed(self, provider):
        """Test getting status of failed batch."""
        batch_id = "vllm_test_123"

        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "failed",
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "input_file": "/tmp/input.jsonl",
            "output_file": "/tmp/output.jsonl",
            "created_at": "2024-01-01T00:00:00",
            "total_requests": 5,
            "completed_at": None,
            "process_info": None,
            "error": {"code": "exit_1", "message": "Process failed"}
        }

        with patch('pathlib.Path.exists', return_value=False):
            status = await provider.get_status(batch_id)

            assert status.status == BatchStatus.FAILED
            assert status.request_counts.total == 5

    @pytest.mark.asyncio
    async def test_get_status_batch_not_found(self, provider):
        """Test getting status of non-existent batch."""
        with pytest.raises(ValueError, match="Batch .* not found"):
            await provider.get_status("non_existent_batch")

    @pytest.mark.asyncio
    async def test_get_results(self, provider):
        """Test getting results from completed batch."""
        batch_id = "vllm_test_123"

        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "completed",
            "output_file": "/tmp/output.jsonl"
        }

        # Mock output data
        output_data = [
            {
                "id": "vllm-1",
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "Response 1"
                            }
                        }]
                    }
                },
                "error": None
            },
            {
                "id": "vllm-2",
                "custom_id": "req-2",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "Response 2"
                            }
                        }]
                    }
                },
                "error": None
            }
        ]

        provider._read_jsonl = AsyncMock(return_value=output_data)
        provider._write_jsonl = AsyncMock()

        with patch('pathlib.Path.exists', return_value=True):
            results = []
            async for result in provider.get_results(batch_id):
                results.append(result)

            assert len(results) == 2
            assert results[0].custom_id == "req-1"
            assert results[0].status == ResultStatus.SUCCEEDED
            assert results[1].custom_id == "req-2"
            assert results[1].status == ResultStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_get_results_batch_not_found(self, provider):
        """Test getting results from non-existent batch."""
        with pytest.raises(ValueError, match="Batch .* not found"):
            async for result in provider.get_results("non_existent_batch"):
                pass

    @pytest.mark.asyncio
    async def test_get_results_not_complete(self, provider):
        """Test getting results from incomplete batch raises error."""
        batch_id = "vllm_test_123"

        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "in_progress",
            "output_file": "/tmp/output.jsonl"
        }

        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(ValueError, match="is not complete"):
                async for result in provider.get_results(batch_id):
                    pass

    @pytest.mark.asyncio
    async def test_cancel_batch(self, provider):
        """Test cancelling a running batch."""
        batch_id = "vllm_test_123"

        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "in_progress",
            "process_info": {"pid": 12345}
        }

        with patch('os.kill') as mock_kill:
            result = await provider.cancel_batch(batch_id)

            assert result is True
            assert provider.processes[batch_id]["status"] == "cancelled"
            assert provider.processes[batch_id]["completed_at"] is not None
            mock_kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_batch_already_complete(self, provider):
        """Test cancelling an already completed batch."""
        batch_id = "vllm_test_123"

        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "completed",
            "process_info": None
        }

        result = await provider.cancel_batch(batch_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_batch_process_not_found(self, provider):
        """Test cancelling batch when process doesn't exist."""
        batch_id = "vllm_test_123"

        provider.processes[batch_id] = {
            "batch_id": batch_id,
            "status": "in_progress",
            "process_info": {"pid": 99999}
        }

        with patch('os.kill', side_effect=ProcessLookupError()):
            result = await provider.cancel_batch(batch_id)

            assert result is True
            assert provider.processes[batch_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_batch_not_found(self, provider):
        """Test cancelling non-existent batch."""
        with pytest.raises(ValueError, match="Batch .* not found"):
            await provider.cancel_batch("non_existent_batch")


class TestVLLMProviderEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance for testing."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr="")
            return VLLMProvider()

    def test_empty_system_prompt(self, provider):
        """Test request with None system prompt."""
        request = UnifiedRequest(
            custom_id="test-1",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[UnifiedMessage.from_text("user", "Hello!")],
            system_prompt=None
        )

        result = provider._convert_to_provider_format([request])

        # Should only have user message, no system message
        messages = result[0]["body"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_provider_kwargs(self, provider):
        """Test provider-specific kwargs are passed through."""
        request = UnifiedRequest(
            custom_id="test-1",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[UnifiedMessage.from_text("user", "Hello!")],
            provider_kwargs={"n": 3, "best_of": 5}
        )

        result = provider._convert_to_provider_format([request])

        body = result[0]["body"]
        assert body["n"] == 3
        assert body["best_of"] == 5

    def test_partial_generation_config(self, provider):
        """Test generation config with only some parameters."""
        config = GenerationConfig(
            temperature=0.5,
            max_tokens=50
            # Other parameters None
        )

        request = UnifiedRequest(
            custom_id="test-1",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[UnifiedMessage.from_text("user", "Hello!")],
            generation_config=config
        )

        result = provider._convert_to_provider_format([request])

        body = result[0]["body"]
        assert body["temperature"] == 0.5
        assert body["max_completion_tokens"] == 50
        assert "top_p" not in body
        assert "top_k" not in body

    def test_metadata_file_path(self, provider):
        """Test metadata file path generation."""
        path = provider._get_metadata_file_path()
        assert str(path).endswith(".batch_router/generated/vllm/batch_metadata.json")

    def test_batch_id_generation(self, provider):
        """Test batch ID generation is unique."""
        id1 = provider._generate_batch_id()
        id2 = provider._generate_batch_id()

        assert id1.startswith("vllm_")
        assert id2.startswith("vllm_")
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_load_batch_metadata(self, provider):
        """Test loading batch metadata from file."""
        test_metadata = {
            "batch_123": {
                "batch_id": "batch_123",
                "status": "completed",
                "model": "test-model"
            }
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(test_metadata))):
                provider._load_batch_metadata()
                assert "batch_123" in provider.processes
                assert provider.processes["batch_123"]["status"] == "completed"

    def test_save_batch_metadata(self, provider):
        """Test saving batch metadata to file."""
        provider.processes = {
            "batch_123": {
                "batch_id": "batch_123",
                "status": "in_progress"
            }
        }

        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                provider._save_batch_metadata()
                mock_file.assert_called_once()
