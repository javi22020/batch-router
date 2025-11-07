"""Tests for Mistral provider implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import json
import io

from src.batch_router.providers.mistral_provider import MistralProvider
from src.batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from src.batch_router.core.messages import UnifiedMessage
from src.batch_router.core.content import TextContent, ImageContent
from src.batch_router.core.config import GenerationConfig
from src.batch_router.core.enums import BatchStatus, ResultStatus
from src.batch_router.core.responses import RequestCounts


class TestMistralProviderInit:
    """Test provider initialization and configuration."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = MistralProvider(api_key="test-key-123")
        assert provider.name == "mistral"
        assert provider.api_key == "test-key-123"
        assert provider.client is not None

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("MISTRAL_API_KEY", "env-key-456")
        provider = MistralProvider()
        assert provider.api_key == "env-key-456"

    def test_init_without_api_key(self, monkeypatch):
        """Test that initialization fails without API key."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Mistral API key is required"):
            MistralProvider()

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        provider = MistralProvider(
            api_key="test-key",
            server_url="https://custom.api.com",
            endpoint="/v1/chat/completions"
        )
        assert provider.config["server_url"] == "https://custom.api.com"
        assert provider.endpoint == "/v1/chat/completions"

    def test_init_default_endpoint(self):
        """Test that default endpoint is set correctly."""
        provider = MistralProvider(api_key="test-key")
        assert provider.endpoint == "/v1/chat/completions"


class TestMistralProviderFormatConversion:
    """Test format conversion methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for testing."""
        return MistralProvider(api_key="test-key")

    def test_convert_text_content(self, provider):
        """Test conversion of text-only content."""
        content = [TextContent(text="Hello, world!")]
        result = provider._convert_content_to_mistral(content)

        assert len(result) == 1
        assert result[0] == {"type": "text", "text": "Hello, world!"}

    def test_convert_image_base64(self, provider):
        """Test conversion of base64 image content."""
        content = [ImageContent(
            source_type="base64",
            media_type="image/jpeg",
            data="base64encodeddata"
        )]
        result = provider._convert_content_to_mistral(content)

        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"] == "data:image/jpeg;base64,base64encodeddata"

    def test_convert_image_url(self, provider):
        """Test conversion of URL image content."""
        content = [ImageContent(
            source_type="url",
            data="https://example.com/image.jpg"
        )]
        result = provider._convert_content_to_mistral(content)

        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"] == "https://example.com/image.jpg"

    def test_convert_multimodal_content(self, provider):
        """Test conversion of mixed text and image content."""
        content = [
            TextContent(text="Look at this:"),
            ImageContent(source_type="url", data="https://example.com/img.jpg")
        ]
        result = provider._convert_content_to_mistral(content)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"

    def test_convert_to_provider_format_simple(self, provider):
        """Test conversion of simple text request."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="mistral-small-latest",
            messages=[UnifiedMessage.from_text("user", "Hello")]
        )

        result = provider._convert_to_provider_format([request])

        assert len(result) == 1
        assert result[0]["custom_id"] == "req-1"
        assert result[0]["body"]["model"] == "mistral-small-latest"
        assert len(result[0]["body"]["messages"]) == 1
        assert result[0]["body"]["messages"][0]["role"] == "user"
        assert result[0]["body"]["messages"][0]["content"] == "Hello"

    def test_convert_to_provider_format_with_system(self, provider):
        """Test conversion with system prompt."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="mistral-small-latest",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt="You are a helpful assistant"
        )

        result = provider._convert_to_provider_format([request])

        assert len(result[0]["body"]["messages"]) == 2
        assert result[0]["body"]["messages"][0]["role"] == "system"
        assert result[0]["body"]["messages"][0]["content"] == "You are a helpful assistant"
        assert result[0]["body"]["messages"][1]["role"] == "user"

    def test_convert_to_provider_format_with_system_list(self, provider):
        """Test conversion with system prompt as list."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="mistral-small-latest",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt=["You are helpful", "Be concise"]
        )

        result = provider._convert_to_provider_format([request])

        assert result[0]["body"]["messages"][0]["role"] == "system"
        assert result[0]["body"]["messages"][0]["content"] == "You are helpful\nBe concise"

    def test_convert_to_provider_format_with_generation_config(self, provider):
        """Test conversion with generation config."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="mistral-small-latest",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            generation_config=GenerationConfig(
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                stop_sequences=["END"]
            )
        )

        result = provider._convert_to_provider_format([request])
        body = result[0]["body"]

        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 2048
        assert body["top_p"] == 0.9
        assert body["stop"] == ["END"]

    def test_convert_to_provider_format_multiple_messages(self, provider):
        """Test conversion with multi-turn conversation."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="mistral-small-latest",
            messages=[
                UnifiedMessage.from_text("user", "Hello"),
                UnifiedMessage.from_text("assistant", "Hi there!"),
                UnifiedMessage.from_text("user", "How are you?")
            ]
        )

        result = provider._convert_to_provider_format([request])
        messages = result[0]["body"]["messages"]

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_convert_to_provider_format_multimodal(self, provider):
        """Test conversion with multimodal content."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="mistral-small-latest",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        TextContent(text="What's in this image?"),
                        ImageContent(source_type="url", data="https://example.com/img.jpg")
                    ]
                )
            ]
        )

        result = provider._convert_to_provider_format([request])
        message = result[0]["body"]["messages"][0]

        assert message["role"] == "user"
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 2

    def test_convert_from_provider_format_succeeded(self, provider):
        """Test conversion of succeeded result."""
        mistral_result = [{
            "custom_id": "req-1",
            "status": 200,
            "body": {
                "id": "chat_123",
                "object": "chat.completion",
                "model": "mistral-small-latest",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            }
        }]

        result = provider._convert_from_provider_format(mistral_result)

        assert len(result) == 1
        assert result[0].custom_id == "req-1"
        assert result[0].status == ResultStatus.SUCCEEDED
        assert result[0].response is not None
        assert result[0].error is None

    def test_convert_from_provider_format_errored(self, provider):
        """Test conversion of errored result."""
        mistral_result = [{
            "custom_id": "req-1",
            "status": 400,
            "body": {
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid model specified"
                }
            }
        }]

        result = provider._convert_from_provider_format(mistral_result)

        assert len(result) == 1
        assert result[0].custom_id == "req-1"
        assert result[0].status == ResultStatus.ERRORED
        assert result[0].response is None
        assert result[0].error is not None

    def test_convert_from_provider_format_server_error(self, provider):
        """Test conversion of server error result."""
        mistral_result = [{
            "custom_id": "req-1",
            "status": 500,
            "body": {}
        }]

        result = provider._convert_from_provider_format(mistral_result)

        assert result[0].status == ResultStatus.ERRORED
        assert result[0].error is not None


class TestMistralProviderBatchOperations:
    """Test batch operation methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance with mocked client."""
        provider = MistralProvider(api_key="test-key")
        provider.client = Mock()
        return provider

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        return UnifiedBatchMetadata(
            provider="mistral",
            requests=[
                UnifiedRequest(
                    custom_id="req-1",
                    model="mistral-small-latest",
                    messages=[UnifiedMessage.from_text("user", "Hello")],
                    system_prompt="You are helpful"
                ),
                UnifiedRequest(
                    custom_id="req-2",
                    model="mistral-small-latest",
                    messages=[UnifiedMessage.from_text("user", "Hi")]
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_send_batch_success(self, provider, sample_batch):
        """Test successful batch sending."""
        # Mock file upload
        mock_file = Mock()
        mock_file.id = "file_test123"
        provider.client.files.upload = Mock(return_value=mock_file)

        # Mock job creation
        mock_job = Mock()
        mock_job.id = "job_test123"
        provider.client.batch.jobs.create = Mock(return_value=mock_job)

        # Mock file operations
        with patch('builtins.open', mock_open()):
            with patch('aiofiles.open', new_callable=MagicMock):
                with patch.object(provider, '_write_jsonl', new=AsyncMock()):
                    with patch('pathlib.Path.exists', return_value=True):
                        with patch('pathlib.Path.unlink'):
                            batch_id = await provider.send_batch(sample_batch)

        assert batch_id == "job_test123"
        provider.client.files.upload.assert_called_once()
        provider.client.batch.jobs.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_batch_upload_error(self, provider, sample_batch):
        """Test batch sending with file upload error."""
        provider.client.files.upload = Mock(side_effect=Exception("Upload Error"))

        with patch('builtins.open', mock_open()):
            with patch('aiofiles.open', new_callable=MagicMock):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.unlink'):
                        with pytest.raises(Exception, match="Failed to create Mistral batch"):
                            await provider.send_batch(sample_batch)

    @pytest.mark.asyncio
    async def test_send_batch_job_creation_error(self, provider, sample_batch):
        """Test batch sending with job creation error."""
        mock_file = Mock()
        mock_file.id = "file_test123"
        provider.client.files.upload = Mock(return_value=mock_file)
        provider.client.batch.jobs.create = Mock(side_effect=Exception("Job Error"))

        with patch('builtins.open', mock_open()):
            with patch('aiofiles.open', new_callable=MagicMock):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.unlink'):
                        with pytest.raises(Exception, match="Failed to create Mistral batch"):
                            await provider.send_batch(sample_batch)

    @pytest.mark.asyncio
    async def test_get_status_queued(self, provider):
        """Test getting status of queued batch."""
        mock_job = Mock()
        mock_job.status = "QUEUED"
        mock_job.total_requests = 10
        mock_job.created_at = "2024-01-01T00:00:00Z"
        mock_job.completed_at = None
        mock_job.model = "mistral-small-latest"
        mock_job.endpoint = "/v1/chat/completions"
        mock_job.input_files = ["file_123"]
        mock_job.output_file = None
        mock_job.error_file = None

        provider.client.batch.jobs.get = Mock(return_value=mock_job)

        status = await provider.get_status("job_test123")

        assert status.batch_id == "job_test123"
        assert status.status == BatchStatus.VALIDATING
        assert status.request_counts.total == 10
        assert status.request_counts.processing == 10

    @pytest.mark.asyncio
    async def test_get_status_running(self, provider):
        """Test getting status of running batch."""
        mock_job = Mock()
        mock_job.status = "RUNNING"
        mock_job.total_requests = 10
        mock_job.created_at = "2024-01-01T00:00:00Z"
        mock_job.completed_at = None
        mock_job.model = "mistral-small-latest"
        mock_job.endpoint = "/v1/chat/completions"
        mock_job.input_files = ["file_123"]
        mock_job.output_file = None
        mock_job.error_file = None

        provider.client.batch.jobs.get = Mock(return_value=mock_job)

        status = await provider.get_status("job_test123")

        assert status.status == BatchStatus.IN_PROGRESS
        assert status.request_counts.processing == 10

    @pytest.mark.asyncio
    async def test_get_status_success(self, provider):
        """Test getting status of successful batch."""
        mock_job = Mock()
        mock_job.status = "SUCCESS"
        mock_job.total_requests = 10
        mock_job.created_at = "2024-01-01T00:00:00Z"
        mock_job.completed_at = "2024-01-01T01:00:00Z"
        mock_job.model = "mistral-small-latest"
        mock_job.endpoint = "/v1/chat/completions"
        mock_job.input_files = ["file_123"]
        mock_job.output_file = "file_456"
        mock_job.error_file = None

        provider.client.batch.jobs.get = Mock(return_value=mock_job)

        status = await provider.get_status("job_test123")

        assert status.status == BatchStatus.COMPLETED
        assert status.request_counts.succeeded == 10
        assert status.request_counts.processing == 0
        assert status.completed_at == "2024-01-01T01:00:00Z"

    @pytest.mark.asyncio
    async def test_get_status_failed(self, provider):
        """Test getting status of failed batch."""
        mock_job = Mock()
        mock_job.status = "FAILED"
        mock_job.total_requests = 10
        mock_job.created_at = "2024-01-01T00:00:00Z"
        mock_job.completed_at = "2024-01-01T00:30:00Z"
        mock_job.model = "mistral-small-latest"
        mock_job.endpoint = "/v1/chat/completions"
        mock_job.input_files = ["file_123"]
        mock_job.output_file = None
        mock_job.error_file = "file_error"

        provider.client.batch.jobs.get = Mock(return_value=mock_job)

        status = await provider.get_status("job_test123")

        assert status.status == BatchStatus.FAILED
        assert status.request_counts.errored == 10

    @pytest.mark.asyncio
    async def test_get_status_cancelled(self, provider):
        """Test getting status of cancelled batch."""
        mock_job = Mock()
        mock_job.status = "CANCELLED"
        mock_job.total_requests = 10
        mock_job.created_at = "2024-01-01T00:00:00Z"
        mock_job.completed_at = "2024-01-01T00:15:00Z"
        mock_job.model = "mistral-small-latest"
        mock_job.endpoint = "/v1/chat/completions"
        mock_job.input_files = ["file_123"]
        mock_job.output_file = None
        mock_job.error_file = None

        provider.client.batch.jobs.get = Mock(return_value=mock_job)

        status = await provider.get_status("job_test123")

        assert status.status == BatchStatus.CANCELLED
        assert status.request_counts.cancelled == 10

    @pytest.mark.asyncio
    async def test_get_results_not_complete(self, provider):
        """Test getting results from incomplete batch."""
        with patch.object(provider, 'get_status') as mock_status:
            mock_status.return_value = Mock(
                is_complete=Mock(return_value=False),
                status=BatchStatus.IN_PROGRESS
            )

            with pytest.raises(Exception, match="not complete yet"):
                async for _ in provider.get_results("job_test123"):
                    pass

    @pytest.mark.asyncio
    async def test_get_results_no_output_file(self, provider):
        """Test getting results when no output file available."""
        mock_status = Mock()
        mock_status.is_complete = Mock(return_value=True)
        mock_status.provider_data = {"output_file": None}

        with patch.object(provider, 'get_status', return_value=mock_status):
            with pytest.raises(Exception, match="No output file available"):
                async for _ in provider.get_results("job_test123"):
                    pass

    @pytest.mark.asyncio
    async def test_get_results_success(self, provider):
        """Test successful results retrieval."""
        # Mock status check
        mock_status = Mock()
        mock_status.is_complete = Mock(return_value=True)
        mock_status.provider_data = {"output_file": "file_456"}

        # Create mock file stream
        result_line = json.dumps({
            "custom_id": "req-1",
            "status": 200,
            "body": {
                "id": "chat_123",
                "object": "chat.completion",
                "model": "mistral-small-latest",
                "choices": [{
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop"
                }]
            }
        })
        mock_stream = Mock()
        mock_stream.read = Mock(return_value=result_line.encode('utf-8'))

        with patch.object(provider, 'get_status', return_value=mock_status):
            provider.client.files.download = Mock(return_value=mock_stream)

            with patch.object(provider, '_write_jsonl', new=AsyncMock()):
                results = []
                async for result in provider.get_results("job_test123"):
                    results.append(result)

                assert len(results) == 1
                assert results[0].custom_id == "req-1"
                assert results[0].status == ResultStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_cancel_batch_success(self, provider):
        """Test successful batch cancellation."""
        mock_status = Mock()
        mock_status.is_complete = Mock(return_value=False)

        with patch.object(provider, 'get_status', return_value=mock_status):
            provider.client.batch.jobs.cancel = Mock()

            result = await provider.cancel_batch("job_test123")

            assert result is True
            provider.client.batch.jobs.cancel.assert_called_once_with(job_id="job_test123")

    @pytest.mark.asyncio
    async def test_cancel_batch_already_complete(self, provider):
        """Test canceling already complete batch."""
        mock_status = Mock()
        mock_status.is_complete = Mock(return_value=True)

        with patch.object(provider, 'get_status', return_value=mock_status):
            result = await provider.cancel_batch("job_test123")

            assert result is False

    @pytest.mark.asyncio
    async def test_list_batches(self, provider):
        """Test listing recent batches."""
        mock_job1 = Mock()
        mock_job1.id = "job_1"
        mock_job1.status = "RUNNING"
        mock_job1.total_requests = 5
        mock_job1.created_at = "2024-01-01T00:00:00Z"
        mock_job1.completed_at = None
        mock_job1.model = "mistral-small-latest"
        mock_job1.endpoint = "/v1/chat/completions"

        mock_job2 = Mock()
        mock_job2.id = "job_2"
        mock_job2.status = "SUCCESS"
        mock_job2.total_requests = 10
        mock_job2.created_at = "2024-01-01T00:00:00Z"
        mock_job2.completed_at = "2024-01-01T01:00:00Z"
        mock_job2.model = "mistral-small-latest"
        mock_job2.endpoint = "/v1/chat/completions"

        mock_list = Mock()
        mock_list.data = [mock_job1, mock_job2]
        provider.client.batch.jobs.list = Mock(return_value=mock_list)

        batches = await provider.list_batches(limit=2)

        assert len(batches) == 2
        assert batches[0].batch_id == "job_1"
        assert batches[0].status == BatchStatus.IN_PROGRESS
        assert batches[1].batch_id == "job_2"
        assert batches[1].status == BatchStatus.COMPLETED


class TestMistralProviderFileOperations:
    """Test file I/O operations."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return MistralProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_write_and_read_jsonl(self, provider, tmp_path):
        """Test writing and reading JSONL files."""
        test_data = [
            {"id": "1", "value": "test1"},
            {"id": "2", "value": "test2"}
        ]

        file_path = tmp_path / "test.jsonl"
        await provider._write_jsonl(file_path, test_data)

        # Verify file was created
        assert file_path.exists()

        # Read back and verify
        read_data = await provider._read_jsonl(file_path)
        assert read_data == test_data

    @pytest.mark.asyncio
    async def test_write_jsonl_empty(self, provider, tmp_path):
        """Test writing empty JSONL file."""
        file_path = tmp_path / "empty.jsonl"
        await provider._write_jsonl(file_path, [])

        assert file_path.exists()
        read_data = await provider._read_jsonl(file_path)
        assert read_data == []

    def test_get_batch_file_path(self, provider):
        """Test batch file path generation."""
        path = provider.get_batch_file_path("batch123", "unified")

        assert "mistral" in str(path)
        assert "batch_batch123_unified.jsonl" in str(path)


class TestMistralProviderEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return MistralProvider(api_key="test-key")

    def test_convert_empty_content(self, provider):
        """Test conversion of empty content list."""
        result = provider._convert_content_to_mistral([])
        assert result == []

    def test_convert_to_provider_format_with_provider_kwargs(self, provider):
        """Test conversion with additional provider-specific kwargs."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="mistral-small-latest",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            provider_kwargs={
                "safe_prompt": True,
                "random_seed": 42
            }
        )

        result = provider._convert_to_provider_format([request])

        assert result[0]["body"]["safe_prompt"] is True
        assert result[0]["body"]["random_seed"] == 42

    @pytest.mark.asyncio
    async def test_get_status_api_error(self, provider):
        """Test status retrieval with API error."""
        provider.client.batch.jobs.get = Mock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception, match="Failed to retrieve batch status"):
            await provider.get_status("job_test123")

    @pytest.mark.asyncio
    async def test_cancel_batch_api_error(self, provider):
        """Test batch cancellation with API error."""
        mock_status = Mock()
        mock_status.is_complete = Mock(return_value=False)

        with patch.object(provider, 'get_status', return_value=mock_status):
            provider.client.batch.jobs.cancel = Mock(side_effect=Exception("Cancel Error"))

            with pytest.raises(Exception, match="Failed to cancel batch"):
                await provider.cancel_batch("job_test123")

    @pytest.mark.asyncio
    async def test_list_batches_api_error(self, provider):
        """Test listing batches with API error."""
        provider.client.batch.jobs.list = Mock(side_effect=Exception("List Error"))

        with pytest.raises(Exception, match="Failed to list batches"):
            await provider.list_batches()

    def test_status_mapping_timeout_exceeded(self, provider):
        """Test that TIMEOUT_EXCEEDED maps to FAILED."""
        mock_job = Mock()
        mock_job.status = "TIMEOUT_EXCEEDED"
        mock_job.total_requests = 5
        mock_job.created_at = "2024-01-01T00:00:00Z"
        mock_job.completed_at = None
        mock_job.model = "mistral-small-latest"
        mock_job.endpoint = "/v1/chat/completions"
        mock_job.input_files = ["file_123"]
        mock_job.output_file = None
        mock_job.error_file = None

        provider.client.batch.jobs.get = Mock(return_value=mock_job)

        import asyncio
        status = asyncio.run(provider.get_status("job_test123"))

        assert status.status == BatchStatus.FAILED

    def test_status_mapping_cancellation_requested(self, provider):
        """Test that CANCELLATION_REQUESTED maps to IN_PROGRESS."""
        mock_job = Mock()
        mock_job.status = "CANCELLATION_REQUESTED"
        mock_job.total_requests = 5
        mock_job.created_at = "2024-01-01T00:00:00Z"
        mock_job.completed_at = None
        mock_job.model = "mistral-small-latest"
        mock_job.endpoint = "/v1/chat/completions"
        mock_job.input_files = ["file_123"]
        mock_job.output_file = None
        mock_job.error_file = None

        provider.client.batch.jobs.get = Mock(return_value=mock_job)

        import asyncio
        status = asyncio.run(provider.get_status("job_test123"))

        assert status.status == BatchStatus.IN_PROGRESS
