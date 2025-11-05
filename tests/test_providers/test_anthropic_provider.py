"""Tests for Anthropic provider implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.batch_router.providers.anthropic_provider import AnthropicProvider
from src.batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from src.batch_router.core.messages import UnifiedMessage
from src.batch_router.core.content import TextContent, ImageContent
from src.batch_router.core.config import GenerationConfig
from src.batch_router.core.enums import BatchStatus, ResultStatus
from src.batch_router.core.responses import RequestCounts


class TestAnthropicProviderInit:
    """Test provider initialization and configuration."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = AnthropicProvider(api_key="test-key-123")
        assert provider.name == "anthropic"
        assert provider.api_key == "test-key-123"
        assert provider.client is not None

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-456")
        provider = AnthropicProvider()
        assert provider.api_key == "env-key-456"

    def test_init_without_api_key(self, monkeypatch):
        """Test that initialization fails without API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            AnthropicProvider()

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        provider = AnthropicProvider(
            api_key="test-key",
            base_url="https://custom.api.com",
            timeout=120
        )
        assert provider.config["base_url"] == "https://custom.api.com"
        assert provider.config["timeout"] == 120


class TestAnthropicProviderFormatConversion:
    """Test format conversion methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for testing."""
        return AnthropicProvider(api_key="test-key")

    def test_convert_text_content(self, provider):
        """Test conversion of text-only content."""
        content = [TextContent(text="Hello, world!")]
        result = provider._convert_content_to_anthropic(content)

        assert len(result) == 1
        assert result[0] == {"type": "text", "text": "Hello, world!"}

    def test_convert_image_base64(self, provider):
        """Test conversion of base64 image content."""
        content = [ImageContent(
            source_type="base64",
            media_type="image/jpeg",
            data="base64encodeddata"
        )]
        result = provider._convert_content_to_anthropic(content)

        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/jpeg"
        assert result[0]["source"]["data"] == "base64encodeddata"

    def test_convert_image_url(self, provider):
        """Test conversion of URL image content."""
        content = [ImageContent(
            source_type="url",
            data="https://example.com/image.jpg"
        )]
        result = provider._convert_content_to_anthropic(content)

        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == "https://example.com/image.jpg"

    def test_convert_multimodal_content(self, provider):
        """Test conversion of mixed text and image content."""
        content = [
            TextContent(text="Look at this:"),
            ImageContent(source_type="base64", data="imgdata")
        ]
        result = provider._convert_content_to_anthropic(content)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"

    def test_convert_to_provider_format_simple(self, provider):
        """Test conversion of simple text request."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="claude-sonnet-4-5",
            messages=[UnifiedMessage.from_text("user", "Hello")]
        )

        result = provider._convert_to_provider_format([request])

        assert len(result) == 1
        assert result[0]["custom_id"] == "req-1"
        assert result[0]["params"]["model"] == "claude-sonnet-4-5"
        assert result[0]["params"]["max_tokens"] == 1024  # Default
        assert len(result[0]["params"]["messages"]) == 1
        assert result[0]["params"]["messages"][0]["role"] == "user"

    def test_convert_to_provider_format_with_system(self, provider):
        """Test conversion with system prompt."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="claude-sonnet-4-5",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt="You are a helpful assistant"
        )

        result = provider._convert_to_provider_format([request])

        assert result[0]["params"]["system"] == "You are a helpful assistant"

    def test_convert_to_provider_format_with_generation_config(self, provider):
        """Test conversion with generation config."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="claude-sonnet-4-5",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            generation_config=GenerationConfig(
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                top_k=40,
                stop_sequences=["END"]
            )
        )

        result = provider._convert_to_provider_format([request])
        params = result[0]["params"]

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 2048
        assert params["top_p"] == 0.9
        assert params["top_k"] == 40
        assert params["stop_sequences"] == ["END"]

    def test_convert_to_provider_format_multiple_messages(self, provider):
        """Test conversion with multi-turn conversation."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="claude-sonnet-4-5",
            messages=[
                UnifiedMessage.from_text("user", "Hello"),
                UnifiedMessage.from_text("assistant", "Hi there!"),
                UnifiedMessage.from_text("user", "How are you?")
            ]
        )

        result = provider._convert_to_provider_format([request])
        messages = result[0]["params"]["messages"]

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_convert_from_provider_format_succeeded(self, provider):
        """Test conversion of succeeded result."""
        anthropic_result = [{
            "custom_id": "req-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                }
            }
        }]

        result = provider._convert_from_provider_format(anthropic_result)

        assert len(result) == 1
        assert result[0].custom_id == "req-1"
        assert result[0].status == ResultStatus.SUCCEEDED
        assert result[0].response is not None
        assert result[0].error is None

    def test_convert_from_provider_format_errored(self, provider):
        """Test conversion of errored result."""
        anthropic_result = [{
            "custom_id": "req-1",
            "result": {
                "type": "errored",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid model specified"
                }
            }
        }]

        result = provider._convert_from_provider_format(anthropic_result)

        assert len(result) == 1
        assert result[0].custom_id == "req-1"
        assert result[0].status == ResultStatus.ERRORED
        assert result[0].response is None
        assert result[0].error is not None

    def test_convert_from_provider_format_expired(self, provider):
        """Test conversion of expired result."""
        anthropic_result = [{
            "custom_id": "req-1",
            "result": {"type": "expired"}
        }]

        result = provider._convert_from_provider_format(anthropic_result)

        assert result[0].status == ResultStatus.EXPIRED

    def test_convert_from_provider_format_canceled(self, provider):
        """Test conversion of canceled result."""
        anthropic_result = [{
            "custom_id": "req-1",
            "result": {"type": "canceled"}
        }]

        result = provider._convert_from_provider_format(anthropic_result)

        assert result[0].status == ResultStatus.CANCELLED


class TestAnthropicProviderBatchOperations:
    """Test batch operation methods."""

    @pytest.fixture
    def provider(self):
        """Create provider instance with mocked client."""
        provider = AnthropicProvider(api_key="test-key")
        provider.client = Mock()
        return provider

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        return UnifiedBatchMetadata(
            provider="anthropic",
            requests=[
                UnifiedRequest(
                    custom_id="req-1",
                    model="claude-sonnet-4-5",
                    messages=[UnifiedMessage.from_text("user", "Hello")],
                    system_prompt="You are helpful"
                ),
                UnifiedRequest(
                    custom_id="req-2",
                    model="claude-sonnet-4-5",
                    messages=[UnifiedMessage.from_text("user", "Hi")]
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_send_batch_success(self, provider, sample_batch):
        """Test successful batch sending."""
        # Mock the API response
        mock_batch = Mock()
        mock_batch.id = "msgbatch_test123"
        provider.client.messages.batches.create = Mock(return_value=mock_batch)

        # Mock file operations
        with patch.object(provider, '_write_jsonl', new=AsyncMock()):
            batch_id = await provider.send_batch(sample_batch)

        assert batch_id == "msgbatch_test123"
        provider.client.messages.batches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_batch_api_error(self, provider, sample_batch):
        """Test batch sending with API error."""
        provider.client.messages.batches.create = Mock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception, match="Failed to create Anthropic batch"):
            await provider.send_batch(sample_batch)

    @pytest.mark.asyncio
    async def test_get_status_in_progress(self, provider):
        """Test getting status of in-progress batch."""
        mock_batch = Mock()
        mock_batch.id = "msgbatch_test123"
        mock_batch.processing_status = "in_progress"
        mock_batch.request_counts = Mock(
            processing=5,
            succeeded=0,
            errored=0,
            canceled=0,
            expired=0
        )
        mock_batch.created_at = "2024-01-01T00:00:00Z"
        mock_batch.ended_at = None
        mock_batch.expires_at = "2024-01-02T00:00:00Z"
        mock_batch.results_url = None

        provider.client.messages.batches.retrieve = Mock(return_value=mock_batch)

        status = await provider.get_status("msgbatch_test123")

        assert status.batch_id == "msgbatch_test123"
        assert status.status == BatchStatus.IN_PROGRESS
        assert status.request_counts.processing == 5
        assert status.request_counts.total == 5

    @pytest.mark.asyncio
    async def test_get_status_completed(self, provider):
        """Test getting status of completed batch."""
        mock_batch = Mock()
        mock_batch.id = "msgbatch_test123"
        mock_batch.processing_status = "ended"
        mock_batch.request_counts = Mock(
            processing=0,
            succeeded=5,
            errored=0,
            canceled=0,
            expired=0
        )
        mock_batch.created_at = "2024-01-01T00:00:00Z"
        mock_batch.ended_at = "2024-01-01T01:00:00Z"
        mock_batch.expires_at = "2024-01-02T00:00:00Z"
        mock_batch.results_url = "https://api.anthropic.com/results/..."

        provider.client.messages.batches.retrieve = Mock(return_value=mock_batch)

        status = await provider.get_status("msgbatch_test123")

        assert status.status == BatchStatus.COMPLETED
        assert status.request_counts.succeeded == 5
        assert status.completed_at == "2024-01-01T01:00:00Z"

    @pytest.mark.asyncio
    async def test_get_results_not_complete(self, provider):
        """Test getting results from incomplete batch."""
        # Mock status check
        with patch.object(provider, 'get_status') as mock_status:
            mock_status.return_value = Mock(
                is_complete=Mock(return_value=False),
                status=BatchStatus.IN_PROGRESS
            )

            with pytest.raises(Exception, match="not complete yet"):
                async for _ in provider.get_results("msgbatch_test123"):
                    pass

    @pytest.mark.asyncio
    async def test_get_results_success(self, provider):
        """Test successful results retrieval."""
        # Mock status check
        mock_status = Mock()
        mock_status.is_complete = Mock(return_value=True)
        mock_status.provider_data = {"results_url": "https://api.anthropic.com/results"}

        with patch.object(provider, 'get_status', return_value=mock_status):
            # Mock results iterator
            mock_result = Mock()
            mock_result.custom_id = "req-1"
            mock_result.result = Mock()
            mock_result.result.type = "succeeded"
            mock_result.result.message = Mock(
                id="msg_123",
                type="message",
                role="assistant",
                model="claude-sonnet-4-5",
                content=[Mock(type="text", text="Hello!")],
                stop_reason="end_turn",
                stop_sequence=None,
                usage=Mock(input_tokens=10, output_tokens=5)
            )

            provider.client.messages.batches.results = Mock(
                return_value=[mock_result]
            )

            # Mock file operations
            with patch.object(provider, '_write_jsonl', new=AsyncMock()):
                results = []
                async for result in provider.get_results("msgbatch_test123"):
                    results.append(result)

                assert len(results) == 1
                assert results[0].custom_id == "req-1"
                assert results[0].status == ResultStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_cancel_batch_success(self, provider):
        """Test successful batch cancellation."""
        # Mock status check (batch is in progress)
        mock_status = Mock()
        mock_status.is_complete = Mock(return_value=False)

        with patch.object(provider, 'get_status', return_value=mock_status):
            provider.client.messages.batches.cancel = Mock()

            result = await provider.cancel_batch("msgbatch_test123")

            assert result is True
            provider.client.messages.batches.cancel.assert_called_once_with("msgbatch_test123")

    @pytest.mark.asyncio
    async def test_cancel_batch_already_complete(self, provider):
        """Test canceling already complete batch."""
        mock_status = Mock()
        mock_status.is_complete = Mock(return_value=True)

        with patch.object(provider, 'get_status', return_value=mock_status):
            result = await provider.cancel_batch("msgbatch_test123")

            assert result is False

    @pytest.mark.asyncio
    async def test_list_batches(self, provider):
        """Test listing recent batches."""
        mock_batch1 = Mock()
        mock_batch1.id = "msgbatch_1"
        mock_batch1.processing_status = "in_progress"
        mock_batch1.request_counts = Mock(
            processing=5, succeeded=0, errored=0, canceled=0, expired=0
        )
        mock_batch1.created_at = "2024-01-01T00:00:00Z"
        mock_batch1.ended_at = None
        mock_batch1.expires_at = "2024-01-02T00:00:00Z"
        mock_batch1.results_url = None

        mock_batch2 = Mock()
        mock_batch2.id = "msgbatch_2"
        mock_batch2.processing_status = "ended"
        mock_batch2.request_counts = Mock(
            processing=0, succeeded=10, errored=0, canceled=0, expired=0
        )
        mock_batch2.created_at = "2024-01-01T00:00:00Z"
        mock_batch2.ended_at = "2024-01-01T01:00:00Z"
        mock_batch2.expires_at = "2024-01-02T00:00:00Z"
        mock_batch2.results_url = "https://api.anthropic.com/results"

        mock_list = Mock()
        mock_list.data = [mock_batch1, mock_batch2]
        provider.client.messages.batches.list = Mock(return_value=mock_list)

        batches = await provider.list_batches(limit=2)

        assert len(batches) == 2
        assert batches[0].batch_id == "msgbatch_1"
        assert batches[0].status == BatchStatus.IN_PROGRESS
        assert batches[1].batch_id == "msgbatch_2"
        assert batches[1].status == BatchStatus.COMPLETED


class TestAnthropicProviderFileOperations:
    """Test file I/O operations."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return AnthropicProvider(api_key="test-key")

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

        assert "anthropic" in str(path)
        assert "batch_batch123_unified.jsonl" in str(path)


class TestAnthropicProviderEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return AnthropicProvider(api_key="test-key")

    def test_convert_empty_content(self, provider):
        """Test conversion of empty content list."""
        result = provider._convert_content_to_anthropic([])
        assert result == []

    def test_convert_to_provider_format_with_provider_kwargs(self, provider):
        """Test conversion with additional provider-specific kwargs."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="claude-sonnet-4-5",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            provider_kwargs={"metadata": {"user_id": "123"}}
        )

        result = provider._convert_to_provider_format([request])

        assert result[0]["params"]["metadata"] == {"user_id": "123"}

    def test_convert_system_prompt_as_list(self, provider):
        """Test conversion of system prompt as list of strings."""
        request = UnifiedRequest(
            custom_id="req-1",
            model="claude-sonnet-4-5",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt=["You are helpful", "Be concise"]
        )

        result = provider._convert_to_provider_format([request])

        assert result[0]["params"]["system"] == ["You are helpful", "Be concise"]

    @pytest.mark.asyncio
    async def test_get_status_api_error(self, provider):
        """Test status retrieval with API error."""
        provider.client.messages.batches.retrieve = Mock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception, match="Failed to retrieve batch status"):
            await provider.get_status("msgbatch_test123")
