"""Tests for OpenAI provider implementation."""

import pytest
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from batch_router.providers.openai_provider import OpenAIProvider
from batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from batch_router.core.messages import UnifiedMessage
from batch_router.core.content import TextContent, ImageContent
from batch_router.core.config import GenerationConfig
from batch_router.core.enums import BatchStatus, ResultStatus


class TestOpenAIProviderConfiguration:
    """Test provider configuration and initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key parameter."""
        provider = OpenAIProvider(api_key="sk-test-123")
        assert provider.name == "openai"
        assert provider.api_key == "sk-test-123"

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-123")
        provider = OpenAIProvider()
        assert provider.api_key == "sk-env-123"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization fails without API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIProvider()

    def test_init_with_base_url(self):
        """Test initialization with custom base URL."""
        provider = OpenAIProvider(
            api_key="sk-test-123",
            base_url="https://custom.openai.com"
        )
        assert provider.api_key == "sk-test-123"


class TestOpenAIProviderFormatConversion:
    """Test conversion between unified and OpenAI formats."""

    def test_convert_simple_text_request(self):
        """Test conversion of simple text-only request."""
        provider = OpenAIProvider(api_key="sk-test-123")

        request = UnifiedRequest(
            custom_id="test-1",
            model="gpt-4o",
            messages=[
                UnifiedMessage.from_text("user", "Hello, world!")
            ]
        )

        result = provider._convert_to_provider_format([request])

        assert len(result) == 1
        assert result[0]["custom_id"] == "test-1"
        assert result[0]["method"] == "POST"
        assert result[0]["url"] == "/v1/chat/completions"
        assert result[0]["body"]["model"] == "gpt-4o"
        assert len(result[0]["body"]["messages"]) == 1
        assert result[0]["body"]["messages"][0]["role"] == "user"
        assert result[0]["body"]["messages"][0]["content"] == "Hello, world!"

    def test_convert_request_with_system_prompt(self):
        """Test system prompt is converted to system message."""
        provider = OpenAIProvider(api_key="sk-test-123")

        request = UnifiedRequest(
            custom_id="test-1",
            model="gpt-4o",
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

    def test_convert_request_with_system_prompt_list(self):
        """Test system prompt list is joined."""
        provider = OpenAIProvider(api_key="sk-test-123")

        request = UnifiedRequest(
            custom_id="test-1",
            model="gpt-4o",
            messages=[
                UnifiedMessage.from_text("user", "Hello!")
            ],
            system_prompt=["You are helpful.", "You are concise."]
        )

        result = provider._convert_to_provider_format([request])

        messages = result[0]["body"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful.\nYou are concise."

    def test_convert_request_with_generation_config(self):
        """Test generation config parameters are included."""
        provider = OpenAIProvider(api_key="sk-test-123")

        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            stop_sequences=["END"]
        )

        request = UnifiedRequest(
            custom_id="test-1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello!")],
            generation_config=config
        )

        result = provider._convert_to_provider_format([request])

        body = result[0]["body"]
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.9
        assert body["presence_penalty"] == 0.5
        assert body["frequency_penalty"] == 0.3
        assert body["stop"] == ["END"]

    def test_convert_multimodal_message_text_and_image_url(self):
        """Test multimodal message with text and image URL."""
        provider = OpenAIProvider(api_key="sk-test-123")

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
            model="gpt-4o",
            messages=[message]
        )

        result = provider._convert_to_provider_format([request])

        content = result[0]["body"]["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What's in this image?"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/image.jpg"

    def test_convert_multimodal_message_base64_image(self):
        """Test multimodal message with base64 image."""
        provider = OpenAIProvider(api_key="sk-test-123")

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
            model="gpt-4o",
            messages=[message]
        )

        result = provider._convert_to_provider_format([request])

        content = result[0]["body"]["messages"][0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_convert_conversation(self):
        """Test multi-turn conversation."""
        provider = OpenAIProvider(api_key="sk-test-123")

        request = UnifiedRequest(
            custom_id="test-1",
            model="gpt-4o",
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

    def test_convert_multiple_requests(self):
        """Test conversion of multiple requests in batch."""
        provider = OpenAIProvider(api_key="sk-test-123")

        requests = [
            UnifiedRequest(
                custom_id="req-1",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hello 1")]
            ),
            UnifiedRequest(
                custom_id="req-2",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hello 2")]
            )
        ]

        result = provider._convert_to_provider_format(requests)

        assert len(result) == 2
        assert result[0]["custom_id"] == "req-1"
        assert result[1]["custom_id"] == "req-2"


class TestOpenAIProviderResultConversion:
    """Test conversion of OpenAI results to unified format."""

    def test_convert_successful_result(self):
        """Test conversion of successful response."""
        provider = OpenAIProvider(api_key="sk-test-123")

        openai_result = {
            "id": "batch_req_123",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "request_id": "req_abc",
                "body": {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
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

        result = provider._convert_from_provider_format([openai_result])

        assert len(result) == 1
        assert result[0].custom_id == "request-1"
        assert result[0].status == ResultStatus.SUCCEEDED
        assert result[0].response is not None
        assert result[0].response["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert result[0].error is None

    def test_convert_error_result(self):
        """Test conversion of error response."""
        provider = OpenAIProvider(api_key="sk-test-123")

        openai_result = {
            "id": "batch_req_123",
            "custom_id": "request-1",
            "response": None,
            "error": {
                "code": "invalid_request",
                "message": "Invalid model specified"
            }
        }

        result = provider._convert_from_provider_format([openai_result])

        assert len(result) == 1
        assert result[0].custom_id == "request-1"
        assert result[0].status == ResultStatus.ERRORED
        assert result[0].response is None
        assert result[0].error["code"] == "invalid_request"
        assert result[0].error["message"] == "Invalid model specified"

    def test_convert_http_error_result(self):
        """Test conversion of non-200 HTTP response."""
        provider = OpenAIProvider(api_key="sk-test-123")

        openai_result = {
            "id": "batch_req_123",
            "custom_id": "request-1",
            "response": {
                "status_code": 429,
                "request_id": "req_abc",
                "body": {}
            },
            "error": None
        }

        result = provider._convert_from_provider_format([openai_result])

        assert len(result) == 1
        assert result[0].status == ResultStatus.ERRORED
        assert result[0].error["code"] == "http_429"

    def test_get_text_response(self):
        """Test extracting text from unified result."""
        provider = OpenAIProvider(api_key="sk-test-123")

        openai_result = {
            "id": "batch_req_123",
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

        result = provider._convert_from_provider_format([openai_result])
        text = result[0].get_text_response()

        assert text == "Test response"


class TestOpenAIProviderBatchOperations:
    """Test batch operations (mocked API calls)."""

    @pytest.mark.asyncio
    async def test_send_batch(self):
        """Test sending a batch."""
        provider = OpenAIProvider(api_key="sk-test-123")

        # Mock the file upload and batch creation
        mock_file_response = Mock()
        mock_file_response.id = "file-123"

        mock_batch_response = Mock()
        mock_batch_response.id = "batch_abc123"

        provider.async_client.files.create = AsyncMock(return_value=mock_file_response)
        provider.async_client.batches.create = AsyncMock(return_value=mock_batch_response)

        # Create batch metadata
        batch = UnifiedBatchMetadata(
            provider="openai",
            requests=[
                UnifiedRequest(
                    custom_id="test-1",
                    model="gpt-4o",
                    messages=[UnifiedMessage.from_text("user", "Hello!")]
                )
            ],
            metadata={"test": "metadata"}
        )

        # Send batch
        batch_id = await provider.send_batch(batch)

        assert batch_id == "batch_abc123"
        provider.async_client.files.create.assert_called_once()
        provider.async_client.batches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting batch status."""
        provider = OpenAIProvider(api_key="sk-test-123")

        # Mock batch response
        mock_batch = Mock()
        mock_batch.id = "batch_123"
        mock_batch.status = "in_progress"
        mock_batch.created_at = 1234567890
        mock_batch.completed_at = None
        mock_batch.expires_at = 1234654290
        mock_batch.input_file_id = "file-in-123"
        mock_batch.output_file_id = None
        mock_batch.error_file_id = None
        mock_batch.request_counts = Mock()
        mock_batch.request_counts.total = 10
        mock_batch.request_counts.completed = 3
        mock_batch.request_counts.failed = 1

        provider.async_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        # Get status
        status = await provider.get_status("batch_123")

        assert status.batch_id == "batch_123"
        assert status.provider == "openai"
        assert status.status == BatchStatus.IN_PROGRESS
        assert status.request_counts.total == 10
        assert status.request_counts.succeeded == 3
        assert status.request_counts.errored == 1
        assert status.request_counts.processing == 6

    @pytest.mark.asyncio
    async def test_get_status_completed(self):
        """Test getting status of completed batch."""
        provider = OpenAIProvider(api_key="sk-test-123")

        mock_batch = Mock()
        mock_batch.id = "batch_123"
        mock_batch.status = "completed"
        mock_batch.created_at = 1234567890
        mock_batch.completed_at = 1234654290
        mock_batch.expires_at = 1234740690
        mock_batch.input_file_id = "file-in-123"
        mock_batch.output_file_id = "file-out-123"
        mock_batch.error_file_id = None
        mock_batch.request_counts = Mock()
        mock_batch.request_counts.total = 10
        mock_batch.request_counts.completed = 10
        mock_batch.request_counts.failed = 0

        provider.async_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        status = await provider.get_status("batch_123")

        assert status.status == BatchStatus.COMPLETED
        assert status.completed_at is not None
        assert status.request_counts.processing == 0

    @pytest.mark.asyncio
    async def test_get_results(self):
        """Test getting results from completed batch."""
        provider = OpenAIProvider(api_key="sk-test-123")

        # Mock batch
        mock_batch = Mock()
        mock_batch.id = "batch_123"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-out-123"

        # Mock file content
        output_data = [
            {
                "id": "batch_req_1",
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
                "id": "batch_req_2",
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

        output_content = "\n".join(json.dumps(item) for item in output_data)

        mock_file_response = Mock()
        mock_file_response.text = output_content

        provider.async_client.batches.retrieve = AsyncMock(return_value=mock_batch)
        provider.async_client.files.content = AsyncMock(return_value=mock_file_response)

        # Get results
        results = []
        async for result in provider.get_results("batch_123"):
            results.append(result)

        assert len(results) == 2
        assert results[0].custom_id == "req-1"
        assert results[0].status == ResultStatus.SUCCEEDED
        assert results[1].custom_id == "req-2"
        assert results[1].status == ResultStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_get_results_not_complete(self):
        """Test getting results from incomplete batch raises error."""
        provider = OpenAIProvider(api_key="sk-test-123")

        mock_batch = Mock()
        mock_batch.status = "in_progress"
        mock_batch.output_file_id = None

        provider.async_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        with pytest.raises(ValueError, match="is not complete"):
            async for result in provider.get_results("batch_123"):
                pass

    @pytest.mark.asyncio
    async def test_cancel_batch(self):
        """Test cancelling a batch."""
        provider = OpenAIProvider(api_key="sk-test-123")

        mock_batch = Mock()
        mock_batch.status = "cancelling"

        provider.async_client.batches.cancel = AsyncMock(return_value=mock_batch)

        result = await provider.cancel_batch("batch_123")

        assert result is True
        provider.async_client.batches.cancel.assert_called_once_with("batch_123")

    @pytest.mark.asyncio
    async def test_list_batches(self):
        """Test listing recent batches."""
        provider = OpenAIProvider(api_key="sk-test-123")

        # Mock batches list
        mock_batch_1 = Mock()
        mock_batch_1.id = "batch_1"
        mock_batch_1.status = "completed"
        mock_batch_1.created_at = 1234567890
        mock_batch_1.completed_at = 1234654290
        mock_batch_1.expires_at = 1234740690
        mock_batch_1.input_file_id = "file-1"
        mock_batch_1.output_file_id = "file-out-1"
        mock_batch_1.error_file_id = None
        mock_batch_1.request_counts = Mock()
        mock_batch_1.request_counts.total = 5
        mock_batch_1.request_counts.completed = 5
        mock_batch_1.request_counts.failed = 0

        mock_batch_2 = Mock()
        mock_batch_2.id = "batch_2"
        mock_batch_2.status = "in_progress"
        mock_batch_2.created_at = 1234567890
        mock_batch_2.completed_at = None
        mock_batch_2.expires_at = 1234654290
        mock_batch_2.input_file_id = "file-2"
        mock_batch_2.output_file_id = None
        mock_batch_2.error_file_id = None
        mock_batch_2.request_counts = Mock()
        mock_batch_2.request_counts.total = 10
        mock_batch_2.request_counts.completed = 3
        mock_batch_2.request_counts.failed = 0

        mock_page = Mock()
        mock_page.data = [mock_batch_1, mock_batch_2]

        provider.async_client.batches.list = AsyncMock(return_value=mock_page)

        # List batches
        batches = await provider.list_batches(limit=10)

        assert len(batches) == 2
        assert batches[0].batch_id == "batch_1"
        assert batches[0].status == BatchStatus.COMPLETED
        assert batches[1].batch_id == "batch_2"
        assert batches[1].status == BatchStatus.IN_PROGRESS


class TestOpenAIProviderEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_system_prompt(self):
        """Test request with None system prompt."""
        provider = OpenAIProvider(api_key="sk-test-123")

        request = UnifiedRequest(
            custom_id="test-1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello!")],
            system_prompt=None
        )

        result = provider._convert_to_provider_format([request])

        # Should only have user message, no system message
        messages = result[0]["body"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_provider_kwargs(self):
        """Test provider-specific kwargs are passed through."""
        provider = OpenAIProvider(api_key="sk-test-123")

        request = UnifiedRequest(
            custom_id="test-1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello!")],
            provider_kwargs={"logprobs": True, "top_logprobs": 5}
        )

        result = provider._convert_to_provider_format([request])

        body = result[0]["body"]
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5

    def test_partial_generation_config(self):
        """Test generation config with only some parameters."""
        provider = OpenAIProvider(api_key="sk-test-123")

        config = GenerationConfig(
            temperature=0.5,
            max_tokens=50
            # Other parameters None
        )

        request = UnifiedRequest(
            custom_id="test-1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello!")],
            generation_config=config
        )

        result = provider._convert_to_provider_format([request])

        body = result[0]["body"]
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 50
        assert "top_p" not in body
        assert "presence_penalty" not in body
