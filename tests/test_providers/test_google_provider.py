"""Tests for Google GenAI provider."""

import pytest
import json
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

from batch_router.providers.google_provider import GoogleProvider
from batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from batch_router.core.messages import UnifiedMessage
from batch_router.core.content import TextContent, ImageContent
from batch_router.core.config import GenerationConfig
from batch_router.core.enums import BatchStatus, ResultStatus
from batch_router.exceptions import (
    ProviderError,
    BatchNotFoundError,
    BatchNotCompleteError,
)


class TestGoogleProviderInitialization:
    """Tests for GoogleProvider initialization and validation."""

    @patch('batch_router.providers.google_provider.genai.Client')
    def test_init_with_api_key(self, mock_client_class):
        """Test initialization with API key."""
        provider = GoogleProvider(api_key="test_key")
        assert provider.name == "google"
        assert provider.api_key == "test_key"
        mock_client_class.assert_called_once_with(api_key="test_key")

    @patch('batch_router.providers.google_provider.genai.Client')
    def test_init_with_env_var(self, mock_client_class):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key"}):
            provider = GoogleProvider()
            assert provider.api_key == "env_key"
            mock_client_class.assert_called_once_with(api_key="env_key")

    @patch('batch_router.providers.google_provider.genai.Client')
    def test_init_without_api_key_raises_error(self, mock_client_class):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key is required"):
                GoogleProvider()

    @patch('batch_router.providers.google_provider.genai.Client')
    def test_init_client_error(self, mock_client_class):
        """Test that client initialization errors are handled."""
        mock_client_class.side_effect = Exception("Client init failed")
        with pytest.raises(ProviderError, match="Failed to initialize"):
            GoogleProvider(api_key="test_key")


class TestGoogleProviderFormatConversion:
    """Tests for format conversion methods."""

    @patch('batch_router.providers.google_provider.genai.Client')
    def setup_provider(self, mock_client_class):
        """Setup provider for tests."""
        return GoogleProvider(api_key="test_key")

    def test_convert_text_content(self):
        """Test converting text content to Google format."""
        provider = self.setup_provider()
        text_content = TextContent(text="Hello, world!")

        result = provider._convert_content_to_google_format(text_content)

        assert result == {"text": "Hello, world!"}

    def test_convert_image_content_base64(self):
        """Test converting base64 image content to Google format."""
        provider = self.setup_provider()
        image_content = ImageContent(
            source_type="base64",
            media_type="image/jpeg",
            data="base64encodeddata"
        )

        result = provider._convert_content_to_google_format(image_content)

        assert result == {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": "base64encodeddata"
            }
        }

    def test_convert_image_content_file_uri(self):
        """Test converting file URI image content to Google format."""
        provider = self.setup_provider()
        image_content = ImageContent(
            source_type="file_uri",
            media_type="image/png",
            data="gs://bucket/image.png"
        )

        result = provider._convert_content_to_google_format(image_content)

        assert result == {
            "file_data": {
                "mime_type": "image/png",
                "file_uri": "gs://bucket/image.png"
            }
        }

    def test_convert_image_content_url_raises_error(self):
        """Test that URL source type raises ValueError."""
        provider = self.setup_provider()
        image_content = ImageContent(
            source_type="url",
            media_type="image/jpeg",
            data="https://example.com/image.jpg"
        )

        with pytest.raises(ValueError, match="URL source type.*not directly supported"):
            provider._convert_content_to_google_format(image_content)

    def test_convert_to_provider_format_simple(self):
        """Test converting simple unified request to Google format."""
        provider = self.setup_provider()
        request = UnifiedRequest(
            custom_id="req_1",
            model="gemini-2.0-flash-exp",
            messages=[UnifiedMessage.from_text("user", "Hello")]
        )

        result = provider._convert_to_provider_format([request])

        assert len(result) == 1
        assert result[0]["key"] == "req_1"
        assert result[0]["request"]["contents"][0]["role"] == "user"
        assert result[0]["request"]["contents"][0]["parts"][0]["text"] == "Hello"

    def test_convert_to_provider_format_with_system_prompt(self):
        """Test converting request with system prompt to Google format."""
        provider = self.setup_provider()
        request = UnifiedRequest(
            custom_id="req_2",
            model="gemini-2.0-flash-exp",
            messages=[UnifiedMessage.from_text("user", "What is 2+2?")],
            system_prompt="You are a math tutor."
        )

        result = provider._convert_to_provider_format([request])

        assert len(result) == 1
        assert "config" in result[0]["request"]
        assert "systemInstruction" in result[0]["request"]["config"]
        assert result[0]["request"]["config"]["systemInstruction"]["parts"][0]["text"] == "You are a math tutor."

    def test_convert_to_provider_format_with_generation_config(self):
        """Test converting request with generation config to Google format."""
        provider = self.setup_provider()
        request = UnifiedRequest(
            custom_id="req_3",
            model="gemini-2.0-flash-exp",
            messages=[UnifiedMessage.from_text("user", "Tell me a story")],
            generation_config=GenerationConfig(
                temperature=0.7,
                max_tokens=1000,
                top_p=0.9,
                top_k=40,
                stop_sequences=["END"]
            )
        )

        result = provider._convert_to_provider_format([request])

        assert len(result) == 1
        gen_config = result[0]["request"]["config"]["generationConfig"]
        assert gen_config["temperature"] == 0.7
        assert gen_config["maxOutputTokens"] == 1000
        assert gen_config["topP"] == 0.9
        assert gen_config["topK"] == 40
        assert gen_config["stopSequences"] == ["END"]

    def test_convert_to_provider_format_multimodal(self):
        """Test converting multimodal request to Google format."""
        provider = self.setup_provider()
        request = UnifiedRequest(
            custom_id="req_4",
            model="gemini-2.0-flash-exp",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        TextContent(text="What is in this image?"),
                        ImageContent(
                            source_type="base64",
                            media_type="image/jpeg",
                            data="base64data"
                        )
                    ]
                )
            ]
        )

        result = provider._convert_to_provider_format([request])

        assert len(result) == 1
        parts = result[0]["request"]["contents"][0]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "What is in this image?"
        assert "inline_data" in parts[1]

    def test_convert_from_provider_format_success(self):
        """Test converting successful Google results to unified format."""
        provider = self.setup_provider()
        google_results = [
            {
                "key": "req_1",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Hello, how can I help?"}],
                                "role": "model"
                            }
                        }
                    ]
                }
            }
        ]

        result = provider._convert_from_provider_format(google_results)

        assert len(result) == 1
        assert result[0].custom_id == "req_1"
        assert result[0].status == ResultStatus.SUCCEEDED
        assert result[0].response is not None
        assert "candidates" in result[0].response

    def test_convert_from_provider_format_error(self):
        """Test converting error Google results to unified format."""
        provider = self.setup_provider()
        google_results = [
            {
                "key": "req_2",
                "error": {
                    "code": 400,
                    "message": "Invalid request"
                }
            }
        ]

        result = provider._convert_from_provider_format(google_results)

        assert len(result) == 1
        assert result[0].custom_id == "req_2"
        assert result[0].status == ResultStatus.ERRORED
        assert result[0].error is not None
        assert result[0].error["code"] == 400


class TestGoogleProviderBatchOperations:
    """Tests for batch operations."""

    @patch('batch_router.providers.google_provider.genai.Client')
    def setup_provider(self, mock_client_class):
        """Setup provider with mocked client."""
        provider = GoogleProvider(api_key="test_key")
        provider.client = Mock()
        return provider

    @pytest.mark.asyncio
    async def test_send_batch(self):
        """Test sending a batch to Google."""
        provider = self.setup_provider()

        # Mock file upload and batch creation
        mock_uploaded_file = Mock()
        mock_uploaded_file.name = "files/test123"
        provider.client.files.upload = Mock(return_value=mock_uploaded_file)

        mock_batch_job = Mock()
        mock_batch_job.name = "batches/test_batch_123"
        provider.client.batches.create = Mock(return_value=mock_batch_job)

        # Create test batch
        request = UnifiedRequest(
            custom_id="req_1",
            model="gemini-2.0-flash-exp",
            messages=[UnifiedMessage.from_text("user", "Hello")]
        )
        batch = UnifiedBatchMetadata(
            provider="google",
            requests=[request]
        )

        # Send batch
        batch_id = await provider.send_batch(batch)

        # Verify
        assert batch_id == "batches/test_batch_123"
        provider.client.files.upload.assert_called_once()
        provider.client.batches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_status_in_progress(self):
        """Test getting status of in-progress batch."""
        provider = self.setup_provider()

        # Mock batch job status
        mock_batch_job = Mock()
        mock_batch_job.name = "batches/test_123"
        mock_state = Mock()
        mock_state.name = "JOB_STATE_RUNNING"  # Set name as attribute
        mock_batch_job.state = mock_state
        mock_batch_job.batch_stats = Mock()
        mock_batch_job.batch_stats.total_request_count = 10
        mock_batch_job.batch_stats.succeeded_request_count = 3
        mock_batch_job.batch_stats.failed_request_count = 0
        mock_batch_job.create_time = "2024-01-01T00:00:00Z"
        mock_batch_job.display_name = "test_batch"

        provider.client.batches.get = Mock(return_value=mock_batch_job)

        # Get status
        status = await provider.get_status("batches/test_123")

        # Verify
        assert status.batch_id == "batches/test_123"
        assert status.provider == "google"
        assert status.status == BatchStatus.IN_PROGRESS
        assert status.request_counts.total == 10
        assert status.request_counts.succeeded == 3
        assert status.request_counts.processing == 7

    @pytest.mark.asyncio
    async def test_get_status_completed(self):
        """Test getting status of completed batch."""
        provider = self.setup_provider()

        # Mock completed batch job
        mock_batch_job = Mock()
        mock_batch_job.name = "batches/test_123"
        mock_state = Mock()
        mock_state.name = "JOB_STATE_SUCCEEDED"  # Set name as attribute
        mock_batch_job.state = mock_state
        mock_batch_job.batch_stats = Mock()
        mock_batch_job.batch_stats.total_request_count = 10
        mock_batch_job.batch_stats.succeeded_request_count = 10
        mock_batch_job.batch_stats.failed_request_count = 0
        mock_batch_job.create_time = "2024-01-01T00:00:00Z"
        mock_batch_job.update_time = "2024-01-01T01:00:00Z"
        mock_batch_job.display_name = "test_batch"

        provider.client.batches.get = Mock(return_value=mock_batch_job)

        # Get status
        status = await provider.get_status("batches/test_123")

        # Verify
        assert status.status == BatchStatus.COMPLETED
        assert status.request_counts.succeeded == 10
        assert status.request_counts.processing == 0
        assert status.completed_at is not None

    @pytest.mark.asyncio
    async def test_get_status_not_found(self):
        """Test getting status of non-existent batch."""
        provider = self.setup_provider()
        provider.client.batches.get = Mock(side_effect=Exception("Batch not found"))

        with pytest.raises(BatchNotFoundError):
            await provider.get_status("batches/nonexistent")

    @pytest.mark.asyncio
    async def test_get_results_not_complete(self):
        """Test getting results from incomplete batch raises error."""
        provider = self.setup_provider()

        # Mock in-progress status
        mock_batch_job = Mock()
        mock_batch_job.name = "batches/test_123"
        mock_state = Mock()
        mock_state.name = "JOB_STATE_RUNNING"  # Set name as attribute
        mock_batch_job.state = mock_state
        mock_batch_job.batch_stats = Mock()
        mock_batch_job.batch_stats.total_request_count = 10
        mock_batch_job.batch_stats.succeeded_request_count = 3
        mock_batch_job.batch_stats.failed_request_count = 0
        mock_batch_job.create_time = "2024-01-01T00:00:00Z"

        provider.client.batches.get = Mock(return_value=mock_batch_job)

        with pytest.raises(BatchNotCompleteError):
            async for _ in provider.get_results("batches/test_123"):
                pass

    @pytest.mark.asyncio
    async def test_get_results_with_file(self):
        """Test getting results from batch with file output."""
        provider = self.setup_provider()

        # Mock completed status
        mock_batch_job = Mock()
        mock_batch_job.name = "batches/test_123"
        mock_state = Mock()
        mock_state.name = "JOB_STATE_SUCCEEDED"  # Set name as attribute
        mock_batch_job.state = mock_state
        mock_batch_job.batch_stats = Mock()
        mock_batch_job.batch_stats.total_request_count = 2
        mock_batch_job.batch_stats.succeeded_request_count = 2
        mock_batch_job.batch_stats.failed_request_count = 0
        mock_batch_job.create_time = "2024-01-01T00:00:00Z"
        mock_batch_job.update_time = "2024-01-01T01:00:00Z"

        # Mock file download
        mock_batch_job.dest = Mock()
        mock_batch_job.dest.file_name = "files/results_123"
        mock_batch_job.dest.inlined_responses = None

        results_jsonl = (
            '{"key": "req_1", "response": {"text": "Response 1"}}\n'
            '{"key": "req_2", "response": {"text": "Response 2"}}\n'
        )
        provider.client.batches.get = Mock(return_value=mock_batch_job)
        provider.client.files.download = Mock(return_value=results_jsonl.encode('utf-8'))

        # Mock _get_local_batch_id to avoid file system operations
        provider._get_local_batch_id = Mock(return_value="google_test_123")

        # Get results
        results = []
        async for result in provider.get_results("batches/test_123"):
            results.append(result)

        # Verify
        assert len(results) == 2
        assert results[0].custom_id == "req_1"
        assert results[0].status == ResultStatus.SUCCEEDED
        assert results[1].custom_id == "req_2"
        assert results[1].status == ResultStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_cancel_batch(self):
        """Test cancelling a batch."""
        provider = self.setup_provider()

        # Mock in-progress status
        mock_batch_job = Mock()
        mock_batch_job.name = "batches/test_123"
        mock_state = Mock()
        mock_state.name = "JOB_STATE_RUNNING"  # Set name as attribute
        mock_batch_job.state = mock_state
        mock_batch_job.batch_stats = Mock()
        mock_batch_job.batch_stats.total_request_count = 10
        mock_batch_job.batch_stats.succeeded_request_count = 3
        mock_batch_job.batch_stats.failed_request_count = 0
        mock_batch_job.create_time = "2024-01-01T00:00:00Z"

        provider.client.batches.get = Mock(return_value=mock_batch_job)
        provider.client.batches.cancel = Mock()

        # Cancel batch
        result = await provider.cancel_batch("batches/test_123")

        # Verify
        assert result is True
        provider.client.batches.cancel.assert_called_once_with(name="batches/test_123")

    @pytest.mark.asyncio
    async def test_cancel_batch_already_complete(self):
        """Test cancelling already complete batch returns False."""
        provider = self.setup_provider()

        # Mock completed status
        mock_batch_job = Mock()
        mock_batch_job.name = "batches/test_123"
        mock_state = Mock()
        mock_state.name = "JOB_STATE_SUCCEEDED"  # Set name as attribute
        mock_batch_job.state = mock_state
        mock_batch_job.batch_stats = Mock()
        mock_batch_job.batch_stats.total_request_count = 10
        mock_batch_job.batch_stats.succeeded_request_count = 10
        mock_batch_job.batch_stats.failed_request_count = 0
        mock_batch_job.create_time = "2024-01-01T00:00:00Z"
        mock_batch_job.update_time = "2024-01-01T01:00:00Z"

        provider.client.batches.get = Mock(return_value=mock_batch_job)

        # Try to cancel
        result = await provider.cancel_batch("batches/test_123")

        # Verify
        assert result is False
        # Cancel should not have been called
        provider.client.batches.cancel.assert_not_called()


class TestGoogleProviderHelperMethods:
    """Tests for helper methods."""

    @patch('batch_router.providers.google_provider.genai.Client')
    def setup_provider(self, mock_client_class):
        """Setup provider for tests."""
        return GoogleProvider(api_key="test_key")

    def test_response_to_dict(self):
        """Test converting Google response object to dict."""
        provider = self.setup_provider()

        # Create a mock response with nested structure
        mock_part = Mock()
        mock_part.text = "Hello, world!"

        mock_content = Mock()
        mock_content.parts = [mock_part]
        mock_content.role = "model"

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Hello, world!"

        result = provider._response_to_dict(mock_response)

        assert "candidates" in result
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["content"]["parts"][0]["text"] == "Hello, world!"
        assert result["text"] == "Hello, world!"

    def test_error_to_dict(self):
        """Test converting Google error object to dict."""
        provider = self.setup_provider()

        mock_error = Mock()
        mock_error.message = "Invalid request"
        mock_error.code = 400
        mock_error.details = "Missing required field"

        result = provider._error_to_dict(mock_error)

        assert result["message"] == "Invalid request"
        assert result["code"] == 400
        assert "Missing required field" in result["details"]

    def test_error_to_dict_with_dict(self):
        """Test converting dict error (already in dict format)."""
        provider = self.setup_provider()

        error_dict = {"message": "Error occurred", "code": 500}

        result = provider._error_to_dict(error_dict)

        assert result == error_dict


class TestGoogleProviderEdgeCases:
    """Tests for edge cases and error handling."""

    @patch('batch_router.providers.google_provider.genai.Client')
    def setup_provider(self, mock_client_class):
        """Setup provider for tests."""
        return GoogleProvider(api_key="test_key")

    def test_convert_to_provider_format_with_list_system_prompt(self):
        """Test converting request with list system prompt."""
        provider = self.setup_provider()
        request = UnifiedRequest(
            custom_id="req_1",
            model="gemini-2.0-flash-exp",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt=["You are helpful.", "You are concise."]
        )

        result = provider._convert_to_provider_format([request])

        system_text = result[0]["request"]["config"]["systemInstruction"]["parts"][0]["text"]
        assert "You are helpful." in system_text
        assert "You are concise." in system_text

    def test_convert_to_provider_format_multiple_requests(self):
        """Test converting multiple requests."""
        provider = self.setup_provider()
        requests = [
            UnifiedRequest(
                custom_id=f"req_{i}",
                model="gemini-2.0-flash-exp",
                messages=[UnifiedMessage.from_text("user", f"Query {i}")]
            )
            for i in range(5)
        ]

        result = provider._convert_to_provider_format(requests)

        assert len(result) == 5
        for i, req in enumerate(result):
            assert req["key"] == f"req_{i}"
            assert req["request"]["contents"][0]["parts"][0]["text"] == f"Query {i}"

    @pytest.mark.asyncio
    async def test_send_batch_error_handling(self):
        """Test error handling during batch send."""
        provider = self.setup_provider()
        provider.client.files.upload = Mock(side_effect=Exception("Upload failed"))

        request = UnifiedRequest(
            custom_id="req_1",
            model="gemini-2.0-flash-exp",
            messages=[UnifiedMessage.from_text("user", "Hello")]
        )
        batch = UnifiedBatchMetadata(provider="google", requests=[request])

        with pytest.raises(ProviderError, match="Failed to send batch"):
            await provider.send_batch(batch)

    @pytest.mark.asyncio
    async def test_get_status_with_missing_stats(self):
        """Test getting status when batch_stats is missing."""
        provider = self.setup_provider()

        mock_batch_job = Mock()
        mock_batch_job.name = "batches/test_123"
        mock_state = Mock()
        mock_state.name = "JOB_STATE_RUNNING"  # Set name as attribute
        mock_batch_job.state = mock_state
        mock_batch_job.batch_stats = None  # Missing stats
        mock_batch_job.create_time = "2024-01-01T00:00:00Z"
        mock_batch_job.display_name = "test_batch"

        provider.client.batches.get = Mock(return_value=mock_batch_job)

        status = await provider.get_status("batches/test_123")

        # Should still return valid status with estimated counts
        assert status.status == BatchStatus.IN_PROGRESS
        assert status.request_counts.total >= 0
