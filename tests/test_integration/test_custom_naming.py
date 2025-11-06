"""Integration tests for custom file naming feature."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from batch_router.providers.openai_provider import OpenAIProvider
from batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from batch_router.core.messages import UnifiedMessage


class TestCustomNamingIntegration:
    """Integration tests for custom file naming across providers."""

    @pytest.mark.asyncio
    async def test_openai_custom_naming_send_batch(self):
        """Test that OpenAI provider uses custom naming when sending batch."""
        provider = OpenAIProvider(api_key="sk-test-123")

        # Mock API responses
        mock_file_response = Mock()
        mock_file_response.id = "file_abc123"

        mock_batch_response = Mock()
        mock_batch_response.id = "batch_custom_123"

        provider.async_client.files.create = AsyncMock(return_value=mock_file_response)
        provider.async_client.batches.create = AsyncMock(return_value=mock_batch_response)

        # Create batch with custom name
        batch = UnifiedBatchMetadata(
            provider="openai",
            requests=[
                UnifiedRequest(
                    custom_id="test-1",
                    model="gpt-4o",
                    messages=[UnifiedMessage.from_text("user", "Hello!")]
                )
            ],
            name="my_experiment"
        )

        # Send batch
        batch_id = await provider.send_batch(batch)

        # Check that files were created with custom naming
        expected_unified = Path(".batch_router/generated/openai/my-experiment_gpt-4o_openai_unified.jsonl")
        expected_provider = Path(".batch_router/generated/openai/my-experiment_gpt-4o_openai_provider.jsonl")

        assert expected_unified.exists()
        assert expected_provider.exists()

        # Cleanup
        expected_unified.unlink(missing_ok=True)
        expected_provider.unlink(missing_ok=True)

        # Cleanup metadata file
        meta_file = Path(".batch_router/generated/openai") / f"batch_{batch_id}.meta.json"
        meta_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_openai_default_naming_send_batch(self):
        """Test that OpenAI provider uses default naming when no custom name."""
        provider = OpenAIProvider(api_key="sk-test-123")

        # Mock API responses
        mock_file_response = Mock()
        mock_file_response.id = "file_def456"

        mock_batch_response = Mock()
        mock_batch_response.id = "batch_default_456"

        provider.async_client.files.create = AsyncMock(return_value=mock_file_response)
        provider.async_client.batches.create = AsyncMock(return_value=mock_batch_response)

        # Create batch without custom name
        batch = UnifiedBatchMetadata(
            provider="openai",
            requests=[
                UnifiedRequest(
                    custom_id="test-1",
                    model="gpt-4o",
                    messages=[UnifiedMessage.from_text("user", "Hello!")]
                )
            ]
        )

        # Send batch
        batch_id = await provider.send_batch(batch)

        # Check that files were created with default naming
        expected_unified = Path(f".batch_router/generated/openai/batch_{batch_id}_unified.jsonl")
        expected_provider = Path(f".batch_router/generated/openai/batch_{batch_id}_provider.jsonl")

        assert expected_unified.exists()
        assert expected_provider.exists()

        # Cleanup
        expected_unified.unlink(missing_ok=True)
        expected_provider.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_metadata_persistence(self):
        """Test that custom naming metadata persists across send_batch and get_results."""
        provider = OpenAIProvider(api_key="sk-test-123")

        # Mock send_batch
        mock_file_response = Mock()
        mock_file_response.id = "file_xyz789"

        mock_batch_response = Mock()
        mock_batch_response.id = "batch_persist_789"

        provider.async_client.files.create = AsyncMock(return_value=mock_file_response)
        provider.async_client.batches.create = AsyncMock(return_value=mock_batch_response)

        # Create and send batch with custom name
        batch = UnifiedBatchMetadata(
            provider="openai",
            requests=[
                UnifiedRequest(
                    custom_id="test-1",
                    model="gpt-4o",
                    messages=[UnifiedMessage.from_text("user", "Test")]
                )
            ],
            name="persistent_test"
        )

        batch_id = await provider.send_batch(batch)

        # Verify metadata was saved
        custom_name, model = provider._load_batch_metadata(batch_id)
        assert custom_name == "persistent_test"
        assert model == "gpt-4o"

        # Mock get_results
        mock_batch = Mock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file_output_123"

        mock_file_content = Mock()
        mock_file_content.text = '{"custom_id": "test-1", "response": {"status_code": 200}}'

        provider.async_client.batches.retrieve = AsyncMock(return_value=mock_batch)
        provider.async_client.files.content = AsyncMock(return_value=mock_file_content)

        # Get results - should use same custom naming
        results = []
        async for result in provider.get_results(batch_id):
            results.append(result)

        # Check that output and results files were created with custom naming
        expected_output = Path(".batch_router/generated/openai/persistent-test_gpt-4o_openai_output.jsonl")
        expected_results = Path(".batch_router/generated/openai/persistent-test_gpt-4o_openai_results.jsonl")

        assert expected_output.exists()
        assert expected_results.exists()

        # Cleanup
        for path in [expected_output, expected_results]:
            path.unlink(missing_ok=True)

        # Cleanup unified and provider files
        unified_path = Path(".batch_router/generated/openai/persistent-test_gpt-4o_openai_unified.jsonl")
        provider_path = Path(".batch_router/generated/openai/persistent-test_gpt-4o_openai_provider.jsonl")
        meta_file = Path(".batch_router/generated/openai") / f"batch_{batch_id}.meta.json"

        for path in [unified_path, provider_path, meta_file]:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_special_characters_sanitization(self):
        """Test that special characters in name and model are properly sanitized."""
        provider = OpenAIProvider(api_key="sk-test-123")

        # Mock API responses
        mock_file_response = Mock()
        mock_file_response.id = "file_special_123"

        mock_batch_response = Mock()
        mock_batch_response.id = "batch_special_123"

        provider.async_client.files.create = AsyncMock(return_value=mock_file_response)
        provider.async_client.batches.create = AsyncMock(return_value=mock_batch_response)

        # Create batch with special characters in name
        batch = UnifiedBatchMetadata(
            provider="openai",
            requests=[
                UnifiedRequest(
                    custom_id="test-1",
                    model="gpt_4o_mini",  # Underscores should become dashes
                    messages=[UnifiedMessage.from_text("user", "Test")]
                )
            ],
            name="my@special#test!"  # Special chars should be removed
        )

        batch_id = await provider.send_batch(batch)

        # Check that files were created with sanitized names
        # my@special#test! -> myspecialtest
        # gpt_4o_mini -> gpt-4o-mini
        expected_unified = Path(".batch_router/generated/openai/myspecialtest_gpt-4o-mini_openai_unified.jsonl")
        expected_provider = Path(".batch_router/generated/openai/myspecialtest_gpt-4o-mini_openai_provider.jsonl")

        assert expected_unified.exists()
        assert expected_provider.exists()

        # Cleanup
        expected_unified.unlink(missing_ok=True)
        expected_provider.unlink(missing_ok=True)

        meta_file = Path(".batch_router/generated/openai") / f"batch_{batch_id}.meta.json"
        meta_file.unlink(missing_ok=True)
