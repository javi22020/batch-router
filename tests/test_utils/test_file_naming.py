"""Tests for file naming and sanitization utilities."""

import pytest
from pathlib import Path
from batch_router.core.enums import Modality
from batch_router.utilities import sanitize_filename_component
from batch_router.core.base import BaseProvider
from batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from batch_router.core.responses import BatchStatusResponse, UnifiedResult
from batch_router.core.messages import UnifiedMessage


class TestSanitizeFilenameComponent:
    """Tests for sanitize_filename_component function."""

    def test_sanitize_basic_string(self):
        """Test sanitizing a basic alphanumeric string."""
        result = sanitize_filename_component("test123")
        assert result == "test123"

    def test_sanitize_with_underscores(self):
        """Test that underscores are replaced with dashes."""
        result = sanitize_filename_component("my_model_name")
        assert result == "my-model-name"

    def test_sanitize_with_dashes(self):
        """Test that dashes are preserved."""
        result = sanitize_filename_component("gpt-4o")
        assert result == "gpt-4o"

    def test_sanitize_with_special_characters(self):
        """Test that special characters are removed."""
        result = sanitize_filename_component("test@#$batch!%")
        assert result == "testbatch"

    def test_sanitize_mixed_case(self):
        """Test that mixed case is preserved."""
        result = sanitize_filename_component("MyModel123")
        assert result == "MyModel123"

    def test_sanitize_complex_string(self):
        """Test sanitizing a complex string."""
        result = sanitize_filename_component("my_test@model-v2.1!")
        assert result == "my-testmodel-v21"

    def test_sanitize_only_special_characters(self):
        """Test sanitizing a string with only special characters."""
        result = sanitize_filename_component("@#$%^&*()")
        assert result == ""

    def test_sanitize_empty_string(self):
        """Test sanitizing an empty string."""
        result = sanitize_filename_component("")
        assert result == ""

    def test_sanitize_spaces(self):
        """Test that spaces are removed."""
        result = sanitize_filename_component("my model name")
        assert result == "mymodelname"

    def test_sanitize_dots(self):
        """Test that dots are removed."""
        result = sanitize_filename_component("model.v1.2.3")
        assert result == "modelv123"


class MockProvider(BaseProvider):
    """Mock provider for testing file naming."""
    
    supported_modalities = {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.DOCUMENT}

    def _validate_configuration(self):
        pass

    def _convert_to_provider_format(self, requests):
        return []

    def _convert_from_provider_format(self, provider_results):
        return []

    async def send_batch(self, batch):
        return "test_batch_id"

    async def get_status(self, batch_id):
        return BatchStatusResponse(
            batch_id=batch_id,
            provider=self.name,
            status="completed",
            request_counts={"total": 1, "completed": 1}
        )

    async def get_results(self, batch_id):
        return
        yield  # Make this an async generator

    async def cancel_batch(self, batch_id):
        return True


class TestCustomFileNaming:
    """Tests for custom file naming with batch metadata."""

    def test_default_file_naming(self):
        """Test default file naming without custom name."""
        provider = MockProvider(name="test_provider")
        batch_id = "batch_123"

        path = provider.get_batch_file_path(batch_id, "unified")

        assert path == Path(".batch_router/generated/test_provider/batch_batch_123_unified.jsonl")

    def test_custom_file_naming(self):
        """Test custom file naming with name and model."""
        provider = MockProvider(name="openai")
        batch_id = "batch_123"

        path = provider.get_batch_file_path(
            batch_id,
            "unified",
            custom_name="my_experiment",
            model="gpt-4o"
        )

        expected = Path(".batch_router/generated/openai/my-experiment_gpt-4o_openai_unified.jsonl")
        assert path == expected

    def test_custom_file_naming_all_types(self):
        """Test custom file naming for all file types."""
        provider = MockProvider(name="anthropic")
        batch_id = "batch_456"
        custom_name = "test_batch"
        model = "claude-sonnet-4"

        file_types = ["unified", "provider", "output", "results"]
        for file_type in file_types:
            path = provider.get_batch_file_path(
                batch_id,
                file_type,
                custom_name=custom_name,
                model=model
            )

            expected_filename = f"test-batch_claude-sonnet-4_anthropic_{file_type}.jsonl"
            assert path.name == expected_filename
            assert path.parent == Path(".batch_router/generated/anthropic")

    def test_custom_file_naming_with_sanitization(self):
        """Test that custom file naming sanitizes all components."""
        provider = MockProvider(name="test_provider")
        batch_id = "batch_789"

        path = provider.get_batch_file_path(
            batch_id,
            "results",
            custom_name="my@special#batch!",
            model="gpt_4o_mini"
        )

        # Note: directory name uses provider.name (test_provider), not sanitized
        # Only the filename components are sanitized
        expected = Path(".batch_router/generated/test_provider/myspecialbatch_gpt-4o-mini_test-provider_results.jsonl")
        assert path == expected

    def test_custom_naming_only_with_both_params(self):
        """Test that custom naming requires both name and model."""
        provider = MockProvider(name="openai")
        batch_id = "batch_123"

        # Only custom_name provided - should use default naming
        path1 = provider.get_batch_file_path(
            batch_id,
            "unified",
            custom_name="my_batch",
            model=None
        )
        assert "batch_batch_123" in str(path1)

        # Only model provided - should use default naming
        path2 = provider.get_batch_file_path(
            batch_id,
            "unified",
            custom_name=None,
            model="gpt-4o"
        )
        assert "batch_batch_123" in str(path2)

    def test_metadata_save_and_load(self):
        """Test saving and loading batch metadata."""
        provider = MockProvider(name="test_provider")
        batch_id = "test_batch_001"
        custom_name = "experiment_1"
        model = "gpt-4o"

        # Save metadata
        provider._save_batch_metadata(batch_id, custom_name, model)

        # Load metadata
        loaded_name, loaded_model = provider._load_batch_metadata(batch_id)

        assert loaded_name == custom_name
        assert loaded_model == model

    def test_metadata_load_nonexistent(self):
        """Test loading metadata for non-existent batch."""
        provider = MockProvider(name="test_provider")
        batch_id = "nonexistent_batch"

        custom_name, model = provider._load_batch_metadata(batch_id)

        assert custom_name is None
        assert model is None

    def test_metadata_save_skip_if_none(self):
        """Test that metadata save is skipped if name or model is None."""
        provider = MockProvider(name="test_provider")
        batch_id = "test_batch_002"

        # Save with None values
        provider._save_batch_metadata(batch_id, None, None)

        # Should not create metadata file
        meta_file = Path(".batch_router/generated/test_provider") / f"batch_{batch_id}.meta.json"
        assert not meta_file.exists()

        # Try loading - should return None
        custom_name, model = provider._load_batch_metadata(batch_id)
        assert custom_name is None
        assert model is None
