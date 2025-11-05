"""Tests for BaseProvider abstract class."""

import pytest
from pathlib import Path
from batch_router.core.base import BaseProvider
from batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from batch_router.core.responses import BatchStatusResponse, UnifiedResult
from batch_router.core.messages import UnifiedMessage


class MockProvider(BaseProvider):
    """Mock provider for testing BaseProvider."""

    def __init__(self, name: str = "mock", api_key: str = "test_key", **kwargs):
        super().__init__(name, api_key, **kwargs)

    def _validate_configuration(self) -> None:
        """Validate mock provider configuration."""
        if not self.api_key:
            raise ValueError("api_key is required for mock provider")

    def _convert_to_provider_format(self, requests):
        """Mock implementation."""
        return [{"mock": "request"}]

    def _convert_from_provider_format(self, provider_results):
        """Mock implementation."""
        return []

    async def send_batch(self, batch):
        """Mock implementation."""
        return "batch_123"

    async def get_status(self, batch_id):
        """Mock implementation."""
        from batch_router.core.responses import RequestCounts
        from batch_router.core.enums import BatchStatus
        return BatchStatusResponse(
            batch_id=batch_id,
            provider=self.name,
            status=BatchStatus.COMPLETED,
            request_counts=RequestCounts(total=1, succeeded=1),
            created_at="2024-01-01T00:00:00Z"
        )

    async def get_results(self, batch_id):
        """Mock implementation."""
        if False:
            yield  # Make this a generator
        return
        yield  # This line is unreachable but makes the function a generator

    async def cancel_batch(self, batch_id):
        """Mock implementation."""
        return True


class TestBaseProvider:
    """Tests for BaseProvider abstract class."""

    def test_provider_initialization(self):
        """Test initializing a provider."""
        provider = MockProvider(name="mock", api_key="test_key")
        assert provider.name == "mock"
        assert provider.api_key == "test_key"
        assert provider.config == {}

    def test_provider_initialization_with_config(self):
        """Test initializing provider with config."""
        provider = MockProvider(
            name="mock",
            api_key="test_key",
            timeout=120,
            base_url="https://api.example.com"
        )
        assert provider.config["timeout"] == 120
        assert provider.config["base_url"] == "https://api.example.com"

    def test_provider_validation_called(self):
        """Test that _validate_configuration is called."""
        with pytest.raises(ValueError, match="api_key is required"):
            MockProvider(name="mock", api_key=None)

    def test_get_batch_file_path(self):
        """Test get_batch_file_path method."""
        provider = MockProvider()
        path = provider.get_batch_file_path("batch_123", "unified")

        assert isinstance(path, Path)
        assert str(path).endswith("batch_batch_123_unified.jsonl")
        assert "mock" in str(path)
        assert ".batch_router/generated" in path.as_posix()

    def test_get_batch_file_path_different_types(self):
        """Test get_batch_file_path with different file types."""
        provider = MockProvider()

        for file_type in ["unified", "provider", "output", "results"]:
            path = provider.get_batch_file_path("batch_123", file_type)
            assert f"batch_123_{file_type}.jsonl" in path.as_posix()

    def test_get_batch_file_path_creates_directory(self):
        """Test that get_batch_file_path creates directory."""
        provider = MockProvider()
        path = provider.get_batch_file_path("batch_123", "unified")

        # Directory should be created
        assert path.parent.exists()
        assert path.parent.is_dir()

    def test_list_batches_not_implemented(self):
        """Test that list_batches raises NotImplementedError by default."""
        provider = MockProvider()

        with pytest.raises(NotImplementedError, match="does not support listing batches"):
            import asyncio
            asyncio.run(provider.list_batches())

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""

        # Try to create a class that doesn't implement abstract methods
        class IncompleteProvider(BaseProvider):
            def _validate_configuration(self):
                pass

        # Should raise TypeError because abstract methods are not implemented
        with pytest.raises(TypeError):
            IncompleteProvider("incomplete")  # type: ignore

    @pytest.mark.asyncio
    async def test_mock_provider_send_batch(self):
        """Test mock provider send_batch."""
        provider = MockProvider()
        batch = UnifiedBatchMetadata(
            provider="openai",  # Use valid provider name for testing
            requests=[
                UnifiedRequest(
                    custom_id="req1",
                    model="mock-model",
                    messages=[UnifiedMessage.from_text("user", "Hello")]
                )
            ]
        )
        batch_id = await provider.send_batch(batch)
        assert batch_id == "batch_123"

    @pytest.mark.asyncio
    async def test_mock_provider_get_status(self):
        """Test mock provider get_status."""
        provider = MockProvider()
        status = await provider.get_status("batch_123")
        assert status.batch_id == "batch_123"
        assert status.provider == "mock"

    @pytest.mark.asyncio
    async def test_mock_provider_cancel_batch(self):
        """Test mock provider cancel_batch."""
        provider = MockProvider()
        result = await provider.cancel_batch("batch_123")
        assert result is True
