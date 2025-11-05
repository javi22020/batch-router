"""Tests for request structures."""

import pytest
from batch_router.core.requests import UnifiedRequest, UnifiedBatchMetadata
from batch_router.core.messages import UnifiedMessage
from batch_router.core.content import TextContent
from batch_router.core.config import GenerationConfig


class TestUnifiedRequest:
    """Tests for UnifiedRequest dataclass."""

    def test_request_creation(self):
        """Test creating a basic UnifiedRequest."""
        request = UnifiedRequest(
            custom_id="req1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello")]
        )
        assert request.custom_id == "req1"
        assert request.model == "gpt-4o"
        assert len(request.messages) == 1
        assert request.system_prompt is None
        assert request.generation_config is None
        assert request.provider_kwargs == {}

    def test_request_with_system_prompt(self):
        """Test UnifiedRequest with system prompt."""
        request = UnifiedRequest(
            custom_id="req1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt="You are a helpful assistant"
        )
        assert request.system_prompt == "You are a helpful assistant"

    def test_request_with_system_prompt_list(self):
        """Test UnifiedRequest with system prompt as list."""
        request = UnifiedRequest(
            custom_id="req1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt=["Line 1", "Line 2"]
        )
        assert request.system_prompt == ["Line 1", "Line 2"]

    def test_request_with_generation_config(self):
        """Test UnifiedRequest with generation config."""
        config = GenerationConfig(temperature=0.7, max_tokens=100)
        request = UnifiedRequest(
            custom_id="req1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            generation_config=config
        )
        assert request.generation_config is not None
        assert request.generation_config.temperature == 0.7
        assert request.generation_config.max_tokens == 100

    def test_request_with_provider_kwargs(self):
        """Test UnifiedRequest with provider kwargs."""
        request = UnifiedRequest(
            custom_id="req1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            provider_kwargs={"timeout": 120}
        )
        assert request.provider_kwargs == {"timeout": 120}

    def test_empty_custom_id_raises_error(self):
        """Test that empty custom_id raises ValueError."""
        with pytest.raises(ValueError, match="custom_id is required"):
            UnifiedRequest(
                custom_id="",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hello")]
            )

    def test_empty_messages_raises_error(self):
        """Test that empty messages list raises ValueError."""
        with pytest.raises(ValueError, match="messages list cannot be empty"):
            UnifiedRequest(
                custom_id="req1",
                model="gpt-4o",
                messages=[]
            )

    def test_system_role_in_messages_raises_error(self):
        """Test that system role in messages raises ValueError."""
        with pytest.raises(ValueError, match="Use system_prompt field"):
            # Try to create a message with system role (will fail in UnifiedMessage)
            # but let's test the validation in UnifiedRequest
            system_msg = UnifiedMessage.__new__(UnifiedMessage)
            system_msg.role = "system"  # type: ignore
            system_msg.content = [TextContent(text="System")]
            system_msg.provider_kwargs = {}

            UnifiedRequest(
                custom_id="req1",
                model="gpt-4o",
                messages=[system_msg]
            )

    def test_to_dict(self):
        """Test converting UnifiedRequest to dictionary."""
        request = UnifiedRequest(
            custom_id="req1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt="Be helpful",
            generation_config=GenerationConfig(temperature=0.7),
            provider_kwargs={"test": "value"}
        )
        request_dict = request.to_dict()

        assert request_dict["custom_id"] == "req1"
        assert request_dict["model"] == "gpt-4o"
        assert len(request_dict["messages"]) == 1
        assert request_dict["system_prompt"] == "Be helpful"
        assert "temperature" in request_dict["generation_config"]
        assert request_dict["provider_kwargs"] == {"test": "value"}

    def test_to_dict_minimal(self):
        """Test to_dict with minimal required fields."""
        request = UnifiedRequest(
            custom_id="req1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello")]
        )
        request_dict = request.to_dict()

        assert "custom_id" in request_dict
        assert "model" in request_dict
        assert "messages" in request_dict
        assert "system_prompt" not in request_dict
        assert "generation_config" not in request_dict
        assert "provider_kwargs" not in request_dict


class TestUnifiedBatchMetadata:
    """Tests for UnifiedBatchMetadata dataclass."""

    def test_batch_metadata_creation(self):
        """Test creating UnifiedBatchMetadata."""
        requests = [
            UnifiedRequest(
                custom_id="req1",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hello")]
            )
        ]
        batch = UnifiedBatchMetadata(
            provider="openai",
            requests=requests
        )
        assert batch.provider == "openai"
        assert len(batch.requests) == 1
        assert batch.metadata == {}

    def test_batch_metadata_with_metadata(self):
        """Test UnifiedBatchMetadata with custom metadata."""
        requests = [
            UnifiedRequest(
                custom_id="req1",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hello")]
            )
        ]
        batch = UnifiedBatchMetadata(
            provider="openai",
            requests=requests,
            metadata={"project": "test", "version": "1.0"}
        )
        assert batch.metadata == {"project": "test", "version": "1.0"}

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError."""
        requests = [
            UnifiedRequest(
                custom_id="req1",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hello")]
            )
        ]
        with pytest.raises(ValueError, match="provider must be one of"):
            UnifiedBatchMetadata(
                provider="invalid",
                requests=requests
            )

    def test_valid_providers(self):
        """Test all valid providers."""
        requests = [
            UnifiedRequest(
                custom_id="req1",
                model="model",
                messages=[UnifiedMessage.from_text("user", "Hello")]
            )
        ]
        for provider in ["openai", "anthropic", "google", "vllm"]:
            batch = UnifiedBatchMetadata(provider=provider, requests=requests)
            assert batch.provider == provider

    def test_empty_requests_raises_error(self):
        """Test that empty requests list raises ValueError."""
        with pytest.raises(ValueError, match="requests list cannot be empty"):
            UnifiedBatchMetadata(
                provider="openai",
                requests=[]
            )

    def test_duplicate_custom_ids_raises_error(self):
        """Test that duplicate custom_ids raise ValueError."""
        requests = [
            UnifiedRequest(
                custom_id="req1",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hello")]
            ),
            UnifiedRequest(
                custom_id="req1",  # Duplicate
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hi")]
            )
        ]
        with pytest.raises(ValueError, match="Duplicate custom_id"):
            UnifiedBatchMetadata(
                provider="openai",
                requests=requests
            )

    def test_unique_custom_ids(self):
        """Test batch with unique custom_ids."""
        requests = [
            UnifiedRequest(
                custom_id="req1",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hello")]
            ),
            UnifiedRequest(
                custom_id="req2",
                model="gpt-4o",
                messages=[UnifiedMessage.from_text("user", "Hi")]
            )
        ]
        batch = UnifiedBatchMetadata(
            provider="openai",
            requests=requests
        )
        assert len(batch.requests) == 2
