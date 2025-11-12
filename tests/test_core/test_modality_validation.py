"""Test modality validation in providers."""

import pytest
from batch_router.core.content import TextContent, AudioContent, ImageContent
from batch_router.core.messages import UnifiedMessage
from batch_router.core.requests import UnifiedRequest
from batch_router.core.enums import Modality
from batch_router.exceptions import UnsupportedModalityError


# Mock provider for testing
class MockProviderWithAudio:
    """Mock provider that supports audio."""
    name = "mock"
    supported_modalities = {Modality.TEXT, Modality.IMAGE, Modality.AUDIO}

    def validate_request_modalities(self, requests):
        """Import and use the BaseProvider's validation logic."""
        from batch_router.core.base import BaseProvider
        # Borrow the validation method
        BaseProvider.validate_request_modalities(self, requests)


class MockProviderWithoutAudio:
    """Mock provider that does NOT support audio."""
    name = "mock_no_audio"
    supported_modalities = {Modality.TEXT, Modality.IMAGE}

    def validate_request_modalities(self, requests):
        """Import and use the BaseProvider's validation logic."""
        from batch_router.core.base import BaseProvider
        # Borrow the validation method
        BaseProvider.validate_request_modalities(self, requests)


def test_provider_accepts_text_content():
    """Test that provider accepts text content."""
    provider = MockProviderWithoutAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[TextContent(text="Hello")]
                )
            ]
        )
    ]
    # Should not raise
    provider.validate_request_modalities(requests)


def test_provider_accepts_image_content():
    """Test that provider accepts image content."""
    provider = MockProviderWithoutAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        ImageContent(
                            source_type="url",
                            media_type="image/jpeg",
                            data="https://example.com/image.jpg"
                        )
                    ]
                )
            ]
        )
    ]
    # Should not raise
    provider.validate_request_modalities(requests)


def test_provider_rejects_audio_when_not_supported():
    """Test that provider rejects audio content when not supported."""
    provider = MockProviderWithoutAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        TextContent(text="Transcribe this"),
                        AudioContent(
                            type="audio",
                            source_type="base64",
                            media_type="audio/wav",
                            data="UklGRiQAAABXQVZF"
                        )
                    ]
                )
            ]
        )
    ]
    
    with pytest.raises(UnsupportedModalityError, match="does not support audio"):
        provider.validate_request_modalities(requests)


def test_provider_accepts_audio_when_supported():
    """Test that provider accepts audio content when supported."""
    provider = MockProviderWithAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        TextContent(text="Transcribe this"),
                        AudioContent(
                            type="audio",
                            source_type="base64",
                            media_type="audio/wav",
                            data="UklGRiQAAABXQVZF"
                        )
                    ]
                )
            ]
        )
    ]
    # Should not raise
    provider.validate_request_modalities(requests)


def test_validation_error_message_includes_provider_name():
    """Test that error message includes provider name."""
    provider = MockProviderWithoutAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        AudioContent(
                            source_type="base64",
                            media_type="audio/wav",
                            data="data"
                        )
                    ]
                )
            ]
        )
    ]
    
    with pytest.raises(UnsupportedModalityError) as exc_info:
        provider.validate_request_modalities(requests)
    
    assert "mock_no_audio" in str(exc_info.value)


def test_validation_error_message_lists_supported_modalities():
    """Test that error message lists supported modalities."""
    provider = MockProviderWithoutAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        AudioContent(
                            source_type="base64",
                            media_type="audio/wav",
                            data="data"
                        )
                    ]
                )
            ]
        )
    ]
    
    with pytest.raises(UnsupportedModalityError) as exc_info:
        provider.validate_request_modalities(requests)
    
    error_msg = str(exc_info.value)
    assert "text" in error_msg.lower()
    assert "image" in error_msg.lower()


def test_validation_checks_all_messages_in_request():
    """Test that validation checks all messages in a request."""
    provider = MockProviderWithoutAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[TextContent(text="First message")]
                ),
                UnifiedMessage(
                    role="assistant",
                    content=[TextContent(text="Response")]
                ),
                UnifiedMessage(
                    role="user",
                    content=[
                        AudioContent(
                            source_type="base64",
                            media_type="audio/wav",
                            data="data"
                        )
                    ]
                )
            ]
        )
    ]
    
    # Should catch audio in third message
    with pytest.raises(UnsupportedModalityError):
        provider.validate_request_modalities(requests)


def test_validation_checks_all_requests_in_batch():
    """Test that validation checks all requests in a batch."""
    provider = MockProviderWithoutAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[TextContent(text="Good request")]
                )
            ]
        ),
        UnifiedRequest(
            custom_id="test-2",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[TextContent(text="Another good request")]
                )
            ]
        ),
        UnifiedRequest(
            custom_id="test-3",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        AudioContent(
                            source_type="base64",
                            media_type="audio/wav",
                            data="data"
                        )
                    ]
                )
            ]
        )
    ]
    
    # Should catch audio in third request
    with pytest.raises(UnsupportedModalityError):
        provider.validate_request_modalities(requests)


def test_mixed_content_validation():
    """Test validation with mixed content types."""
    provider = MockProviderWithAudio()
    requests = [
        UnifiedRequest(
            custom_id="test-1",
            model="test-model",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[
                        TextContent(text="Analyze this"),
                        ImageContent(
                            source_type="url",
                            media_type="image/jpeg",
                            data="https://example.com/image.jpg"
                        ),
                        AudioContent(
                            source_type="base64",
                            media_type="audio/mp3",
                            data="SUQzBAAAAAAAI1RTU0UAAAA"
                        )
                    ]
                )
            ]
        )
    ]
    # Should not raise - all modalities supported
    provider.validate_request_modalities(requests)


