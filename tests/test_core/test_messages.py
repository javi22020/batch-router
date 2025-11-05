"""Tests for UnifiedMessage."""

import pytest
from batch_router.core.messages import UnifiedMessage
from batch_router.core.content import TextContent, ImageContent


class TestUnifiedMessage:
    """Tests for UnifiedMessage dataclass."""

    def test_message_creation(self):
        """Test creating a basic UnifiedMessage."""
        message = UnifiedMessage(
            role="user",
            content=[TextContent(text="Hello")]
        )
        assert message.role == "user"
        assert len(message.content) == 1
        assert message.content[0].text == "Hello"
        assert message.provider_kwargs == {}

    def test_message_with_provider_kwargs(self):
        """Test UnifiedMessage with provider kwargs."""
        message = UnifiedMessage(
            role="assistant",
            content=[TextContent(text="Response")],
            provider_kwargs={"cache_control": {"type": "ephemeral"}}
        )
        assert message.provider_kwargs == {"cache_control": {"type": "ephemeral"}}

    def test_from_text_constructor(self):
        """Test from_text convenience constructor."""
        message = UnifiedMessage.from_text("user", "Hello, world!")
        assert message.role == "user"
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextContent)
        assert message.content[0].text == "Hello, world!"

    def test_from_text_with_kwargs(self):
        """Test from_text with provider kwargs."""
        message = UnifiedMessage.from_text(
            "user",
            "Hello",
            cache_control={"type": "ephemeral"}
        )
        assert message.provider_kwargs == {"cache_control": {"type": "ephemeral"}}

    def test_invalid_role(self):
        """Test that invalid roles raise ValueError."""
        with pytest.raises(ValueError, match="Invalid role 'system'"):
            UnifiedMessage(
                role="system",  # type: ignore
                content=[TextContent(text="System prompt")]
            )

        with pytest.raises(ValueError, match="Invalid role"):
            UnifiedMessage(
                role="invalid",  # type: ignore
                content=[TextContent(text="Test")]
            )

    def test_empty_content_raises_error(self):
        """Test that empty content list raises ValueError."""
        with pytest.raises(ValueError, match="content list cannot be empty"):
            UnifiedMessage(role="user", content=[])

    def test_multimodal_content(self):
        """Test UnifiedMessage with multiple content items."""
        message = UnifiedMessage(
            role="user",
            content=[
                TextContent(text="What is in this image?"),
                ImageContent(
                    source_type="url",
                    media_type="image/jpeg",
                    data="https://example.com/image.jpg"
                )
            ]
        )
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextContent)
        assert isinstance(message.content[1], ImageContent)

    def test_to_dict(self):
        """Test converting UnifiedMessage to dictionary."""
        message = UnifiedMessage(
            role="user",
            content=[TextContent(text="Hello")],
            provider_kwargs={"test": "value"}
        )
        message_dict = message.to_dict()

        assert message_dict["role"] == "user"
        assert len(message_dict["content"]) == 1
        assert message_dict["content"][0]["type"] == "text"
        assert message_dict["content"][0]["text"] == "Hello"
        assert message_dict["provider_kwargs"] == {"test": "value"}

    def test_to_dict_multimodal(self):
        """Test to_dict with multimodal content."""
        message = UnifiedMessage(
            role="user",
            content=[
                TextContent(text="Describe this"),
                ImageContent(
                    source_type="url",
                    media_type="image/png",
                    data="https://example.com/img.png"
                )
            ]
        )
        message_dict = message.to_dict()

        assert len(message_dict["content"]) == 2
        assert message_dict["content"][0]["type"] == "text"
        assert message_dict["content"][1]["type"] == "image"
