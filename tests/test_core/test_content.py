"""Tests for content types."""

import pytest
from batch_router.core.content import TextContent, ImageContent, DocumentContent


class TestTextContent:
    """Tests for TextContent dataclass."""

    def test_text_content_creation(self):
        """Test creating TextContent."""
        content = TextContent(text="Hello, world!")
        assert content.type == "text"
        assert content.text == "Hello, world!"

    def test_text_content_defaults(self):
        """Test TextContent with default values."""
        content = TextContent()
        assert content.type == "text"
        assert content.text == ""

    def test_text_content_is_dataclass(self):
        """Test TextContent is a dataclass."""
        content = TextContent(text="test")
        assert hasattr(content, "__dataclass_fields__")


class TestImageContent:
    """Tests for ImageContent dataclass."""

    def test_image_content_creation(self):
        """Test creating ImageContent with base64."""
        content = ImageContent(
            source_type="base64",
            media_type="image/png",
            data="iVBORw0KGgo..."
        )
        assert content.type == "image"
        assert content.source_type == "base64"
        assert content.media_type == "image/png"
        assert content.data == "iVBORw0KGgo..."

    def test_image_content_url(self):
        """Test creating ImageContent with URL."""
        content = ImageContent(
            source_type="url",
            media_type="image/jpeg",
            data="https://example.com/image.jpg"
        )
        assert content.source_type == "url"
        assert content.data == "https://example.com/image.jpg"

    def test_image_content_file_uri(self):
        """Test creating ImageContent with file URI."""
        content = ImageContent(
            source_type="file_uri",
            media_type="image/png",
            data="gs://bucket/image.png"
        )
        assert content.source_type == "file_uri"
        assert content.data == "gs://bucket/image.png"

    def test_image_content_defaults(self):
        """Test ImageContent with default values."""
        content = ImageContent()
        assert content.type == "image"
        assert content.source_type == "base64"
        assert content.media_type == "image/jpeg"
        assert content.data == ""


class TestDocumentContent:
    """Tests for DocumentContent dataclass."""

    def test_document_content_creation(self):
        """Test creating DocumentContent."""
        content = DocumentContent(
            source_type="base64",
            media_type="application/pdf",
            data="JVBERi0xLjQK..."
        )
        assert content.type == "document"
        assert content.source_type == "base64"
        assert content.media_type == "application/pdf"
        assert content.data == "JVBERi0xLjQK..."

    def test_document_content_url(self):
        """Test creating DocumentContent with URL."""
        content = DocumentContent(
            source_type="url",
            media_type="application/pdf",
            data="https://example.com/doc.pdf"
        )
        assert content.source_type == "url"
        assert content.data == "https://example.com/doc.pdf"

    def test_document_content_defaults(self):
        """Test DocumentContent with default values."""
        content = DocumentContent()
        assert content.type == "document"
        assert content.source_type == "base64"
        assert content.media_type == "application/pdf"
        assert content.data == ""
