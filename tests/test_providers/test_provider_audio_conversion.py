"""Test audio conversion methods in providers."""

import pytest
from batch_router.core.content import TextContent, AudioContent
from batch_router.core.messages import UnifiedMessage
from batch_router.core.requests import UnifiedRequest
from batch_router.providers.openai_provider import OpenAIProvider
from batch_router.providers.google_provider import GoogleProvider
from batch_router.providers.vllm_provider import VLLMProvider


class TestOpenAIAudioConversion:
    """Test OpenAI provider audio conversion."""

    def test_convert_audio_to_openai_wav(self):
        """Test converting WAV audio to OpenAI format."""
        provider = OpenAIProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="base64",
            media_type="audio/wav",
            data="UklGRiQAAABXQVZF"
        )
        
        result = provider._convert_audio_to_openai(audio)
        
        assert result["type"] == "input_audio"
        assert result["input_audio"]["data"] == "UklGRiQAAABXQVZF"
        assert result["input_audio"]["format"] == "wav"

    def test_convert_audio_to_openai_wave_mime(self):
        """Test converting audio/wave MIME type to OpenAI format."""
        provider = OpenAIProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="base64",
            media_type="audio/wave",
            data="UklGRiQAAABXQVZF"
        )
        
        result = provider._convert_audio_to_openai(audio)
        assert result["input_audio"]["format"] == "wav"

    def test_convert_audio_to_openai_mp3(self):
        """Test converting MP3 audio to OpenAI format."""
        provider = OpenAIProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="base64",
            media_type="audio/mp3",
            data="SUQzBAAAAAAAI1RTU0UAAAA"
        )
        
        result = provider._convert_audio_to_openai(audio)
        
        assert result["type"] == "input_audio"
        assert result["input_audio"]["data"] == "SUQzBAAAAAAAI1RTU0UAAAA"
        assert result["input_audio"]["format"] == "mp3"

    def test_convert_audio_to_openai_mpeg_mime(self):
        """Test converting audio/mpeg MIME type to OpenAI format."""
        provider = OpenAIProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="base64",
            media_type="audio/mpeg",
            data="SUQzBAAAAAAAI1RTU0UAAAA"
        )
        
        result = provider._convert_audio_to_openai(audio)
        assert result["input_audio"]["format"] == "mp3"

    def test_convert_audio_to_openai_url_source_fails(self):
        """Test that URL source is rejected for OpenAI."""
        provider = OpenAIProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="url",
            media_type="audio/mp3",
            data="https://example.com/audio.mp3"
        )
        
        with pytest.raises(ValueError, match="only supports base64-encoded audio"):
            provider._convert_audio_to_openai(audio)

    def test_convert_audio_to_openai_file_uri_fails(self):
        """Test that file_uri source is rejected for OpenAI."""
        provider = OpenAIProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="file_uri",
            media_type="audio/wav",
            data="gs://bucket/audio.wav"
        )
        
        with pytest.raises(ValueError, match="only supports base64-encoded audio"):
            provider._convert_audio_to_openai(audio)

    def test_convert_message_with_audio(self):
        """Test converting a message with audio content."""
        provider = OpenAIProvider(api_key="test-key")
        
        message = UnifiedMessage(
            role="user",
            content=[
                TextContent(text="Transcribe this"),
                AudioContent(
                    source_type="base64",
                    media_type="audio/wav",
                    data="UklGRiQAAABXQVZF"
                )
            ]
        )
        
        result = provider._convert_message_to_openai(message)
        
        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "input_audio"

    def test_convert_to_provider_format_adds_modalities(self):
        """Test that modalities field is added when audio is present."""
        provider = OpenAIProvider(api_key="test-key")
        
        requests = [
            UnifiedRequest(
                custom_id="test-1",
                model="gpt-4o-audio-preview",
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
        
        result = provider._convert_to_provider_format(requests)
        
        assert "modalities" in result[0]["body"]
        assert result[0]["body"]["modalities"] == ["text", "audio"]


class TestGoogleAudioConversion:
    """Test Google provider audio conversion."""

    def test_convert_audio_to_google_base64_wav(self):
        """Test converting base64 WAV audio to Google format."""
        provider = GoogleProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="base64",
            media_type="audio/wav",
            data="UklGRiQAAABXQVZF"
        )
        
        result = provider._convert_content_to_google_format(audio)
        
        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "audio/wav"
        assert result["inline_data"]["data"] == "UklGRiQAAABXQVZF"

    def test_convert_audio_to_google_base64_mp3(self):
        """Test converting base64 MP3 audio to Google format."""
        provider = GoogleProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="base64",
            media_type="audio/mp3",
            data="SUQzBAAAAAAAI1RTU0UAAAA"
        )
        
        result = provider._convert_content_to_google_format(audio)
        
        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "audio/mp3"

    def test_convert_audio_to_google_file_uri(self):
        """Test converting file URI audio to Google format."""
        provider = GoogleProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="file_uri",
            media_type="audio/wav",
            data="gs://bucket/audio.wav"
        )
        
        result = provider._convert_content_to_google_format(audio)
        
        assert "file_data" in result
        assert result["file_data"]["mime_type"] == "audio/wav"
        assert result["file_data"]["file_uri"] == "gs://bucket/audio.wav"

    def test_convert_audio_to_google_invalid_file_uri(self):
        """Test that non-gs:// file URIs are rejected."""
        provider = GoogleProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="file_uri",
            media_type="audio/wav",
            data="s3://bucket/audio.wav"
        )
        
        with pytest.raises(ValueError, match="must use gs:// URI format"):
            provider._convert_content_to_google_format(audio)

    def test_convert_audio_to_google_url_fails(self):
        """Test that URL source is rejected for Google."""
        provider = GoogleProvider(api_key="test-key")
        
        audio = AudioContent(
            source_type="url",
            media_type="audio/mp3",
            data="https://example.com/audio.mp3"
        )
        
        with pytest.raises(ValueError, match="does not support URL source_type"):
            provider._convert_content_to_google_format(audio)


class TestVLLMAudioConversion:
    """Test vLLM provider audio conversion."""

    def test_convert_audio_to_vllm_wav(self):
        """Test converting WAV audio to vLLM format."""
        provider = VLLMProvider()
        
        audio = AudioContent(
            source_type="base64",
            media_type="audio/wav",
            data="UklGRiQAAABXQVZF"
        )
        
        result = provider._convert_audio_to_vllm(audio)
        
        assert result["type"] == "input_audio"
        assert result["input_audio"]["data"] == "UklGRiQAAABXQVZF"
        assert result["input_audio"]["format"] == "wav"

    def test_convert_audio_to_vllm_mp3(self):
        """Test converting MP3 audio to vLLM format."""
        provider = VLLMProvider()
        
        audio = AudioContent(
            source_type="base64",
            media_type="audio/mp3",
            data="SUQzBAAAAAAAI1RTU0UAAAA"
        )
        
        result = provider._convert_audio_to_vllm(audio)
        
        assert result["type"] == "input_audio"
        assert result["input_audio"]["format"] == "mp3"

    def test_convert_audio_to_vllm_url_source_fails(self):
        """Test that URL source is rejected for vLLM."""
        provider = VLLMProvider()
        
        audio = AudioContent(
            source_type="url",
            media_type="audio/mp3",
            data="https://example.com/audio.mp3"
        )
        
        with pytest.raises(ValueError, match="only supports base64-encoded audio"):
            provider._convert_audio_to_vllm(audio)

    def test_convert_message_with_audio(self):
        """Test converting a message with audio content."""
        provider = VLLMProvider()
        
        message = UnifiedMessage(
            role="user",
            content=[
                TextContent(text="Transcribe this"),
                AudioContent(
                    source_type="base64",
                    media_type="audio/wav",
                    data="UklGRiQAAABXQVZF"
                )
            ]
        )
        
        result = provider._convert_message_to_vllm(message)
        
        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "input_audio"

    def test_convert_to_provider_format_adds_modalities(self):
        """Test that modalities field is added when audio is present."""
        provider = VLLMProvider()
        
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
        
        result = provider._convert_to_provider_format(requests)
        
        assert "modalities" in result[0]["body"]
        assert result[0]["body"]["modalities"] == ["text", "audio"]


