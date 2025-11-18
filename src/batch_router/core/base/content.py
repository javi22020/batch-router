from pydantic import BaseModel
from typing import Literal
from batch_router.core.base.modality import Modality
import base64

class ThinkingContent(BaseModel):
    modality: Literal[Modality.THINKING] = Modality.THINKING
    thinking: str

class TextContent(BaseModel):
    modality: Literal[Modality.TEXT] = Modality.TEXT
    text: str

class ImageContent(BaseModel):
    modality: Literal[Modality.IMAGE] = Modality.IMAGE
    image_base64: str # base64-encoded image

    @classmethod
    def from_file(cls, file_path: str) -> "ImageContent":
        """Create an ImageContent from a file.
        Args:
            file_path: The path to the image file.
        Returns:
            The ImageContent with the base64-encoded image.
        """
        with open(file_path, "rb") as file:
            image_base64 = base64.b64encode(file.read()).decode("utf-8")
        return cls(image_base64=image_base64)

class AudioContent(BaseModel):
    modality: Literal[Modality.AUDIO] = Modality.AUDIO
    audio_base64: str # base64-encoded audio

    @classmethod
    def from_file(cls, file_path: str) -> "AudioContent":
        """Create an AudioContent from a file.
        Args:
            file_path: The path to the audio file.
        Returns:
            The AudioContent with the base64-encoded audio.
        """
        with open(file_path, "rb") as file:
            audio_base64 = base64.b64encode(file.read()).decode("utf-8")
        return cls(audio_base64=audio_base64)

MessageContent = TextContent | ThinkingContent | ImageContent | AudioContent
