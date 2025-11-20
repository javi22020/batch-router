from pydantic import BaseModel
from typing import Literal
from batch_router.core.base.modality import Modality
import base64

class ThinkingContent(BaseModel):
    """
    Represents thinking content within a message.

    Attributes:
        modality (Literal[Modality.THINKING]): The modality type, always 'thinking'.
        thinking (str): The content of the thinking process.
    """
    modality: Literal[Modality.THINKING] = Modality.THINKING
    thinking: str

class TextContent(BaseModel):
    """
    Represents text content within a message.

    Attributes:
        modality (Literal[Modality.TEXT]): The modality type, always 'text'.
        text (str): The actual text content.
    """
    modality: Literal[Modality.TEXT] = Modality.TEXT
    text: str

class ImageContent(BaseModel):
    """
    Represents image content within a message.

    Attributes:
        modality (Literal[Modality.IMAGE]): The modality type, always 'image'.
        image_base64 (str): The base64-encoded string of the image.
    """
    modality: Literal[Modality.IMAGE] = Modality.IMAGE
    image_base64: str # base64-encoded image

    @classmethod
    def from_file(cls, file_path: str) -> "ImageContent":
        """
        Create an ImageContent object from a local image file.

        Args:
            file_path (str): The path to the image file.

        Returns:
            ImageContent: An instance containing the base64-encoded image.
        """
        with open(file_path, "rb") as file:
            image_base64 = base64.b64encode(file.read()).decode("utf-8")
        return cls(image_base64=image_base64)

class AudioContent(BaseModel):
    """
    Represents audio content within a message.

    Attributes:
        modality (Literal[Modality.AUDIO]): The modality type, always 'audio'.
        audio_base64 (str): The base64-encoded string of the audio.
    """
    modality: Literal[Modality.AUDIO] = Modality.AUDIO
    audio_base64: str # base64-encoded audio

    @classmethod
    def from_file(cls, file_path: str) -> "AudioContent":
        """
        Create an AudioContent object from a local audio file.

        Args:
            file_path (str): The path to the audio file.

        Returns:
            AudioContent: An instance containing the base64-encoded audio.
        """
        with open(file_path, "rb") as file:
            audio_base64 = base64.b64encode(file.read()).decode("utf-8")
        return cls(audio_base64=audio_base64)

MessageContent = TextContent | ThinkingContent | ImageContent | AudioContent
