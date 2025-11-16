from pydantic import BaseModel
from typing import Literal
from batch_router.core.base.modality import Modality

class ThinkingContent(BaseModel):
    modality: Literal[Modality.THINKING]
    thinking: str

class TextContent(BaseModel):
    modality: Literal[Modality.TEXT]
    text: str

class ImageContent(BaseModel):
    modality: Literal[Modality.IMAGE]
    image_base64: str # base64-encoded image

class AudioContent(BaseModel):
    modality: Literal[Modality.AUDIO]
    audio_base64: str # base64-encoded audio

MessageContent = TextContent | ThinkingContent | ImageContent | AudioContent
