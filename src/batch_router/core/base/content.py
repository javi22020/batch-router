from pydantic import BaseModel
from typing import Literal
from batch_router.core.base.modality import Modality

class TextContent(BaseModel):
    modality: Literal[Modality.TEXT]
    content: str

class ImageContent(BaseModel):
    modality: Literal[Modality.IMAGE]
    content: str # base64-encoded image

class AudioContent(BaseModel):
    modality: Literal[Modality.AUDIO]
    content: str # base64-encoded audio

MessageContent = TextContent | ImageContent | AudioContent
