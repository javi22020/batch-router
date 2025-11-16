from enum import Enum

class Modality(Enum):
    """The modality of a content or provider."""
    TEXT = "text"
    THINKING = "thinking"
    IMAGE = "image"
    AUDIO = "audio"
