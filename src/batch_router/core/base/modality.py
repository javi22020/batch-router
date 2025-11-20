from enum import Enum

class Modality(Enum):
    """
    Enumeration representing the supported modalities for content or provider capabilities.

    Attributes:
        TEXT (str): Represents text modality.
        THINKING (str): Represents thinking or reasoning modality.
        IMAGE (str): Represents image modality.
        AUDIO (str): Represents audio modality.
    """
    TEXT = "text"
    THINKING = "thinking"
    IMAGE = "image"
    AUDIO = "audio"
