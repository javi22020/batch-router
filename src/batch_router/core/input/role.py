from enum import Enum

class InputMessageRole(str, Enum):
    """
    Enumeration representing the role of a message sender in an input batch.

    Note: The SYSTEM role is not supported at the message level; it must be defined at the request level.

    Attributes:
        USER (str): Represents a message from the user.
        ASSISTANT (str): Represents a message from the assistant.
    """
    USER = "user"
    ASSISTANT = "assistant"
