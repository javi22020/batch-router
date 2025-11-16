from enum import Enum

class InputMessageRole(Enum):
    """Role of a message in a batch input. SYSTEM role is not supported, it must be in the request level."""
    USER = "user"
    ASSISTANT = "assistant"
