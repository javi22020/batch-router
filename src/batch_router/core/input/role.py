from enum import Enum

class InputMessageRole(Enum):
    """Role of a message in a batch input. Only ASSISTANT and TOOL are supported, as the USER messages are not in the input."""
    ASSISTANT = "assistant"
    TOOL = "tool"
