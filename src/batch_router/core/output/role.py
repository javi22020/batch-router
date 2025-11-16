from enum import Enum

class OutputMessageRole(Enum):
    """Role of a message in a batch output. Only ASSISTANT and TOOL are supported, as the USER messages are not in the output."""
    ASSISTANT = "assistant"
    TOOL = "tool"
