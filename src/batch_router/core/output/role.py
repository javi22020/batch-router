from enum import Enum

class OutputMessageRole(str, Enum):
    """
    Enumeration representing the role of a message sender in an output batch.

    Note: Only ASSISTANT and TOOL roles are supported in the output, as USER messages are part of the input.

    Attributes:
        ASSISTANT (str): Represents a message from the assistant.
        TOOL (str): Represents a message from a tool.
    """
    ASSISTANT = "assistant"
    TOOL = "tool"
