from pydantic import BaseModel
from batch_router.core.output.role import OutputMessageRole
from batch_router.core.base.content import MessageContent

class OutputMessage(BaseModel):
    """
    Represents a single message within an output request.

    Attributes:
        role (OutputMessageRole): The role of the entity sending the message (e.g., assistant, tool).
        contents (list[MessageContent]): A list of content items (text, images, etc.) contained in the message.
    """
    role: OutputMessageRole
    contents: list[MessageContent]
