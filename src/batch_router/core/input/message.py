from pydantic import BaseModel
from batch_router.core.input.role import InputMessageRole
from batch_router.core.base.content import MessageContent

class InputMessage(BaseModel):
    """
    Represents a single message within an input request.

    Attributes:
        role (InputMessageRole): The role of the entity sending the message (e.g., user, assistant).
        contents (list[MessageContent]): A list of content items (text, images, etc.) contained in the message.
    """
    role: InputMessageRole
    contents: list[MessageContent]

    def __str__(self) -> str:
        """
        Returns a string representation of the InputMessage.

        Returns:
            str: A string description of the object's attributes.
        """
        return f"InputMessage(role={self.role}, contents={self.contents})"

    def __repr__(self) -> str:
        """
        Returns a formal string representation of the InputMessage.

        Returns:
            str: A string representation suitable for debugging.
        """
        return self.__str__()
