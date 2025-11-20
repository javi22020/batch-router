from pydantic import BaseModel, Field
from batch_router.core.output.message import OutputMessage

class OutputRequest(BaseModel):
    """
    Represents an individual output request containing the result of an inference.

    Attributes:
        custom_id (str): A unique identifier for the request, matching the input request.
        messages (list[OutputMessage]): A list of messages generated as output. Must contain at least one message.
        success (bool): Indicates whether the request was processed successfully. Defaults to True.
        error_message (str | None): An error message if the request failed. Defaults to None.
    """
    custom_id: str = Field(description="The custom ID of the request.")
    messages: list[OutputMessage] = Field(description="The messages of the output request.", min_length=1)
    success: bool = Field(default=True, description="Whether the request was successful.")
    error_message: str | None = Field(default=None, description="The error message of the request if it failed.")
