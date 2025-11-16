from pydantic import BaseModel, Field
from batch_router.core.output.message import OutputMessage

class OutputRequest(BaseModel):
    custom_id: str = Field(description="The custom ID of the request.")
    messages: list[OutputMessage] = Field(description="The messages of the output request.", min_length=1)
    success: bool = Field(default=True, description="Whether the request was successful.")
    error_message: str | None = Field(default=None, description="The error message of the request if it failed.")
