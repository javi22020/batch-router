from pydantic import BaseModel, Field
from batch_router.core.input.message import InputMessage

class InputRequest(BaseModel):
    custom_id: str = Field(description="The custom ID of the request.")
    messages: list[InputMessage] = Field(description="The messages of the input request.")
    success: bool = Field(description="Whether the request was successful.")
    error_message: str | None = Field(description="The error message of the request if it failed.")
