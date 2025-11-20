from pydantic import BaseModel
from typing import Literal

class Error(BaseModel):
    """
    Represents an error returned by the OpenAI API in a batch context.

    Attributes:
        code (Literal["batch_expired", "batch_cancelled", "request_timeout"]): The error code indicating the type of failure.
        message (str): A descriptive message about the error.
    """
    code: Literal["batch_expired", "batch_cancelled", "request_timeout"]
    message: str
