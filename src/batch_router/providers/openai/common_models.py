from pydantic import BaseModel
from typing import Literal

class Error(BaseModel):
    code: Literal["batch_expired", "batch_cancelled", "request_timeout"]
    message: str
