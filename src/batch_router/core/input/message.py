from pydantic import BaseModel
from batch_router.core.input.role import InputMessageRole
from batch_router.core.base.content import MessageContent

class InputMessage(BaseModel):
    role: InputMessageRole
    contents: list[MessageContent]
