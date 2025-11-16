from pydantic import BaseModel
from batch_router.core.output.role import OutputMessageRole
from batch_router.core.base.content import MessageContent

class OutputMessage(BaseModel):
    role: OutputMessageRole
    contents: list[MessageContent]
