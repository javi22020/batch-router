from pydantic import BaseModel
from batch_router.core.input.role import InputMessageRole
from batch_router.core.base.content import MessageContent

class InputMessage(BaseModel):
    role: InputMessageRole
    contents: list[MessageContent]

    def __str__(self) -> str:
        return f"InputMessage(role={self.role}, contents={self.contents})"

    def __repr__(self) -> str:
        return self.__str__()
