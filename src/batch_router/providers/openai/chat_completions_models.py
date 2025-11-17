from pydantic import BaseModel
from typing import Optional, List, Literal


class Message(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseBody(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class Response(BaseModel):
    status_code: int
    request_id: str
    body: ResponseBody


class Error(BaseModel):
    code: Literal["batch_expired", "batch_cancelled", "request_timeout"]
    message: str


class BatchOutputRequest(BaseModel):
    """A batch output request from OpenAI (each line in the JSONL file returned by the API).
    Attributes:
        id: The ID of the request.
        custom_id: The custom ID of the request.
        response: The response from the request.
        error: The error from the request.
    """
    id: str
    custom_id: str
    response: Response
    error: Optional[Error] = None
