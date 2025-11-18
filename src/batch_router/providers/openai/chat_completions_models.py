from pydantic import BaseModel
from typing import Optional, List
from batch_router.providers.openai.common_models import Error


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


class ChatCompletionsResponseBody(BaseModel):
    id: str
    object: str # chat.completion
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class ChatCompletionsResponse(BaseModel):
    status_code: int
    request_id: str
    body: ChatCompletionsResponseBody


class ChatCompletionsBatchOutputRequest(BaseModel):
    """A batch output request from OpenAI (each line in the JSONL file returned by the API).
    Attributes:
        id: The ID of the request.
        custom_id: The custom ID of the request.
        response: The response from the request.
        error: The error from the request.
    """
    id: str
    custom_id: str
    response: ChatCompletionsResponse
    error: Optional[Error] = None
