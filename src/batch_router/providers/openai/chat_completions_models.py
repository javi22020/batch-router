from pydantic import BaseModel
from typing import Optional, List
from batch_router.providers.openai.common_models import Error


class Message(BaseModel):
    """
    Represents a message in an OpenAI chat completion choice.

    Attributes:
        role (str): The role of the message sender (e.g., 'assistant').
        content (str): The text content of the message.
    """
    role: str
    content: str


class Choice(BaseModel):
    """
    Represents a choice in an OpenAI chat completion response.

    Attributes:
        index (int): The index of the choice.
        message (Message): The generated message.
        finish_reason (str): The reason the generation finished (e.g., 'stop', 'length').
    """
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """
    Represents token usage statistics for an OpenAI request.

    Attributes:
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        total_tokens (int): Total number of tokens used.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponseBody(BaseModel):
    """
    Represents the body of a successful OpenAI chat completion response.

    Attributes:
        id (str): The unique identifier of the response.
        object (str): The object type, typically 'chat.completion'.
        created (int): The timestamp of creation.
        model (str): The model used for generation.
        choices (List[Choice]): The list of generated choices.
        usage (Usage): The token usage statistics.
        system_fingerprint (Optional[str]): A unique fingerprint for the system configuration. Defaults to None.
    """
    id: str
    object: str # chat.completion
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class ChatCompletionsResponse(BaseModel):
    """
    Represents the full response wrapper from an OpenAI batch request.

    Attributes:
        status_code (int): The HTTP status code of the response.
        request_id (str): The request identifier.
        body (ChatCompletionsResponseBody): The actual response body containing the completion.
    """
    status_code: int
    request_id: str
    body: ChatCompletionsResponseBody


class ChatCompletionsBatchOutputRequest(BaseModel):
    """
    Represents a single item in the output JSONL file from an OpenAI batch process.

    Attributes:
        id (str): The unique identifier of the batch item.
        custom_id (str): The custom identifier provided in the input request.
        response (ChatCompletionsResponse): The response data if successful.
        error (Optional[Error]): The error data if the request failed. Defaults to None.
    """
    id: str
    custom_id: str
    response: ChatCompletionsResponse
    error: Optional[Error] = None
