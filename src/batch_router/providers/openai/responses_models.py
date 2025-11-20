# TODO: Implement these models

from pydantic import BaseModel
from openai.types.responses.response import Response
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming
from batch_router.providers.openai.common_models import Error

class ResponsesBatchOutputRequest(BaseModel):
    """
    Represents a single item in the output JSONL file from an OpenAI Responses batch process.

    Note: This model is a placeholder and not fully implemented.

    Attributes:
        custom_id (str): The custom identifier provided in the input request.
        response (Response): The response data if successful.
        error (Error): The error data if the request failed. Defaults to None.
    """
    custom_id: str
    response: Response
    error: Error = None
