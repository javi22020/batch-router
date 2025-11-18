# TODO: Implement these models

from pydantic import BaseModel
from openai.types.responses.response import Response
from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming
from batch_router.providers.openai.common_models import Error

class ResponsesBatchOutputRequest(BaseModel):
    custom_id: str
    response: Response
    error: Error = None

