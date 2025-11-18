from pydantic import BaseModel, Field
from typing import Any
from batch_router.core.base.provider import ProviderId

class InferenceParams(BaseModel):
    model_id: str = Field(description="The model to use for the request.")
    provider_id: ProviderId = Field(description="The provider to use for the request.")
    system_prompt: str | None = Field(description="The system prompt to use for the inference.", default=None)
    max_output_tokens: int = Field(description="The maximum number of tokens to output.", default=1024)
    temperature: float | None = Field(description="The temperature to use for the inference.", default=None)
    response_format: BaseModel | None = Field(description="The response format to use for the inference; provider must support structured outputs.", default=None)
    additional_params: dict[str, Any] = Field(description="Additional parameters to use for the inference.", default_factory=dict)
