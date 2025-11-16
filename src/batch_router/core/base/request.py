from pydantic import BaseModel, Field
from typing import Any

class InferenceParams(BaseModel):
    system_prompt: str | None = Field(description="The system prompt to use for the inference.", default=None)
    max_output_tokens: int = Field(description="The maximum number of tokens to output.", default=1024)
    temperature: float | None = Field(description="The temperature to use for the inference.", default=None)
    additional_params: dict[str, Any] = Field(description="Additional parameters to use for the inference.", default_factory=dict)
