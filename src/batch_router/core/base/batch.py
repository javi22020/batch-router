from enum import Enum
from pydantic import BaseModel, Field
from typing import Any
from batch_router.core.base.provider import ProviderId

class BatchStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class InferenceParams(BaseModel):
    max_output_tokens: int = Field(description="The maximum number of tokens to output.", default=1024)
    temperature: float | None = Field(description="The temperature to use for the inference.", default=None)
    additional_params: dict[str, Any] = Field(description="Additional parameters to use for the inference.", default_factory=dict)

class BatchConfig(BaseModel):
    name: str = Field(description="The name of the batch.")
    provider_id: ProviderId = Field(description="The provider to use for the batch.")
    model_id: str = Field(description="The model to use for the batch.")
    params: InferenceParams = Field(description="The params to use for the batch.")
