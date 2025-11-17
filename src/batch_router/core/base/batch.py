from enum import Enum
from pydantic import BaseModel, Field
from batch_router.core.base.request import InferenceParams
from batch_router.core.base.provider import ProviderId

class BatchStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class BatchConfig(BaseModel):
    """Config for a batch.
    Attributes:
        name: The name of the batch. E.g. "my_batch".
        provider_id: The provider to use for the batch. E.g. ProviderId.OPENAI.
        model_id: The model to use for the batch. E.g. "gpt-5-mini".
        params: The params to use for all requests in the batch.
    """
    name: str = Field(description="The name of the batch.")
    provider_id: ProviderId = Field(description="The provider to use for the batch.")
    model_id: str = Field(description="The model to use for the batch.")
    params: InferenceParams = Field(description="The params to use for all requests in the batch.")
