from pydantic import BaseModel, Field
from batch_router.core.input.request import InputRequest
from batch_router.core.base.batch import BatchConfig

class InputBatch(BaseModel):
    """An input batch (inputs of a batch inference)."""
    requests: list[InputRequest] = Field(description="The requests of the batch.", min_length=1)
    config: BatchConfig = Field(description="The config of the batch.")
