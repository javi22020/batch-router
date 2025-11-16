from pydantic import BaseModel, Field
from batch_router.core.output.request import OutputRequest
from batch_router.core.base.batch import BatchConfig

class OutputBatch(BaseModel):
    """An output batch (results of a batch inference)."""
    requests: list[OutputRequest] = Field(description="The requests of the batch.", min_length=1)
