from pydantic import BaseModel, Field
from batch_router.core.input.request import InputRequest, InputRequestConfig
from batch_router.core.base.batch import BatchConfig

class InputBatch(BaseModel):
    """An input batch (inputs of a batch inference)."""
    requests: list[InputRequest] = Field(description="The requests of the batch.", min_length=1)

    def with_config(self, config: BatchConfig) -> "InputBatch":
        request_config = InputRequestConfig(
            model_id=config.model_id,
            provider_id=config.provider_id
        )
        return InputBatch(
            requests=[request.with_config(request_config) for request in self.requests],
        )
