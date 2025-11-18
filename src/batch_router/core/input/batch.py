from pydantic import BaseModel, Field
from batch_router.core.input.request import InputRequest, InputRequestConfig
from batch_router.core.base.batch import BatchConfig

class InputBatch(BaseModel):
    """An input batch (inputs of a batch inference)."""
    requests: list[InputRequest] = Field(description="The requests of the batch.", min_length=1)

    def with_config(self, config: BatchConfig) -> "InputBatch":
        """Configure the batch with a config. This will set the model and provider for all requests in the batch.
        Args:
            config: The BatchConfig to use for the batch.
        Returns:
            The InputBatch with the configured requests.
        """
        request_config = InputRequestConfig(
            model_id=config.model_id,
            provider_id=config.provider_id
        )
        request_params = config.params
        requests = [request.with_config(request_config).with_params(request_params) for request in self.requests]
        return InputBatch(
            requests=requests
        )
    
    def save_to_jsonl(self, file_path: str) -> None:
        text = ""
        for request in self.requests:
            text += request.model_dump_json(ensure_ascii=False) + "\n"
        text = text.strip() + "\n"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
