from pydantic import BaseModel, Field
from batch_router.core.input.request import InputRequest
from batch_router.core.base.request import InferenceParams

class InputBatch(BaseModel):
    """An input batch (inputs of a batch inference)."""
    requests: list[InputRequest] = Field(description="The requests of the batch.", min_length=1)

    def with_params(self, params: InferenceParams) -> "InputBatch":
        """Configure all requests in the batch with the same inference params.
        Args:
            params: The InferenceParams to use for all requests in the batch.
        Returns:
            The InputBatch with the configured inference params.
        """
        requests = [request.with_params(params) for request in self.requests]
        
        return self.model_copy(update={"requests": requests})
    
    def __str__(self) -> str:
        return f"InputBatch(requests={self.requests})"

    def __repr__(self) -> str:
        return self.__str__()
    
    def save_to_jsonl(self, file_path: str) -> None:
        text = ""
        for request in self.requests:
            text += request.model_dump_json(ensure_ascii=False, exclude_none=True) + "\n"
        text = text.strip() + "\n"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
