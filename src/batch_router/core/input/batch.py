from pydantic import BaseModel, Field
from batch_router.core.input.request import InputRequest
from batch_router.core.base.request import InferenceParams

class InputBatch(BaseModel):
    """
    Represents a batch of input requests for inference.

    Attributes:
        requests (list[InputRequest]): A list of input requests to be processed in the batch. Must contain at least one request.
    """
    requests: list[InputRequest] = Field(description="The requests of the batch.", min_length=1)

    def with_params(self, params: InferenceParams) -> "InputBatch":
        """
        Configure all requests in the batch with the same inference parameters.

        Args:
            params (InferenceParams): The inference parameters to apply to all requests in the batch.

        Returns:
            InputBatch: A new InputBatch instance with the configured parameters applied to all requests.
        """
        requests = [request.with_params(params) for request in self.requests]
        
        return self.model_copy(update={"requests": requests})
    
    def __str__(self) -> str:
        """
        Returns a string representation of the InputBatch.

        Returns:
            str: A string description of the object's attributes.
        """
        return f"InputBatch(requests={self.requests})"

    def __repr__(self) -> str:
        """
        Returns a formal string representation of the InputBatch.

        Returns:
            str: A string representation suitable for debugging.
        """
        return self.__str__()
    
    def save_to_jsonl(self, file_path: str) -> None:
        """
        Saves the batch requests to a JSONL (JSON Lines) file.

        Args:
            file_path (str): The path where the JSONL file should be saved.
        """
        text = ""
        for request in self.requests:
            text += request.model_dump_json(ensure_ascii=False, exclude_none=True) + "\n"
        text = text.strip() + "\n"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
