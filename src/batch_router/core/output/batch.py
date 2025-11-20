from pydantic import BaseModel, Field
from batch_router.core.output.request import OutputRequest

class OutputBatch(BaseModel):
    """
    Represents a batch of output results from an inference process.

    Attributes:
        requests (list[OutputRequest]): A list of output requests containing the results. Must contain at least one request.
    """
    requests: list[OutputRequest] = Field(description="The requests of the batch.", min_length=1)

    def save_to_jsonl(self, file_path: str) -> None:
        """
        Saves the batch results to a JSONL (JSON Lines) file.

        Args:
            file_path (str): The path where the JSONL file should be saved.
        """
        text = ""
        for request in self.requests:
            text += request.model_dump_json(ensure_ascii=False) + "\n"
        text = text.strip() + "\n"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
