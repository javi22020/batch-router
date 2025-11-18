from pydantic import BaseModel, Field
from batch_router.core.output.request import OutputRequest

class OutputBatch(BaseModel):
    """An output batch (results of a batch inference)."""
    requests: list[OutputRequest] = Field(description="The requests of the batch.", min_length=1)

    def save_to_jsonl(self, file_path: str) -> None:
        text = ""
        for request in self.requests:
            text += request.model_dump_json(ensure_ascii=False) + "\n"
        text = text.strip() + "\n"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
