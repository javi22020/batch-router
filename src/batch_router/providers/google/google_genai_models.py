from pydantic import BaseModel
from google.genai import types

class GoogleGenAIRequestBody(BaseModel):
    contents: list[types.Content]
    generation_config: types.GenerateContentConfig

class GoogleGenAIRequest(BaseModel):
    key: str
    request: GoogleGenAIRequestBody
    system_instruction: str | None
