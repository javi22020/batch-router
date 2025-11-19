from pydantic import BaseModel
from google.genai import types
from typing import Any

class GoogleGenAIInputRequestBody(BaseModel):
    contents: list[types.Content]

class GoogleGenAIInputRequest(BaseModel):
    key: str
    request: GoogleGenAIInputRequestBody
    config: dict[str, Any]

class GoogleGenAIOutputRequest(BaseModel):
    response: types.GenerateContentResponse
    config: types.GenerateContentConfig
    key: str
