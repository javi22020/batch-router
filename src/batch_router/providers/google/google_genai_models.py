from pydantic import BaseModel
from google.genai import types

class GoogleGenAIRequestBody(BaseModel):
    contents: list[types.Content]

class GoogleGenAIRequest(BaseModel):
    key: str
    request: GoogleGenAIRequestBody
    config: types.GenerateContentConfig
