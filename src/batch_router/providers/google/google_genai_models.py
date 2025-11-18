from pydantic import BaseModel
from google.genai import types

class GoogleGenAIInputRequestBody(BaseModel):
    contents: list[types.Content]

class GoogleGenAIInputRequest(BaseModel):
    key: str
    request: GoogleGenAIInputRequestBody
    config: types.GenerateContentConfig

class GoogleGenAIOutputRequest(BaseModel):
    response: types.GenerateContentResponse
    config: types.GenerateContentConfig
    key: str
