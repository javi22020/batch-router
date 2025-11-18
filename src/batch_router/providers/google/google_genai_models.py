from pydantic import BaseModel
from google.genai import types

request = {
    "key": "request-1",
    "request": {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Describe the process of photosynthesis."
                    }
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.7
        }
    }
}

class GoogleGenAIRequestBody(BaseModel):
    contents: list[types.Content]
    generation_config: types.GenerateContentConfig

class GoogleGenAIRequest(BaseModel):
    key: str
    request: GoogleGenAIRequestBody
