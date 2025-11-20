from pydantic import BaseModel
from google.genai import types
from typing import Any

class GoogleGenAIInputRequestBody(BaseModel):
    """
    Represents the body of an input request for Google GenAI.

    Attributes:
        contents (list[types.Content]): A list of content items for the request.
    """
    contents: list[types.Content]

class GoogleGenAIInputRequest(BaseModel):
    """
    Represents a complete input request structure for Google GenAI.

    Attributes:
        key (str): The custom identifier for the request.
        request (GoogleGenAIInputRequestBody): The body of the request containing the content.
        config (dict[str, Any]): Configuration parameters for the request.
    """
    key: str
    request: GoogleGenAIInputRequestBody
    config: dict[str, Any]

class GoogleGenAIOutputRequest(BaseModel):
    """
    Represents an output response structure from Google GenAI.

    Attributes:
        response (types.GenerateContentResponse): The actual response object from GenAI.
        config (types.GenerateContentConfig): The configuration used for generation.
        key (str): The custom identifier corresponding to the input request.
    """
    response: types.GenerateContentResponse
    config: types.GenerateContentConfig
    key: str
