from google.genai import types, Client
from batch_router.core.base.provider import ProviderId
from typing import Any
from batch_router.providers.base.batch_provider import BaseBatchProvider
from batch_router.core.base.request import InferenceParams
import os

class GoogleGenAIProvider(BaseBatchProvider):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            provider_id=ProviderId.GOOGLE
        )
        self.client = Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))

    def inference_params_to_provider(self, params: InferenceParams) -> types.GenerateContentConfig:
        provider_params = {
            "system_instruction": params.system_prompt,
            "max_output_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            **params.additional_params
        }
        if params.response_format is not None:
            schema = params.response_format.model_json_schema()
            provider_params["response_mime_type"] = "application/json"
            provider_params["response_json_schema"] = schema

        return types.GenerateContentConfig.model_validate(provider_params, extra="ignore")
