from pydantic import BaseModel, Field
from batch_router.core.input.message import InputMessage
from batch_router.core.base.batch import InferenceParams
from batch_router.core.base.provider import ProviderId

class InputRequestConfig(BaseModel):
    """Config for an input request.
    Attributes:
        model_id: The model to use for the request. E.g. "gpt-5-mini".
        provider_id: The provider to use for the request. E.g. ProviderId.OPENAI.
    """
    model_id: str = Field(description="The model to use for the request.")
    provider_id: ProviderId = Field(description="The provider to use for the request.")

class InputRequest(BaseModel):
    """An input request (individual input of a batch inference).
    Can be created without a config, but will need to be set before sending the request using `with_config`.
    This is useful for using the same request or batch for multiple models or providers.
    """
    custom_id: str = Field(description="The custom ID of the request.")
    messages: list[InputMessage] = Field(description="The messages of the input request.")
    params: InferenceParams | None = Field(default=None, description="The params of the request. Can be created without it, but will need to be set before sending the request.")
    config: InputRequestConfig | None = Field(default=None, description="The config of the request. Can be created without it, but will need to be set before sending the request.")

    def with_config(self, config: InputRequestConfig) -> "InputRequest":
        return InputRequest(
            custom_id=self.custom_id,
            messages=self.messages,
            params=self.params,
            config=config
        )

    def with_params(self, params: InferenceParams) -> "InputRequest":
        """Set the params of the request.
        Args:
            params: The InferenceParams to use for the request.
        Returns:
            The InputRequest with the configured params.
        """
        return InputRequest(
            custom_id=self.custom_id,
            messages=self.messages,
            params=params,
            config=self.config
        )
