from pydantic import BaseModel, Field
from batch_router.core.input.message import InputMessage
from batch_router.core.base.request import InferenceParams

class InputRequest(BaseModel):
    """An input request (individual input of a batch inference).
    Can be created without params, but will need to be set before sending the request using `with_params`.
    This is useful for using the same request or batch for multiple models or providers.
    """
    custom_id: str = Field(description="The custom ID of the request.")
    messages: list[InputMessage] = Field(description="The messages of the input request.")
    params: InferenceParams | None = Field(default=None, description="The params of the request. Can be created without it for generic requests, but will need to be set before sending the request to a specific model and provider.")

    def __str__(self) -> str:
        return f"InputRequest(custom_id={self.custom_id}, messages={self.messages}, params={self.params})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
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
            params=params
        )
