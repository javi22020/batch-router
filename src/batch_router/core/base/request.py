from pydantic import BaseModel, Field
from typing import Any
from batch_router.core.base.provider import ProviderId

class InferenceParams(BaseModel):
    """
    Parameters for configuring an inference request.

    Attributes:
        model_id (str): The identifier of the model to use for the request.
        provider_id (ProviderId): The provider to be used for the request.
        system_prompt (str | None): Optional system prompt to guide the model's behavior. Defaults to None.
        max_output_tokens (int): The maximum number of tokens the model should generate. Defaults to 1024.
        temperature (float | None): The sampling temperature to control randomness. Defaults to None.
        response_format (BaseModel | None): The expected structure for the response, used if the provider supports structured outputs. Defaults to None.
        additional_params (dict[str, Any]): Any additional provider-specific parameters. Defaults to an empty dictionary.
    """
    model_id: str = Field(description="The model to use for the request.")
    provider_id: ProviderId = Field(description="The provider to use for the request.")
    system_prompt: str | None = Field(description="The system prompt to use for the inference.", default=None)
    max_output_tokens: int = Field(description="The maximum number of tokens to output.", default=1024)
    temperature: float | None = Field(description="The temperature to use for the inference.", default=None)
    response_format: BaseModel | None = Field(description="The response format to use for the inference; provider must support structured outputs.", default=None)
    additional_params: dict[str, Any] = Field(description="Additional parameters to use for the inference.", default_factory=dict)

    def __str__(self) -> str:
        """
        Returns a string representation of the InferenceParams object.

        Returns:
            str: A string description of the object's attributes.
        """
        return f"InferenceParams(model_id={self.model_id}, provider_id={self.provider_id}, system_prompt={self.system_prompt}, max_output_tokens={self.max_output_tokens}, temperature={self.temperature}, response_format={self.response_format}, additional_params={self.additional_params})"
    
    def __repr__(self) -> str:
        """
        Returns a formal string representation of the InferenceParams object.

        Returns:
            str: A string representation suitable for debugging.
        """
        return self.__str__()
