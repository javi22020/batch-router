from pydantic import BaseModel, Field
from batch_router.core.input.message import InputMessage
from batch_router.core.base.request import InferenceParams

class InputRequest(BaseModel):
    """
    Represents an individual input request for batch inference.

    This class can be instantiated without inference parameters, allowing for generic requests
    that can be configured later for specific models or providers using the `with_params` method.

    Attributes:
        custom_id (str): A unique identifier for the request.
        messages (list[InputMessage]): A list of messages associated with the request.
        params (InferenceParams | None): Optional parameters for inference. Defaults to None.
    """
    custom_id: str = Field(description="The custom ID of the request.")
    messages: list[InputMessage] = Field(description="The messages of the input request.")
    params: InferenceParams | None = Field(default=None, description="The params of the request. Can be created without it for generic requests, but will need to be set before sending the request to a specific model and provider.")

    def __str__(self) -> str:
        """
        Returns a string representation of the InputRequest.

        Returns:
            str: A string description of the object's attributes.
        """
        return f"InputRequest(custom_id={self.custom_id}, messages={self.messages}, params={self.params})"
    
    def __repr__(self) -> str:
        """
        Returns a formal string representation of the InputRequest.

        Returns:
            str: A string representation suitable for debugging.
        """
        return self.__str__()
    
    def with_params(self, params: InferenceParams) -> "InputRequest":
        """
        Creates a new InputRequest with the specified inference parameters.

        Args:
            params (InferenceParams): The inference parameters to apply to the request.

        Returns:
            InputRequest: A new instance of InputRequest with the updated parameters.
        """
        return InputRequest(
            custom_id=self.custom_id,
            messages=self.messages,
            params=params
        )
