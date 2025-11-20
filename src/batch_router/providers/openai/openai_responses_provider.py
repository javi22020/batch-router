# TODO: Implement this provider

from openai import OpenAI
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_message_item import ResponseInputMessageItem
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from batch_router.providers.base.batch_provider import BaseBatchProvider
from batch_router.core.base.provider import ProviderId
from batch_router.core.input.message import InputMessage
from batch_router.core.input.role import InputMessageRole
from batch_router.core.base.request import InferenceParams
from batch_router.core.base.content import MessageContent
from batch_router.core.base.modality import Modality
from batch_router.core.output.role import OutputMessageRole

import os
from typing import Any

class OpenAIResponsesProvider(BaseBatchProvider):
    """
    A provider for OpenAI batch inference using the Responses API (not yet fully implemented).

    Attributes:
        client (OpenAI): The OpenAI client instance.
    """
    def __init__(self, api_key: str | None = None) -> None:
        """
        Initializes the OpenAIResponsesProvider.

        Args:
            api_key (str | None): The API key for authenticating with OpenAI.
                                  If not provided, it defaults to the OPENAI_API_KEY environment variable.
        """
        super().__init__(
            provider_id=ProviderId.OPENAI
        )
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def input_message_role_to_provider(self, role: InputMessageRole) -> str:
        """
        Convert unified input message role to OpenAI Responses role.

        Args:
            role (InputMessageRole): The unified input message role.

        Returns:
            str: The OpenAI role string ('user' or 'assistant').
        """
        if role == InputMessageRole.USER:
            return "user"
        elif role == InputMessageRole.ASSISTANT:
            return "assistant"
    
    def inference_params_to_provider(self, params: InferenceParams) -> dict[str, Any]:
        """
        Convert unified inference parameters to OpenAI Responses parameters.

        Args:
            params (InferenceParams): The unified inference parameters.

        Returns:
            dict[str, Any]: The configuration dictionary for OpenAI.
        """
        provider_params = {
            "max_output_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            "instructions": params.system_prompt
        }
        provider_params = {k:v for k,v in provider_params.items() if v is not None}
        provider_params.update(params.additional_params)
        return provider_params
    
    def output_message_role_to_unified(self, role: str) -> OutputMessageRole:
        """
        Convert OpenAI Responses output message role to unified role.

        Args:
            role (str): The OpenAI role string.

        Returns:
            OutputMessageRole: The unified output message role.

        Raises:
            ValueError: If the role is invalid.
        """
        if role == "assistant":
            return OutputMessageRole.ASSISTANT
        elif role == "tool":
            return OutputMessageRole.TOOL
        else:
            raise ValueError(f"Invalid output message role: {role}")
    
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> ResponseInputTextParam | ResponseInputImageParam:
        """
        Convert unified content to OpenAI Responses input content parameter.

        Args:
            content (MessageContent): The unified content.

        Returns:
            ResponseInputTextParam | ResponseInputImageParam: The OpenAI content part.

        Raises:
            ValueError: If the content modality is unsupported.
        """
        if content.modality == Modality.TEXT:
            return ResponseInputTextParam(text=content.text)
        elif content.modality == Modality.IMAGE:
            return ResponseInputImageParam(image_url=content.image_base64)
        elif content.modality == Modality.AUDIO:
            raise ValueError(f"Audio content is not supported for OpenAI Responses: {content.modality}")
        else:
            raise ValueError(f"Unsupported input content modality: {content.modality}")
        
    def convert_output_content_from_provider_to_unified(self, content: Any) -> MessageContent:
        """
        Convert OpenAI Responses output content to unified message content.

        Args:
            content (Any): The OpenAI content part.

        Returns:
            MessageContent: The unified message content.
        """
        return super().convert_output_content_from_provider_to_unified(content)
    
    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> Any:
        """
        Convert unified input message to OpenAI Responses input message item.

        Args:
            message (InputMessage): The unified input message.

        Returns:
            Any: The OpenAI response input message item.
        """
        return ResponseInputMessageItem(

        )
