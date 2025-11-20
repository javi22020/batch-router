from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.message_param import MessageParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.image_block_param import ImageBlockParam
from anthropic.types.base64_image_source_param import Base64ImageSourceParam
from anthropic.types.messages.batch_create_params import BatchCreateParams
from anthropic.types.messages.message_batch_individual_response import MessageBatchIndividualResponse
from anthropic.types.text_block import TextBlock
from anthropic.types.thinking_block import ThinkingBlock

from typing import Any

from batch_router.core.output.message import OutputMessage
from batch_router.providers.base.batch_provider import BaseBatchProvider
from batch_router.core.base.provider import ProviderId
from batch_router.core.base.modality import Modality
from batch_router.core.input.request import InputRequest
from batch_router.core.output.request import OutputRequest
from batch_router.core.input.message import InputMessage
from batch_router.core.input.role import InputMessageRole
from batch_router.core.output.role import OutputMessageRole
from batch_router.core.base.request import InferenceParams
from batch_router.core.base.batch import BatchStatus
from batch_router.core.base.content import (
    MessageContent,
    TextContent,
    ThinkingContent
)
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.batch import OutputBatch

class AnthropicProvider(BaseBatchProvider):
    """
    A provider for Anthropic batch inference.

    This class implements the BaseBatchProvider interface to interact with the Anthropic API
    for sending and retrieving batch requests.

    Attributes:
        client (Anthropic): The Anthropic client instance.
    """
    def __init__(self, api_key: str) -> None:
        """
        Initializes the AnthropicProvider.

        Args:
            api_key (str): The API key for authenticating with the Anthropic API.
        """
        super().__init__(
            provider_id=ProviderId.ANTHROPIC
        )
        self.client = Anthropic(api_key=api_key)
    
    def input_message_role_to_provider(self, role: InputMessageRole) -> str:
        """
        Convert input message role to Anthropic role.

        Args:
            role (InputMessageRole): The unified input message role.

        Returns:
            str: The Anthropic-compatible role string ('user' or 'assistant').
        """
        if role == InputMessageRole.USER:
            return "user"
        elif role == InputMessageRole.ASSISTANT:
            return "assistant"
    
    def output_message_role_to_unified(self, role: str) -> OutputMessageRole:
        """
        Convert Anthropic output message role to unified role.

        Args:
            role (str): The Anthropic output role.

        Returns:
            OutputMessageRole: The corresponding unified role.

        Raises:
            ValueError: If the role is unknown.
        """
        if role == "assistant":
            return OutputMessageRole.ASSISTANT
        elif role == "tool":
            return OutputMessageRole.TOOL
        else:
            raise ValueError(f"Invalid output message role: {role}")
    
    def inference_params_to_provider(self, params: InferenceParams) -> dict[str, Any]:
        """
        Convert unified inference parameters to Anthropic parameters.

        Args:
            params (InferenceParams): The unified inference parameters.

        Returns:
            dict[str, Any]: A dictionary of parameters suitable for the Anthropic API.
        """
        provider_params = {
            "max_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            "system": params.system_prompt
        }
        if params.response_format is not None:
            schema = params.response_format.model_json_schema()
            provider_params["output_format"] = {
                "type": "json_schema",
                "schema": schema
            }
        provider_params = {k:v for k,v in provider_params.items() if v is not None}
        provider_params.update(params.additional_params)
        return provider_params
    
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> TextBlockParam | ImageBlockParam:
        """
        Convert unified input content to Anthropic block parameter.

        Args:
            content (MessageContent): The unified message content.

        Returns:
            TextBlockParam | ImageBlockParam: The content in Anthropic's format.

        Raises:
            ValueError: If the modality is not supported.
        """
        if content.modality == Modality.TEXT:
            return TextBlockParam(text=content.text)
        elif content.modality == Modality.IMAGE:
            return ImageBlockParam(
                source=Base64ImageSourceParam(
                    data=content.image_base64
                )
            )
        else:
            raise ValueError(f"Unsupported modality: {content.modality}")
        
    def convert_output_content_from_provider_to_unified(self, content: TextBlock | ThinkingBlock) -> MessageContent:
        """
        Convert Anthropic output content to unified message content.

        Args:
            content (TextBlock | ThinkingBlock): The content from Anthropic's response.

        Returns:
            MessageContent: The unified message content.
        """
        if content.type == "text":
            return TextContent(text=content.text)
        elif content.type == "thinking":
            return ThinkingContent(thinking=content.thinking)

    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> MessageParam:
        """
        Convert unified input message to Anthropic message parameter.

        Args:
            message (InputMessage): The unified input message.

        Returns:
            MessageParam: The message in Anthropic's format.
        """
        return MessageParam(
            content=[
                self.convert_input_content_from_unified_to_provider(content)
                for content in message.contents
            ],
            role=self.input_message_role_to_provider(message.role)
        )
    
    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> Request:
        """
        Convert unified input request to Anthropic batch request object.

        Args:
            request (InputRequest): The unified input request.

        Returns:
            Request: The request object for Anthropic's batch API.

        Raises:
            ValueError: If request params are missing.
        """
        if request.params is None:
            raise ValueError("Request params are required for Anthropic.")
        return Request(
            custom_id=request.custom_id,
            params=MessageCreateParamsNonStreaming(
                messages=[
                    self.convert_input_message_from_unified_to_provider(message)
                    for message in request.messages
                ],
                **self.inference_params_to_provider(request.params)
            )
        )
    
    def convert_output_request_from_provider_to_unified(self, request: MessageBatchIndividualResponse) -> OutputRequest:
        """
        Convert Anthropic batch response to unified output request.

        Args:
            request (MessageBatchIndividualResponse): The individual response from Anthropic's batch results.

        Returns:
            OutputRequest: The unified output request.

        Raises:
            ValueError: If the result type is unknown.
        """
        custom_id = request.custom_id
        if request.result.type == "canceled":
            current = OutputRequest(
                custom_id=custom_id,
                messages=[],
                success=False,
                error_message="This request was canceled."
            )
        elif request.result.type == "errored":
            current = OutputRequest(
                custom_id=custom_id,
                messages=[],
                success=False,
                error_message=request.result.error.error.message
            )
        elif request.result.type == "expired":
            current = OutputRequest(
                custom_id=custom_id,
                messages=[],
                success=False,
                error_message="This request expired."
            )
        elif request.result.type == "succeeded":
            current = OutputRequest(
                custom_id=custom_id,
                messages=[
                    OutputMessage(
                        role=self.output_message_role_to_unified(request.result.message.role),
                        contents=[
                            self.convert_output_content_from_provider_to_unified(content)
                            for content in request.result.message.content
                        ]
                    )
                ]
            )
        else:
            raise ValueError(f"Invalid output request result type: {request.result.type}")
        return current
    
    def convert_input_batch_from_unified_to_provider(self, batch: InputBatch) -> BatchCreateParams:
        """
        Convert unified input batch to Anthropic batch create parameters.

        Args:
            batch (InputBatch): The unified input batch.

        Returns:
            BatchCreateParams: The parameters to create a batch in Anthropic's API.
        """
        input_requests = batch.requests
        requests = [
            self.convert_input_request_from_unified_to_provider(request)
            for request in input_requests
        ]
        return BatchCreateParams(
            requests=requests
        )
    
    def convert_output_batch_from_provider_to_unified(self, batch: Any) -> OutputBatch:
        """
        Convert output batch from provider-specific format to unified format.

        Note: This method is not explicitly implemented in the original code but is required by the base class.
        The `get_results` method handles this conversion logic directly using `convert_output_request_from_provider_to_unified`.

        Args:
            batch (Any): The batch in the provider's format.

        Returns:
            OutputBatch: The unified output batch.
        """
        # Logic is handled in get_results, but for strict compliance we might implement it.
        # Since get_results is the entry point, this might just be a helper if needed.
        pass

    def send_batch(self, input_batch: InputBatch) -> str:
        """
        Send the input batch to Anthropic.

        Args:
            input_batch (InputBatch): The batch of requests to send.

        Returns:
            str: The ID of the created batch.
        """
        provider_batch = self.convert_input_batch_from_unified_to_provider(input_batch)
        batch = self.client.messages.batches.create(**provider_batch)
        return batch.id
    
    def poll_status(self, batch_id: str) -> BatchStatus:
        """
        Check the status of a submitted batch.

        Args:
            batch_id (str): The ID of the batch to check.

        Returns:
            BatchStatus: The current status of the batch.
        """
        batch = self.client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "in_progress":
            return BatchStatus.RUNNING
        elif batch.processing_status == "canceling":
            return BatchStatus.CANCELLED
        elif batch.processing_status == "ended":
            return BatchStatus.COMPLETED
        else:
            return BatchStatus.PENDING
    
    def get_results(self, batch_id: str) -> OutputBatch:
        """
        Retrieve and parse the results of a completed batch.

        Args:
            batch_id (str): The ID of the batch to retrieve results for.

        Returns:
            OutputBatch: The unified output batch containing the results.
        """
        output_requests = self.client.messages.batches.results(batch_id)
        batch_requests = [self.convert_output_request_from_provider_to_unified(request) for request in output_requests]
        return OutputBatch(
            requests=batch_requests
        )
    
    def count_input_request_tokens(self, request: InputRequest) -> int:
        """
        Count the tokens for a single input request.

        Args:
            request (InputRequest): The unified input request.

        Returns:
            int: The token count.
        """
        anthropic_request = self.convert_input_request_from_unified_to_provider(request)
        messages = anthropic_request["params"]["messages"]
        model = anthropic_request["params"]["model"]
        system = anthropic_request["params"]["system"]
        response = self.client.messages.count_tokens(
            messages=messages,
            model=model,
            system=system
        )
        return response.input_tokens

    def count_input_batch_tokens(self, batch: InputBatch) -> int:
        """
        Count the total tokens for an input batch.

        Args:
            batch (InputBatch): The unified input batch.

        Returns:
            int: The total token count.
        """
        input_requests = batch.requests
        total_tokens = 0
        for request in input_requests:
            total_tokens += self.count_input_request_tokens(request)
        
        return total_tokens

    def convert_output_message_from_provider_to_unified(self, message: Any) -> OutputMessage:
        """
        Convert output message from provider-specific format to unified format.

        Args:
            message (Any): The message in the provider's format.

        Returns:
            OutputMessage: The unified output message.
        """
        return OutputMessage(
            role=self.output_message_role_to_unified(message.role),
            contents=[
                self.convert_output_content_from_provider_to_unified(content)
                for content in message.content
            ]
        )

__all__ = ["AnthropicProvider"]
