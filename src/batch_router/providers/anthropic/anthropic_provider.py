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
    """A provider for Anthropic batch inference. To use this provider, you need to have a Anthropic API key."""
    def __init__(self, api_key: str) -> None:
        super().__init__(
            provider_id=ProviderId.ANTHROPIC
        )
        self.client = Anthropic(api_key=api_key)
    
    def input_message_role_to_provider(self, role: InputMessageRole) -> str:
        if role == InputMessageRole.USER:
            return "user"
        elif role == InputMessageRole.ASSISTANT:
            return "assistant"
    
    def output_message_role_to_unified(self, role: str) -> OutputMessageRole:
        if role == "assistant":
            return OutputMessageRole.ASSISTANT
        elif role == "tool":
            return OutputMessageRole.TOOL
        else:
            raise ValueError(f"Invalid output message role: {role}")
    
    def inference_params_to_provider(self, params: InferenceParams) -> dict[str, Any]:
        provider_params = {
            "max_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            "system": params.system_prompt
        }
        provider_params = {k:v for k,v in provider_params.items() if v is not None}
        provider_params.update(params.additional_params)
        return provider_params
    
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> TextBlockParam | ImageBlockParam:
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
        if content.type == "text":
            return TextContent(text=content.text)
        elif content.type == "thinking":
            return ThinkingContent(thinking=content.thinking)

    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> MessageParam:
        return MessageParam(
            content=[
                self.convert_input_content_from_unified_to_provider(content)
                for content in message.contents
            ],
            role=self.input_message_role_to_provider(message.role)
        )
    
    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> Request:
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
        input_requests = batch.requests
        requests = [
            self.convert_input_request_from_unified_to_provider(request)
            for request in input_requests
        ]
        return BatchCreateParams(
            requests=requests
        )
    
    def send_batch(self, input_batch: InputBatch) -> str:
        provider_batch = self.convert_input_batch_from_unified_to_provider(input_batch)
        batch = self.client.messages.batches.create(**provider_batch)
        return batch.id
    
    def poll_status(self, batch_id: str) -> BatchStatus:
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
        output_requests = self.client.messages.batches.results(batch_id)
        batch_requests = [self.convert_output_request_from_provider_to_unified(request) for request in output_requests]
        return OutputBatch(
            requests=batch_requests
        )
    
    def count_input_batch_tokens(self, batch: InputBatch) -> int:
        input_requests = batch.requests
        total_tokens = 0
        for request in input_requests:
            anthropic_request = self.convert_input_request_from_unified_to_provider(request)
            messages = anthropic_request["params"]["messages"]
            model = anthropic_request["params"]["model"]
            system = anthropic_request["params"]["system"]
            response = self.client.messages.count_tokens(
                messages=messages,
                model=model,
                system=system
            )
            total_tokens += response.input_tokens
        
        return total_tokens

__all__ = ["AnthropicProvider"]
