from batch_router.providers.base.stream_provider import BaseStreamProvider
from batch_router.core.base.provider import ProviderId, ProviderMode
from batch_router.core.base.modality import Modality
from batch_router.core.base.content import MessageContent
from batch_router.core.input.message import InputMessage
from batch_router.core.output.message import OutputMessage
from batch_router.core.input.role import InputMessageRole
from batch_router.core.output.role import OutputMessageRole
from batch_router.core.input.request import InputRequest
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.request import OutputRequest
from batch_router.core.output.batch import OutputBatch
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage
from typing import Any

def convert_input_message_role_to_provider_role(role: InputMessageRole) -> str:
    if role == InputMessageRole.USER:
        return "user"
    elif role == InputMessageRole.ASSISTANT:
        return "assistant"

def convert_output_message_role_to_unified_role(role: str) -> OutputMessageRole:
    if role == "assistant":
        return OutputMessageRole.ASSISTANT
    elif role == "tool":
        return OutputMessageRole.TOOL
    else:
        raise ValueError(f"Invalid output message role: {role}")

class vLLMProvider(BaseStreamProvider):
    """A provider for vLLM local inference. To use this provider, you need to have a vLLM server running."""
    def __init__(
        self,
        server_url: str
    ) -> None:
        super().__init__(
            provider_id=ProviderId.VLLM,
            mode=ProviderMode.STREAM,
            modalities=[Modality.TEXT, Modality.IMAGE, Modality.AUDIO]
        )
        self.server_url = server_url
        self.client = OpenAI(api_key="EMPTY", base_url=server_url)
    
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> dict[str, str | dict[str, str]]:
        if content.modality == Modality.TEXT:
            return {
                "type": "text",
                "text": content.text
            }
        elif content.modality == Modality.IMAGE:
            return {
                "type": "image_url",
                "image_url": {
                    "url": content.image_base64
                }
            }
        elif content.modality == Modality.AUDIO:
            return {
                "type": "input_audio",
                "input_audio": content.audio_base64
            }
    
    def convert_output_content_from_provider_to_unified(self, content: ChatCompletionContentPartParam) -> MessageContent:
        raise NotImplementedError("Not implemented")

    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> Any:
        return ChatCompletionMessageParam(
            role=convert_input_message_role_to_provider_role(message.role),
            content=[self.convert_input_content_from_unified_to_provider(content) for content in message.contents]
        )
    
    def convert_output_message_from_provider_to_unified(self, message: ChatCompletionMessage) -> OutputMessage:
        raise NotImplementedError("Not implemented")
    
    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> Any:
        return
    
    def convert_input_batch_from_unified_to_provider(self, batch: InputBatch) -> Any:
        return super().convert_input_batch_from_unified_to_provider(batch)
