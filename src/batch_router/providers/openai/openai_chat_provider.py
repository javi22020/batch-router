from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
import tempfile
from typing import Any
import json
import requests
from tiktoken import encoding_for_model
import base64
from batch_router.core.base.batch import BatchStatus
from batch_router.core.base.content import (
    MessageContent,
    TextContent,
    ImageContent,
    AudioContent
)
from batch_router.core.base.modality import Modality
from batch_router.core.base.provider import ProviderId
from batch_router.core.input.role import InputMessageRole
from batch_router.core.output.role import OutputMessageRole
from batch_router.core.input.message import InputMessage
from batch_router.core.output.message import OutputMessage
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.batch import OutputBatch
from batch_router.core.input.request import InputRequest
from batch_router.core.output.request import OutputRequest
from batch_router.core.base.request import InferenceParams
from batch_router.providers.base.batch_provider import BaseBatchProvider
from batch_router.providers.openai.chat_completions_models import ChatCompletionsBatchOutputRequest
from logging import getLogger
import os

logger = getLogger(__name__)

class OpenAIChatProvider(BaseBatchProvider):
    """A provider for OpenAI batch inference. To use this provider, you need to have a OpenAI API key."""
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            provider_id=ProviderId.OPENAI
        )
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def input_message_role_to_provider(self, role: InputMessageRole) -> str:
        if role == InputMessageRole.USER:
            return "user"
        elif role == InputMessageRole.ASSISTANT:
            return "assistant"
    
    def inference_params_to_provider(self, params: InferenceParams) -> dict[str, Any]:
        provider_params = {
            "max_completion_tokens": params.max_output_tokens,
            "temperature": params.temperature
        }
        provider_params = {k:v for k,v in provider_params.items() if v is not None}
        provider_params.update(params.additional_params)
        return provider_params
    
    def output_message_role_to_unified(self, role: str) -> OutputMessageRole:
        if role == "assistant":
            return OutputMessageRole.ASSISTANT
        elif role == "tool":
            return OutputMessageRole.TOOL
        else:
            raise ValueError(f"Invalid output message role: {role}")
    
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> dict[str, Any]:
        if content.modality == Modality.TEXT:
            return {
                "type": "text",
                "text": content.text
            }
        elif content.modality == Modality.IMAGE:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{content.image_base64}"
                }
            }
        elif content.modality == Modality.AUDIO:
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": content.audio_base64
                }
            }
        else:
            raise ValueError(f"Unsupported input content modality: {content.modality}")
    
    def convert_output_content_from_provider_to_unified(self, content: ChatCompletionContentPartParam) -> MessageContent:
        if content["type"] == "text":
            text = content["text"]
            return TextContent(text=text)
        elif content["type"] == "image_url":
            url = content["image_url"]["url"]
            response = requests.get(url)
            image_base64 = base64.b64encode(response.content).decode("utf-8")
            return ImageContent(image_base64=image_base64)
        elif content["type"] == "input_audio":
            audio_base64 = content["input_audio"]["data"]
            return AudioContent(audio_base64=audio_base64)
        else:
            raise ValueError(f"Unsupported output content type: {content['type']}")
    
    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> dict[str, Any]:
        return {
            "role": self.input_message_role_to_provider(message.role),
            "content": [
                self.convert_input_content_from_unified_to_provider(content)
                for content in message.contents
            ]
        }
    
    def convert_output_message_from_provider_to_unified(self, message: ChatCompletionMessage) -> OutputMessage:
        return OutputMessage(
            role=self.output_message_role_to_unified(message.role),
            contents=[
                self.convert_output_content_from_provider_to_unified(content)
                for content in message.content
            ]
        )

    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> dict[str, Any]:
        messages = [
            self.convert_input_message_from_unified_to_provider(message)
            for message in request.messages
        ]
        if request.params.system_prompt is not None:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": request.params.system_prompt
                        }
                    ]
                }
            )
        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": request.config.model_id,
                "messages": messages,
                **self.inference_params_to_provider(request.params)
            }
        }
    
    def convert_output_request_from_provider_to_unified(self, request: ChatCompletionsBatchOutputRequest) -> OutputRequest:
        custom_id = request.custom_id
        if request.error is not None:
            error_template = "This request failed with the following error: {error.code} - {error.message}"
            error_message = error_template.format(error=request.error)
            return OutputRequest(
                custom_id=custom_id,
                messages=[],
                success=False,
                error_message=error_message
            )
        else:
            message: str = request.response.body.choices[0].message.content
            return OutputRequest(
                custom_id=custom_id,
                messages=[
                    OutputMessage(
                        role=OutputMessageRole.ASSISTANT,
                        contents=[
                            TextContent(text=message)
                        ]
                    )
                ]
            )
    
    def convert_input_batch_from_unified_to_provider(self, batch: InputBatch) -> str:
        """OpenAI needs to upload a file to the API for batch inference, so this method returns the path of the created input file."""
        input_requests = [
            self.convert_input_request_from_unified_to_provider(request)
            for request in batch.requests
        ]
        jsonl_content = ""
        for request in input_requests:
            line = json.dumps(request, ensure_ascii=False) + "\n"
            jsonl_content += line
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            prefix="temp_openai_input_",
            delete=False,
            delete_on_close=False
        ) as temp_file:
            temp_file.write(jsonl_content)
            file_path = temp_file.name
        return file_path
    
    def convert_output_batch_from_provider_to_unified(self, batch: str) -> OutputBatch:
        """OpenAI returns a file object, this method takes the file content and converts it to a OutputBatch."""
        lines = [line.strip() for line in batch.splitlines() if line.strip()]
        responses = [ChatCompletionsBatchOutputRequest.model_validate_json(line, extra="ignore") for line in lines]
        output_batch = OutputBatch(
            requests=[
                self.convert_output_request_from_provider_to_unified(response)
                for response in responses
            ]
        )
        return output_batch

    def count_input_request_tokens(self, request: InputRequest) -> int:
        encoding = encoding_for_model(request.config.model_id)
        total_tokens = 0
        messages = request.messages
        for message in messages:
            for content in message.contents:
                if content.modality == Modality.TEXT:
                    text = content.text
                    tokens = encoding.encode(text=text)
                    total_tokens += len(tokens)
                else:
                    logger.warning(f"Could not count tokens for content modality: {content.modality}")
        return total_tokens

    def send_batch(self, input_batch: InputBatch) -> str:
        input_file_path = self.convert_input_batch_from_unified_to_provider(input_batch)
        logger.info(f"Converted OpenAI input batch to file path: {input_file_path}")
        with open(input_file_path, "rb") as input_file:
            file_response = self.client.files.create(
                file=input_file,
                purpose="batch"
            )
        input_file_id = file_response.id
        logger.info(f"Uploaded OpenAI input file to ID: {input_file_id}")
        batch_response = self.client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=input_file_id
        )
        batch_id = batch_response.id
        logger.info(f"Created OpenAI batch with ID: {batch_id}")
        return batch_id
    
    def poll_status(self, batch_id: str) -> BatchStatus:
        batch = self.client.batches.retrieve(batch_id)
        if batch.status in ["validating"]:
            current_status = BatchStatus.PENDING
        elif batch.status in ["in_progress", "finalizing"]:
            current_status = BatchStatus.RUNNING
        elif batch.status in ["cancelling", "cancelled"]:
            current_status = BatchStatus.CANCELLED
        elif batch.status in ["failed"]:
            current_status = BatchStatus.FAILED
        elif batch.status in ["completed"]:
            current_status = BatchStatus.COMPLETED
        elif batch.status in ["expired"]:
            current_status = BatchStatus.EXPIRED
        else:
            raise ValueError(f"Invalid batch status: {batch.status}")
        logger.info(f"OpenAI batch status: {current_status.value}")
        return current_status
    
    def get_results(self, batch_id: str) -> OutputBatch:
        result_batch = self.client.batches.retrieve(batch_id)
        output_file_id = result_batch.output_file_id
        output_file = self.client.files.content(file_id=output_file_id)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".jsonl",
            prefix="temp_openai_output_",
            delete=False,
            delete_on_close=False
        ) as temp_file:
            temp_file.write(output_file.content)
            file_path = temp_file.name
        logger.info(f"OpenAI output file path: {file_path}")
        output_file_text = output_file.text
        output_batch = self.convert_output_batch_from_provider_to_unified(output_file_text)
        
        return output_batch
