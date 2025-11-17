from typing import Any
import json
import subprocess as sp
import psutil
import re
from batch_router.core.base.batch import BatchStatus
from batch_router.core.base.content import (
    MessageContent,
    TextContent,
    ImageContent,
    AudioContent
)
from datetime import datetime as dt
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
from batch_router.providers.openai.chat_completions_models import BatchOutputRequest
from logging import getLogger
import os

logger = getLogger(__name__)

class vLLMProvider(BaseBatchProvider):
    """A provider for vLLM local batch inference. You need to have vLLM installed."""
    def __init__(self, model_path: str) -> None:
        super().__init__(
            provider_id=ProviderId.VLLM
        )
        self.model_path = model_path
    
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
    
    def convert_output_content_from_provider_to_unified(self, content: Any) -> MessageContent:
        raise NotImplementedError("vLLM does not support output content conversion.")
    
    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> dict[str, Any]:
        return {
            "role": self.input_message_role_to_provider(message.role),
            "content": [
                self.convert_input_content_from_unified_to_provider(content)
                for content in message.contents
            ]
        }
    
    def convert_output_message_from_provider_to_unified(self, message: Any) -> OutputMessage:
        raise NotImplementedError("vLLM does not support output message conversion.")

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
    
    def convert_output_request_from_provider_to_unified(self, request: BatchOutputRequest) -> OutputRequest:
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
        input_requests = [
            self.convert_input_request_from_unified_to_provider(request)
            for request in batch.requests
        ]
        jsonl_content = ""
        for request in input_requests:
            line = json.dumps(request, ensure_ascii=False) + "\n"
            jsonl_content += line
        input_file_path = f"temp_vllm_input_{dt.now().strftime('%Y%m%d%H%M%S')}.jsonl"
        with open(input_file_path, "w", encoding="utf-8") as input_file:
            input_file.write(jsonl_content)
        return input_file_path
    
    def convert_output_batch_from_provider_to_unified(self, batch: str) -> OutputBatch:
        """vLLM returns a file object, this method takes the file content and converts it to a OutputBatch."""
        lines = [line.strip() for line in batch.splitlines() if line.strip()]
        responses = [BatchOutputRequest.model_validate_json(line, extra="ignore") for line in lines]
        output_batch = OutputBatch(
            requests=[
                self.convert_output_request_from_provider_to_unified(response)
                for response in responses
            ]
        )
        return output_batch
    
    def vllm_run_batch(self, input_file_path: str, output_file_path: str) -> int:
        command = [
            "vllm",
            "run-batch",
            "--model",
            self.model_path,
            "--input",
            input_file_path,
            "--output",
            output_file_path
        ]
        process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
        return process.pid
    
    def read_vllm_batch_id(self, batch_id: str) -> tuple[int, str]:
        pattern = r'vllm_pid_(\d+)_path_temp_(.+)'
        match = re.search(pattern, batch_id)
        if match:
            pid = match.group(1)
            output_file_path = match.group(2)
            return int(pid), output_file_path
        else:
            raise ValueError(f"Invalid vLLM batch_id: {batch_id}")

    def send_batch(self, input_batch: InputBatch) -> str:
        input_file_path = self.convert_input_batch_from_unified_to_provider(input_batch)
        logger.info(f"Converted vLLM input batch to file path: {input_file_path}")
        output_file_path = f"temp_vllm_output_{dt.now().strftime('%Y%m%d%H%M%S')}.jsonl"
        pid = self.vllm_run_batch(input_file_path, output_file_path)
        batch_id = f"vllm_pid_{pid}_path_{output_file_path}"
        logger.info(f"Created vLLM batch with ID: {batch_id}")
        return batch_id
    
    def poll_status(self, batch_id: str) -> BatchStatus:
        pid, _ = self.read_vllm_batch_id(batch_id)
        process = psutil.Process(pid)
        is_running = process.is_running()
        if is_running:
            return BatchStatus.RUNNING
        else:
            return BatchStatus.COMPLETED
    
    def get_results(self, batch_id: str) -> OutputBatch:
        _, output_file_path = self.read_vllm_batch_id(batch_id)
        with open(output_file_path, "r", encoding="utf-8") as output_file:
            output_file_text = output_file.read()
        output_batch = self.convert_output_batch_from_provider_to_unified(output_file_text)
        return output_batch
