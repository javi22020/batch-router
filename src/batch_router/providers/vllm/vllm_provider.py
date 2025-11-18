import time
from typing import Any, override
import json
import subprocess as sp
import psutil
import re
from batch_router.core.base.batch import BatchStatus
from batch_router.core.base.content import (
    MessageContent,
    TextContent
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
from batch_router.providers.openai.chat_completions_models import ChatCompletionsBatchOutputRequest
from logging import getLogger
import os

logger = getLogger(__name__)

class vLLMProvider(BaseBatchProvider):
    """A provider for vLLM local batch inference. You need to have vLLM installed."""
    def __init__(self, model_path: str, run_batch_kwargs: dict[str, Any] | None = None) -> None:
        """Initialize the vLLMProvider.
        Args:
            model_path: The path to the vLLM model.
            run_batch_kwargs: Additional kwargs to pass to the vLLM run-batch command (-i, -o and --model are already set by the provider).
        """
        super().__init__(
            provider_id=ProviderId.VLLM
        )
        self.model_path = model_path
        self.run_batch_kwargs = run_batch_kwargs
    
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
            message = request.response.body.choices[0].message
            return OutputRequest(
                custom_id=custom_id,
                messages=[
                    OutputMessage(
                        role=self.output_message_role_to_unified(message.role),
                        contents=[
                            TextContent(text=message.content)
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
        responses = [ChatCompletionsBatchOutputRequest.model_validate_json(line, extra="ignore") for line in lines]
        output_batch = OutputBatch(
            requests=[
                self.convert_output_request_from_provider_to_unified(response)
                for response in responses
            ]
        )
        return output_batch
    
    def count_input_request_tokens(self, request: InputRequest) -> int:
        raise NotImplementedError("vLLM does not support input request token counting.")
    
    @override
    def count_input_batch_tokens(self, batch: InputBatch) -> int:
        raise NotImplementedError("vLLM does not support input batch token counting.")
    
    def vllm_run_batch(self, input_file_path: str, output_file_path: str) -> int:
        """Run the vLLM run-batch command.
        Args:
            input_file_path: The path to the input file.
            output_file_path: The path to the output file.
        Returns:
            The PID of the vLLM run-batch process.
        """
        command = [
            "vllm",
            "run-batch",
            "-i",
            input_file_path,
            "-o",
            output_file_path,
            "--model",
            self.model_path
        ]
        if self.run_batch_kwargs is not None:
            for key, value in self.run_batch_kwargs.items():
                if key not in ["-i", "-o", "--model"]:
                    command.extend([key, str(value)])
        process = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
        return process.pid
    
    def read_vllm_batch_id(self, batch_id: str) -> tuple[int, str]:
        """Read the vLLM batch ID and return the PID and output file path.
        Args:
            batch_id: The ID of the batch to read the ID of.
        Returns:
            The PID and output file path of the batch in a tuple.
        """
        pattern = r'vllm_pid_(\d+)_path_(.+)'
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
        try:
            process = psutil.Process(pid)
            status = process.status()
            if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                return BatchStatus.COMPLETED
            if process.is_running():
                return BatchStatus.RUNNING
            else:
                return BatchStatus.COMPLETED
        except psutil.NoSuchProcess:
            logger.info(f"vLLM process {pid} no longer exists, marking as completed")
            return BatchStatus.COMPLETED
    
    def get_results(self, batch_id: str) -> OutputBatch:
        _, output_file_path = self.read_vllm_batch_id(batch_id)
        for i in range(100):
            if os.path.exists(output_file_path):
                with open(output_file_path, "r", encoding="utf-8") as output_file:
                    output_file_text = output_file.read()
                    break
            time.sleep(3)
        else:
            raise TimeoutError(f"vLLM output file {output_file_path} not found after 100 attempts, batch {batch_id} is still running")
        output_batch = self.convert_output_batch_from_provider_to_unified(output_file_text)
        
        return output_batch
