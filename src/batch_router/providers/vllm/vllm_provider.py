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
from vllm import LLM
import os

logger = getLogger(__name__)

class vLLMProvider(BaseBatchProvider):
    """
    A provider for vLLM local batch inference.

    This provider manages local vLLM processes for batch inference.
    Note: Requires vLLM to be installed and accessible.

    Attributes:
        model_path (str): The path to the model to be used by vLLM.
        run_batch_kwargs (dict[str, Any] | None): Additional command-line arguments for the vLLM process.
    """
    _llm: LLM | None = None
    
    def __init__(self, model_path: str, run_batch_kwargs: dict[str, Any] | None = None) -> None:
        """
        Initialize the vLLMProvider.

        Args:
            model_path (str): The path to the vLLM model.
            run_batch_kwargs (dict[str, Any] | None): Additional kwargs to pass to the vLLM run-batch command.
                                                      Arguments -i, -o and --model are already handled by the provider.
        """
        super().__init__(
            provider_id=ProviderId.VLLM
        )
        self.model_path = model_path
        self.run_batch_kwargs = run_batch_kwargs
    
    def input_message_role_to_provider(self, role: InputMessageRole) -> str:
        """
        Convert unified input message role to vLLM role.

        Args:
            role (InputMessageRole): The unified input message role.

        Returns:
            str: The vLLM role string ('user' or 'assistant').
        """
        if role == InputMessageRole.USER:
            return "user"
        elif role == InputMessageRole.ASSISTANT:
            return "assistant"
    
    def inference_params_to_provider(self, params: InferenceParams) -> dict[str, Any]:
        """
        Convert unified inference parameters to vLLM parameters.

        Args:
            params (InferenceParams): The unified inference parameters.

        Returns:
            dict[str, Any]: The configuration dictionary for vLLM.
        """
        provider_params = {
            "max_completion_tokens": params.max_output_tokens,
            "temperature": params.temperature
        }
        if params.response_format is not None:
            json_schema = params.response_format.model_json_schema()
            provider_params["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        provider_params = {k:v for k,v in provider_params.items() if v is not None}
        provider_params.update(params.additional_params)
        return provider_params
    
    def output_message_role_to_unified(self, role: str) -> OutputMessageRole:
        """
        Convert vLLM output message role to unified role.

        Args:
            role (str): The vLLM role string.

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
    
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> dict[str, Any]:
        """
        Convert unified content to vLLM content part.

        Args:
            content (MessageContent): The unified content.

        Returns:
            dict[str, Any]: The vLLM content part.

        Raises:
            ValueError: If the content modality is unsupported.
        """
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
        """
        Convert vLLM output content to unified message content.

        Note: Not currently implemented as vLLM output structure handling is done via ChatCompletionsBatchOutputRequest directly in request conversion.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("vLLM does not support output content conversion.")
    
    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> dict[str, Any]:
        """
        Convert unified input message to vLLM message format.

        Args:
            message (InputMessage): The unified input message.

        Returns:
            dict[str, Any]: The vLLM message dictionary.
        """
        return {
            "role": self.input_message_role_to_provider(message.role),
            "content": [
                self.convert_input_content_from_unified_to_provider(content)
                for content in message.contents
            ]
        }
    
    def convert_output_message_from_provider_to_unified(self, message: Any) -> OutputMessage:
        """
        Convert vLLM output message to unified output message.

        Note: Not currently implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("vLLM does not support output message conversion.")

    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> dict[str, Any]:
        """
        Convert unified input request to vLLM request object.

        Args:
            request (InputRequest): The unified input request.

        Returns:
            dict[str, Any]: The vLLM request dictionary.

        Raises:
            ValueError: If request params are missing.
        """
        if request.params is None:
            raise ValueError("Request params are required for vLLM.")
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
                "model": request.params.model_id,
                "messages": messages,
                **self.inference_params_to_provider(request.params)
            }
        }
    
    def convert_output_request_from_provider_to_unified(self, request: ChatCompletionsBatchOutputRequest) -> OutputRequest:
        """
        Convert vLLM (OpenAI-compatible) batch output request to unified output request.

        Args:
            request (ChatCompletionsBatchOutputRequest): The batch output item.

        Returns:
            OutputRequest: The unified output request.
        """
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
        """
        Convert unified input batch to a JSONL file path for vLLM.

        This method writes the batch requests to a temporary JSONL file and returns the file path.

        Args:
            batch (InputBatch): The unified input batch.

        Returns:
            str: The path to the temporary JSONL file.
        """
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
        """
        Convert vLLM output file content to unified output batch.

        Args:
            batch (str): The content of the output JSONL file as a string.

        Returns:
            OutputBatch: The unified output batch.
        """
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
        """
        Count input request tokens. Not supported for vLLM.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("vLLM does not support input request token counting.")
    
    @override
    def count_input_batch_tokens(self, batch: InputBatch) -> int:
        """
        Count input batch tokens. Not supported for vLLM.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("vLLM does not support input batch token counting.")
    
    def vllm_run_batch(self, input_file_path: str, output_file_path: str) -> int:
        """
        Execute the vLLM run-batch command as a subprocess.

        Args:
            input_file_path (str): The path to the input JSONL file.
            output_file_path (str): The path where the output JSONL file should be written.

        Returns:
            int: The PID of the vLLM process.
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
        """
        Parse the vLLM batch ID to retrieve the PID and output file path.

        Args:
            batch_id (str): The batch ID string.

        Returns:
            tuple[int, str]: A tuple containing the PID and the output file path.

        Raises:
            ValueError: If the batch ID format is invalid.
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
        """
        Send the input batch to vLLM.

        This method creates an input file, starts the vLLM process, and constructs a batch ID.

        Args:
            input_batch (InputBatch): The unified input batch.

        Returns:
            str: The constructed batch ID.
        """
        input_file_path = self.convert_input_batch_from_unified_to_provider(input_batch)
        logger.info(f"Converted vLLM input batch to file path: {input_file_path}")
        output_file_path = f"temp_vllm_output_{dt.now().strftime('%Y%m%d%H%M%S')}.jsonl"
        pid = self.vllm_run_batch(input_file_path, output_file_path)
        batch_id = f"vllm_pid_{pid}_path_{output_file_path}"
        logger.info(f"Created vLLM batch with ID: {batch_id}")
        return batch_id
    
    def poll_status(self, batch_id: str) -> BatchStatus:
        """
        Check the status of a running vLLM batch process.

        Args:
            batch_id (str): The batch ID.

        Returns:
            BatchStatus: The current status of the batch.
        """
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
        """
        Retrieve the results of a completed vLLM batch.

        Waits for the output file to exist before reading it.

        Args:
            batch_id (str): The batch ID.

        Returns:
            OutputBatch: The unified output batch.

        Raises:
            TimeoutError: If the output file is not found within the timeout period.
        """
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
