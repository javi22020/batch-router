from google.genai import types, Client
from batch_router.core.base import BatchStatus, ImageContent, TextContent, ThinkingContent
from batch_router.core.base.provider import ProviderId
from batch_router.core.output.message import OutputMessage
from batch_router.core.output.batch import OutputBatch
from batch_router.providers.base.batch_provider import BaseBatchProvider
from batch_router.core.base.request import InferenceParams
from batch_router.core.base.content import MessageContent, Modality
from batch_router.core.input.message import InputMessage, InputMessageRole
from batch_router.core.input.request import InputRequest
from batch_router.core.output.request import OutputRequest
from batch_router.utilities.mime_type import get_mime_type
from batch_router.providers.google.google_genai_models import GoogleGenAIRequestBody, GoogleGenAIRequest
from batch_router.core.input.batch import InputBatch
from batch_router.providers.google.google_genai_models import GoogleGenAIRequest
import os
import base64
import tempfile


class GoogleGenAIProvider(BaseBatchProvider):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            provider_id=ProviderId.GOOGLE
        )
        self.client = Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))

    def inference_params_to_provider(self, params: InferenceParams) -> types.GenerateContentConfig:
        provider_params = {
            "system_instruction": params.system_prompt,
            "max_output_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            **params.additional_params
        }
        if params.response_format is not None:
            schema = params.response_format.model_json_schema()
            provider_params["response_mime_type"] = "application/json"
            provider_params["response_json_schema"] = schema

        return types.GenerateContentConfig.model_validate(provider_params, extra="ignore")
    
    def input_message_role_to_provider(self, role: InputMessageRole):
        if role == InputMessageRole.USER:
            return "user"
        elif role == InputMessageRole.ASSISTANT:
            return "model"
    
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> types.Part:
        if content.modality == Modality.TEXT:
            return types.Part.from_text(content.text)
        elif content.modality == Modality.IMAGE:
            data = base64.b64decode(content.image_base64)
            mime_type = get_mime_type(data)
            return types.Part.from_bytes(data, mime_type=mime_type)
        elif content.modality == Modality.AUDIO:
            data = base64.b64decode(content.audio_base64)
            mime_type = get_mime_type(data)
            return types.Part.from_bytes(data, mime_type=mime_type)
        else:
            raise ValueError(f"Unsupported input content modality: {content.modality}")
        
    def convert_output_content_from_provider_to_unified(self, content: types.Part) -> MessageContent:
        if content.text is not None:
            return TextContent(text=content.text)
        elif content.thought is not None:
            return ThinkingContent(thinking=content.thought)
        elif image := content.as_image():
            image_base64 = base64.b64encode(image.image_bytes).decode("utf-8")
            return ImageContent(image_base64=image_base64)
        else:
            raise ValueError(f"Unsupported output content part: {content.model_dump_json(ensure_ascii=False)}")
    
    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> types.Content:
        return types.Content(
            parts=[
                self.convert_input_content_from_unified_to_provider(content)
                for content in message.contents
            ],
            role=self.input_message_role_to_provider(message.role)
        )
    
    def convert_output_message_from_provider_to_unified(self, message: types.Content) -> OutputMessage:
        return OutputMessage(
            role=self.output_message_role_to_unified(message.role),
            contents=[
                self.convert_output_content_from_provider_to_unified(part)
                for part in message.parts
            ]
        )
    
    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> GoogleGenAIRequest:
        if request.params is None:
            raise ValueError("Request params are required for Google GenAI.")
        if request.config is None:
            raise ValueError("Request config is required for Google GenAI.")
        return GoogleGenAIRequest(
            key=request.custom_id,
            request=GoogleGenAIRequestBody(
                contents=[
                    self.convert_input_message_from_unified_to_provider(message)
                    for message in request.messages
                ],
                generation_config=self.inference_params_to_provider(request.params)
            )
        )

    def convert_output_request_from_provider_to_unified(self, request: types.GenerateContentResponse) -> OutputRequest:
        custom_id = request.response_id
        content = request.candidates[0].content
        if content is None:
            raise ValueError("Content was None.")
        return OutputRequest(
            custom_id=custom_id,
            messages=[
                self.convert_output_message_from_provider_to_unified(content)
            ]
        )

    
    def convert_input_batch_from_unified_to_provider(self, batch: InputBatch) -> str:
        """Google GenAI needs to upload a file to the API for batch inference, so this method returns the path of the created input file."""
        requests = [
            self.convert_input_request_from_unified_to_provider(request)
            for request in batch.requests
        ]
        jsonl_content = ""
        for request in requests:
            line = request.model_dump_json(ensure_ascii=False) + "\n"
            jsonl_content += line
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            prefix="temp_google_genai_input_",
            delete=False,
            delete_on_close=False
        ) as temp_file:
            temp_file.write(jsonl_content)
            file_path = temp_file.name
        
        return file_path

    def convert_output_batch_from_provider_to_unified(self, batch: str) -> OutputBatch:
        """Google GenAI returns a file object, this method takes the file content and converts it to a OutputBatch."""
        lines = [line.strip() for line in batch.splitlines() if line.strip()]
        responses = [types.GenerateContentResponse.model_validate_json(line, extra="ignore") for line in lines]
        output_batch = OutputBatch(
            requests=[
                self.convert_output_request_from_provider_to_unified(response)
                for response in responses
            ]
        )
    
    def count_input_request_tokens(self, request: InputRequest) -> int:
        if request.params is None:
            raise ValueError("Request params are required for Google GenAI.")
        if request.config is None:
            raise ValueError("Request config is required for Google GenAI.")
        model_id = request.config.model_id
        messages = request.messages
        response = self.client.models.count_tokens(
            model=model_id,
            contents=[
                self.convert_input_message_from_unified_to_provider(message)
                for message in messages
            ]
        )
        total_tokens = response.total_tokens
        if total_tokens is None:
            raise ValueError("Response total tokens is None.")
        return total_tokens
    
    def send_batch(self, input_batch: InputBatch) -> str:
        if input_batch.requests[0].config is None:
            raise ValueError("Request config is required for Google GenAI.")
        model_id = input_batch.requests[0].config.model_id
        input_file_path = self.convert_input_batch_from_unified_to_provider(input_batch)
        file = self.client.files.upload(
            file=input_file_path
        )
        file_name = file.name
        batch_job = self.client.batches.create(
            model=model_id,
            src=file_name
        )
        name = batch_job.name
        if name is None:
            raise ValueError("Batch job creation failed, name was None.")
        return name
    
    def poll_status(self, batch_id: str) -> BatchStatus:
        batch_job = self.client.batches.get(name=batch_id)
        status = batch_job.state
        if status is None:
            return BatchStatus.PENDING
        elif status.name in ["JOB_STATE_PENDING", "JOB_STATE_QUEUED"]:
            return BatchStatus.PENDING
        elif status.name == "JOB_STATE_RUNNING":
            return BatchStatus.RUNNING
        elif status.name == "JOB_STATE_CANCELLED":
            return BatchStatus.CANCELLED
        elif status.name == "JOB_STATE_FAILED":
            return BatchStatus.FAILED
        elif status.name == "JOB_STATE_SUCCEEDED":
            return BatchStatus.COMPLETED
        elif status.name == "JOB_STATE_EXPIRED":
            return BatchStatus.EXPIRED
        else:
            return BatchStatus.PENDING
    
    def get_results(self, batch_id: str) -> OutputBatch:
        batch_job = self.client.batches.get(name=batch_id)
        output = batch_job.dest
        if output is None:
            raise ValueError("Batch job output (BatchJob.dest) is None.")
        output_file_name = output.file_name
        data = self.client.files.download(name=output_file_name)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".jsonl",
            prefix="temp_google_genai_output_",
            delete=False,
            delete_on_close=False
        ) as temp_file:
            temp_file.write(data)
            file_path = temp_file.name
        output_batch = self.convert_output_batch_from_provider_to_unified(file_path)
        
        return output_batch
