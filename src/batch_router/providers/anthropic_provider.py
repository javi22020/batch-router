from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from batch_router.providers.base.batch_provider import BaseBatchProvider
from batch_router.core.base.provider import ProviderId, ProviderMode
from batch_router.core.base.modality import Modality
from batch_router.core.input.request import InputRequest
from batch_router.core.input.batch import InputBatch

class AnthropicProvider(BaseBatchProvider):
    """A provider for Anthropic batch inference. To use this provider, you need to have a Anthropic API key."""
    def __init__(self, api_key: str) -> None:
        super().__init__(
            provider_id=ProviderId.ANTHROPIC,
            mode=ProviderMode.BATCH,
            modalities=[Modality.TEXT, Modality.IMAGE]
        )
        self.client = Anthropic(api_key=api_key)
    
    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> Request:
        return Request(
            custom_id=request.custom_id,
            params=MessageCreateParamsNonStreaming(
                max_tokens
            )
        )
    
    def convert_input_batch_from_unified_to_provider(self, batch: InputBatch) -> Any:
        return super().convert_input_batch_from_unified_to_provider(batch)
    
    def send_batch(self, input_batch: InputBatch) -> str:
        requests = []
        batch = self.client.messages.batches.create(
            requests=requests
        )
        return batch.id