from abc import ABC, abstractmethod
from typing import Any
from batch_router.core.base.provider import ProviderId
from batch_router.core.base.modality import Modality
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.batch import OutputBatch
from batch_router.core.input.message import InputMessage
from batch_router.core.output.message import OutputMessage
from batch_router.core.input.request import InputRequest
from batch_router.core.output.request import OutputRequest
from batch_router.core.base.content import MessageContent
from batch_router.core.base.provider import ProviderMode

class BaseProvider(ABC):
    """A base class for all providers. Use `BaseBatchProvider` for batch providers and `BaseStreamProvider` for stream providers."""
    provider_id: ProviderId
    mode: ProviderMode
    modalities: list[Modality]

    def __init__(self, provider_id: ProviderId, mode: ProviderMode, modalities: list[Modality]) -> None:
        self.provider_id = provider_id
        self.mode = mode
        self.modalities = modalities

    @abstractmethod
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> Any:
        """Convert input content from unified to provider format."""
        pass

    @abstractmethod
    def convert_output_content_from_provider_to_unified(self, content: Any) -> MessageContent:
        """Convert output content from provider to unified format."""
        pass

    @abstractmethod
    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> Any:
        """Convert input message from unified to provider format."""
        pass

    @abstractmethod
    def convert_output_message_from_provider_to_unified(self, message: Any) -> OutputMessage:
        """Convert output message from provider to unified format."""
        pass

    @abstractmethod
    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> Any:
        """Convert input request from unified to provider format."""
        pass

    @abstractmethod
    def convert_output_request_from_provider_to_unified(self, request: Any) -> OutputRequest:
        """Convert output request from provider to unified format."""
        pass

    @abstractmethod
    def convert_input_batch_from_unified_to_provider(self, batch: InputBatch) -> Any:
        """Convert input batch from unified to provider format."""
        pass

    @abstractmethod
    def convert_output_batch_from_provider_to_unified(self, batch: Any) -> OutputBatch:
        """Convert output batch from provider to unified format."""
        pass
