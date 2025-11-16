from abc import ABC, abstractmethod
from typing import Any
from batch_router.core.base.provider import ProviderId
from batch_router.core.base.modality import Modality
from batch_router.core.input.batch import InputBatch
from batch_router.core.input.message import InputMessage
from batch_router.core.output.batch import OutputBatch
from batch_router.core.output.message import OutputMessage
from batch_router.core.base.content import MessageContent
from enum import Enum

class ProviderMode(Enum):
    """The mode of a provider."""
    BATCH = "batch"
    STREAM = "stream"

class BaseProvider(ABC):
    """A base class for all providers."""
    provider_id: ProviderId
    mode: ProviderMode
    modalities: list[Modality]

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
    def convert_input_batch_from_unified_to_provider(self, batch: InputBatch) -> Any:
        """Convert input batch from unified to provider format."""
        pass

    @abstractmethod
    def convert_output_batch_from_provider_to_unified(self, batch: Any) -> OutputBatch:
        """Convert output batch from provider to unified format."""
        pass
