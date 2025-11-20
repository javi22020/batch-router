from abc import ABC, abstractmethod
from typing import Any
from batch_router.core.base.provider import ProviderId
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.batch import OutputBatch
from batch_router.core.base.request import InferenceParams
from batch_router.core.input.message import InputMessage
from batch_router.core.output.message import OutputMessage
from batch_router.core.input.request import InputRequest
from batch_router.core.output.request import OutputRequest
from batch_router.core.base.content import MessageContent
from batch_router.core.input.role import InputMessageRole
from batch_router.core.output.role import OutputMessageRole
from batch_router.core.base.provider import ProviderMode

class BaseProvider(ABC):
    """
    A base class for all providers.

    This class defines the interface for interacting with different model providers,
    requiring implementations for converting between unified formats and provider-specific formats.
    Use `BaseBatchProvider` for batch providers and `BaseStreamProvider` for stream providers.

    Attributes:
        provider_id (ProviderId): The identifier of the provider.
        mode (ProviderMode): The mode of the provider (BATCH or STREAM).
    """
    provider_id: ProviderId
    mode: ProviderMode

    def __init__(self, provider_id: ProviderId, mode: ProviderMode) -> None:
        """
        Initializes the BaseProvider.

        Args:
            provider_id (ProviderId): The identifier of the provider.
            mode (ProviderMode): The operational mode of the provider.
        """
        self.provider_id = provider_id
        self.mode = mode
    
    @abstractmethod
    def input_message_role_to_provider(self, role: InputMessageRole) -> str:
        """
        Convert input message role to provider-specific role string.

        Args:
            role (InputMessageRole): The unified input message role.

        Returns:
            str: The corresponding role string for the provider.
        """
        pass

    @abstractmethod
    def output_message_role_to_unified(self, role: str) -> OutputMessageRole:
        """
        Convert provider-specific output message role string to unified role.

        Args:
            role (str): The provider's role string.

        Returns:
            OutputMessageRole: The corresponding unified output message role.
        """
        pass

    @abstractmethod
    def inference_params_to_provider(self, params: InferenceParams) -> Any:
        """
        Convert unified inference parameters to provider-specific format.

        Args:
            params (InferenceParams): The unified inference parameters.

        Returns:
            Any: The parameters in the format expected by the provider.
        """
        pass

    @abstractmethod
    def convert_input_content_from_unified_to_provider(self, content: MessageContent) -> Any:
        """
        Convert input content from unified format to provider-specific format.

        Args:
            content (MessageContent): The unified message content.

        Returns:
            Any: The content in the format expected by the provider.
        """
        pass

    @abstractmethod
    def convert_output_content_from_provider_to_unified(self, content: Any) -> MessageContent:
        """
        Convert output content from provider-specific format to unified format.

        Args:
            content (Any): The content in the provider's format.

        Returns:
            MessageContent: The unified message content.
        """
        pass

    @abstractmethod
    def convert_input_message_from_unified_to_provider(self, message: InputMessage) -> Any:
        """
        Convert input message from unified format to provider-specific format.

        Args:
            message (InputMessage): The unified input message.

        Returns:
            Any: The message in the format expected by the provider.
        """
        pass

    @abstractmethod
    def convert_output_message_from_provider_to_unified(self, message: Any) -> OutputMessage:
        """
        Convert output message from provider-specific format to unified format.

        Args:
            message (Any): The message in the provider's format.

        Returns:
            OutputMessage: The unified output message.
        """
        pass

    @abstractmethod
    def convert_input_request_from_unified_to_provider(self, request: InputRequest) -> Any:
        """
        Convert input request from unified format to provider-specific format.

        Args:
            request (InputRequest): The unified input request.

        Returns:
            Any: The request in the format expected by the provider.
        """
        pass

    @abstractmethod
    def convert_output_request_from_provider_to_unified(self, request: Any) -> OutputRequest:
        """
        Convert output request from provider-specific format to unified format.

        Args:
            request (Any): The request in the provider's format.

        Returns:
            OutputRequest: The unified output request.
        """
        pass

    @abstractmethod
    def convert_input_batch_from_unified_to_provider(self, batch: InputBatch) -> Any:
        """
        Convert input batch from unified format to provider-specific format.

        Args:
            batch (InputBatch): The unified input batch.

        Returns:
            Any: The batch in the format expected by the provider.
        """
        pass

    @abstractmethod
    def convert_output_batch_from_provider_to_unified(self, batch: Any) -> OutputBatch:
        """
        Convert output batch from provider-specific format to unified format.

        Args:
            batch (Any): The batch in the provider's format.

        Returns:
            OutputBatch: The unified output batch.
        """
        pass

    @abstractmethod
    def count_input_request_tokens(self, request: InputRequest) -> int:
        """
        Count the total number of tokens in a single input request.

        Args:
            request (InputRequest): The unified input request.

        Returns:
            int: The total token count for the request.
        """
        pass

    def count_input_batch_tokens(self, batch: InputBatch) -> int:
        """
        Count the total number of tokens in the input batch.

        Iterates through all requests in the batch and sums up their token counts.

        Args:
            batch (InputBatch): The unified input batch.

        Returns:
            int: The total token count for the entire batch.
        """
        input_requests = batch.requests
        total_tokens = 0
        for request in input_requests:
            total_tokens += self.count_input_request_tokens(request)
        
        return total_tokens

__all__ = ["BaseProvider"]
