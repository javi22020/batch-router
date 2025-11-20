from batch_router.providers.base.base_provider import BaseProvider
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.batch import OutputBatch
from batch_router.core.base.batch import BatchStatus
from batch_router.core.base.provider import ProviderId, ProviderMode
from abc import abstractmethod

class BaseBatchProvider(BaseProvider):
    """
    A base class for all batch processing providers.

    This class extends BaseProvider to support batch-specific operations like sending batches,
    polling status, and retrieving results.

    Attributes:
        provider_id (ProviderId): The identifier of the provider.
    """
    def __init__(self, provider_id: ProviderId) -> None:
        """
        Initializes the BaseBatchProvider.

        Args:
            provider_id (ProviderId): The identifier of the provider.
        """
        super().__init__(
            provider_id=provider_id,
            mode=ProviderMode.BATCH
        )
    @abstractmethod
    def send_batch(self, input_batch: InputBatch) -> str:
        """
        Send the batch to the provider for processing.

        This method converts the InputBatch to the provider's format and initiates the batch process.

        Args:
            input_batch (InputBatch): The batch of requests to send.

        Returns:
            str: The unique identifier (batch_id) for the submitted batch.
        """
        pass

    @abstractmethod
    def poll_status(self, batch_id: str) -> BatchStatus:
        """
        Poll the status of the batch from the provider's servers.

        Args:
            batch_id (str): The unique identifier of the batch to check.

        Returns:
            BatchStatus: The current status of the batch (e.g., PENDING, RUNNING, COMPLETED).
        """
        pass

    @abstractmethod
    def get_results(self, batch_id: str) -> OutputBatch:
        """
        Retrieve the results of the batch from the provider.

        This method fetches the results and converts them into the unified OutputBatch format.

        Args:
            batch_id (str): The unique identifier of the completed batch.

        Returns:
            OutputBatch: The results of the batch processing.
        """
        pass

__all__ = ["BaseBatchProvider"]
