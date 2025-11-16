from batch_router.providers.base.base_provider import BaseProvider
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.batch import OutputBatch
from batch_router.core.base.batch import BatchStatus
from batch_router.core.base.provider import ProviderId, ProviderMode
from abc import abstractmethod

class BaseBatchProvider(BaseProvider):
    """A base class for all batch providers."""
    def __init__(self, provider_id: ProviderId) -> None:
        super().__init__(
            provider_id=provider_id,
            mode=ProviderMode.BATCH
        )
    @abstractmethod
    def send_batch(self, input_batch: InputBatch) -> str:
        """Send the batch to the provider. Convert InputBatch to provider format and send the batch, return the batch_id."""
        pass

    @abstractmethod
    def poll_status(self, batch_id: str) -> BatchStatus:
        """Poll the status of the batch from the provider servers. Return the status of the batch."""
        pass

    @abstractmethod
    def get_results(self, batch_id: str) -> OutputBatch:
        """Get the results of the batch from the provider. Convert the results to OutputBatch."""
        pass

__all__ = ["BaseBatchProvider"]
