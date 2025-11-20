from batch_router.providers.base.base_provider import BaseProvider
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.batch import OutputBatch
from abc import abstractmethod

class BaseStreamProvider(BaseProvider):
    """
    A base class for all stream processing providers.

    This class extends BaseProvider to support streaming-like operations where a batch is run directly.
    """
    @abstractmethod
    def run_batch(self, input_batch: InputBatch) -> OutputBatch:
        """
        Run the batch inference immediately.

        This method converts the InputBatch to the provider's format, processes it, and returns the results.

        Args:
            input_batch (InputBatch): The batch of requests to process.

        Returns:
            OutputBatch: The results of the batch processing.
        """
        pass

__all__ = ["BaseStreamProvider"]
