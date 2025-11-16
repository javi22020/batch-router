from batch_router.providers.base.base_provider import BaseProvider
from batch_router.core.input.batch import InputBatch
from batch_router.core.output.batch import OutputBatch
from abc import abstractmethod

class BaseStreamProvider(BaseProvider):
    """A base class for all stream providers."""
    @abstractmethod
    def run_batch(self, input_batch: InputBatch) -> OutputBatch:
        """Run the batch inference. Convert InputBatch to provider format and run the batch, return the OutputBatch."""
        pass

__all__ = ["BaseStreamProvider"]
