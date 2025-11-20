from enum import Enum

class ProviderId(Enum):
    """
    Enumeration representing the supported model providers.

    Attributes:
        ANTHROPIC (str): Represents the Anthropic provider.
        OPENAI (str): Represents the OpenAI provider.
        GOOGLE (str): Represents the Google provider.
        VLLM (str): Represents the vLLM provider.
    """
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    VLLM = "vllm"

class ProviderMode(Enum):
    """
    Enumeration representing the operational modes of a provider.

    Attributes:
        BATCH (str): Represents batch processing mode.
        STREAM (str): Represents streaming processing mode.
    """
    BATCH = "batch"
    STREAM = "stream"
