from enum import Enum

class ProviderId(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    VLLM = "vllm"

class ProviderMode(Enum):
    """The mode of a provider."""
    BATCH = "batch"
    STREAM = "stream"
