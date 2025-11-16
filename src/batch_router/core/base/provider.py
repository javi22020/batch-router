from enum import Enum

class ProviderId(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    VLLM = "vllm"
