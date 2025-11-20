"""
The `batch_router.providers` package contains implementations for various LLM providers.

It exposes the following providers:
- `vLLMProvider`: For local batch inference using vLLM.
- `OpenAIChatCompletionsProvider`: For batch inference using OpenAI's Chat Completions API.
- `AnthropicProvider`: For batch inference using Anthropic's API.
- `GoogleGenAIProvider`: For batch inference using Google's GenAI API.
"""

from .vllm.vllm_provider import vLLMProvider
from .openai import OpenAIChatCompletionsProvider
from .anthropic import AnthropicProvider
from .google import GoogleGenAIProvider

__all__ = ["vLLMProvider", "OpenAIChatCompletionsProvider", "AnthropicProvider", "GoogleGenAIProvider"]
