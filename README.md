# Batch Router

A Python package designed to facilitate batch LLM requests efficiently across multiple providers with a unified interface.

> [!NOTE]
> **We're Looking for Contributors!**
> This project is actively seeking contributors to help improve and expand the batch-router library. Whether you're interested in adding new features, improving documentation, fixing bugs, or adding support for additional providers, your contributions are welcome!

## Overview

Batch Router provides a standardized way to send batch requests to different LLM providers, abstracting away provider-specific formats and APIs. The base package includes OpenAI and Anthropic providers, with optional support for Google, Mistral, and local processing via vLLM. This allows you to:

- Write requests once in a unified format
- Switch between providers seamlessly
- Process large volumes of LLM requests cost-effectively
- Track batch status and retrieve results consistently
- Maintain full transparency with JSONL file logging

## Features

- **Unified Request Format**: Single format for all providers
- **Multi-Provider Support**: OpenAI and Anthropic (Claude) included by default, with optional support for Google (Gemini) and Mistral
- **Cost Reduction**: Leverage batch APIs for up to 50% cost savings (OpenAI)
- **Async Support**: Fully asynchronous operations
- **Type Safe**: Comprehensive type hints throughout
- **Transparent**: All requests and responses saved as JSONL files
- **Flexible**: Support for text, images, and multimodal content
- **Local Processing**: vLLM provider for local batch processing

## Installation

```bash
# Basic installation (includes OpenAI and Anthropic providers)
pip install batch-router

# With optional provider dependencies
pip install batch-router[google]    # For Google Gemini support
pip install batch-router[mistral]   # For Mistral support
pip install batch-router[all]       # Install all optional providers

# For local processing with vLLM
pip install vllm
```

## Quick Start

```python
import asyncio
from batch_router import (
    UnifiedRequest,
    UnifiedBatchMetadata,
    UnifiedMessage,
    TextContent,
    OpenAIProvider,
)

async def main():
    # Create provider
    provider = OpenAIProvider(api_key="your-api-key")

    # Create requests
    requests = [
        UnifiedRequest(
            custom_id="request-1",
            model="gpt-4o-mini",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[TextContent(text="What is the capital of France?")]
                )
            ]
        ),
        UnifiedRequest(
            custom_id="request-2",
            model="gpt-4o-mini",
            messages=[
                UnifiedMessage(
                    role="user",
                    content=[TextContent(text="What is 2+2?")]
                )
            ],
            system_prompt="You are a helpful math tutor."
        )
    ]

    # Create batch
    batch = UnifiedBatchMetadata(
        provider="openai",
        requests=requests
    )

    # Send batch
    batch_id = await provider.send_batch(batch)
    print(f"Batch submitted: {batch_id}")

    # Check status
    status = await provider.get_status(batch_id)
    print(f"Status: {status.status.value}")
    print(f"Progress: {status.request_counts.succeeded}/{status.request_counts.total}")

    # Get results (when complete)
    if status.is_complete():
        async for result in provider.get_results(batch_id):
            print(f"{result.custom_id}: {result.status.value}")
            if result.response:
                print(f"Response: {result.response}")

asyncio.run(main())
```

## Supported Providers

### Included Providers

The following providers are included in the base installation:

#### OpenAI
- **API**: Batch API for Chat Completions
- **Cost**: 50% reduction compared to sync API
- **Completion**: 24-hour window
- **Models**: All chat completion models (gpt-4o, gpt-4o-mini, etc.)

```python
from batch_router import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")
```

#### Anthropic (Claude)
- **API**: Message Batches API
- **Cost**: 50% reduction compared to sync API
- **Completion**: 24-hour window
- **Models**: Claude models (claude-sonnet-4-5, claude-3-5-sonnet-20241022, etc.)

```python
from batch_router import AnthropicProvider

provider = AnthropicProvider(api_key="sk-ant-...")
```

### Optional Providers

The following providers require additional installation:

#### Google (Gemini)

Requires: `pip install batch-router[google]`
- **API**: Batch Prediction API
- **Cost**: Varies by model
- **Completion**: Varies
- **Models**: Gemini models (gemini-2.0-flash-exp, etc.)

```python
from batch_router import GoogleProvider

provider = GoogleProvider(api_key="...")
```

#### Mistral

Requires: `pip install batch-router[mistral]`
- **API**: Batch Inference API
- **Cost**: Reduced pricing for batch operations
- **Completion**: Varies by batch size
- **Models**: Mistral models (mistral-small-latest, mistral-large-latest, etc.)

```python
from batch_router import MistralProvider

provider = MistralProvider(api_key="...")
```

### Local Provider

#### vLLM (Local)

Requires separate installation: `pip install vllm`
- **API**: Local batch processing via vLLM CLI
- **Cost**: Free (runs locally)
- **Completion**: Immediate (based on hardware)
- **Models**: Any model compatible with vLLM

```python
from batch_router import VLLMProvider

# Requires vLLM installed: pip install vllm
provider = VLLMProvider(
    vllm_command="vllm",
    additional_args=["--tensor-parallel-size", "2"]
)
```

## Core Concepts

### Unified Request Format

All providers use the same request structure:

```python
UnifiedRequest(
    custom_id="unique-identifier",  # Your unique ID
    model="model-name",             # Provider-specific model
    messages=[...],                  # Conversation messages
    system_prompt="...",            # Optional system prompt
    generation_config=GenerationConfig(...),  # Optional params
    provider_kwargs={...}           # Provider-specific options
)
```

### System Prompts

System prompts are handled differently by each provider:

- **OpenAI**: Converted to a message with `role="system"`
- **Anthropic**: Uses the `system` parameter
- **Google**: Uses `systemInstruction` in config
- **Mistral**: Converted to a message with `role="system"`
- **vLLM**: Converted to a message with `role="system"` (OpenAI-compatible)

Batch Router abstracts this at the request level with `system_prompt`.

### Messages

Messages use a unified content format:

```python
from batch_router import UnifiedMessage, TextContent, ImageContent

# Text message
message = UnifiedMessage(
    role="user",
    content=[TextContent(text="Hello!")]
)

# Multimodal message
message = UnifiedMessage(
    role="user",
    content=[
        TextContent(text="What's in this image?"),
        ImageContent(
            source_type="url",
            media_type="image/jpeg",
            data="https://example.com/image.jpg"
        )
    ]
)

# Base64 image
message = UnifiedMessage(
    role="user",
    content=[
        ImageContent(
            source_type="base64",
            media_type="image/png",
            data="iVBORw0KGgoAAAANS..."
        )
    ]
)
```

### Generation Configuration

Control generation parameters uniformly:

```python
from batch_router import GenerationConfig

config = GenerationConfig(
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    stop_sequences=["END"],
    presence_penalty=0.1,
    frequency_penalty=0.1
)

request = UnifiedRequest(
    custom_id="req-1",
    model="gpt-4o",
    messages=[...],
    generation_config=config
)
```

## Batch Operations

### Sending a Batch

```python
batch_id = await provider.send_batch(batch)
```

This will:
1. Convert requests to provider format
2. Save unified format JSONL to `.batch_router/generated/<provider>/`
3. Save provider-specific format JSONL
4. Upload/submit to provider API
5. Return a batch ID for tracking

### Checking Status

```python
status = await provider.get_status(batch_id)

print(f"Status: {status.status.value}")
print(f"Total: {status.request_counts.total}")
print(f"Succeeded: {status.request_counts.succeeded}")
print(f"Failed: {status.request_counts.errored}")
print(f"Processing: {status.request_counts.processing}")
print(f"Created: {status.created_at}")
print(f"Completed: {status.completed_at}")
```

Possible statuses:
- `VALIDATING`: Initial validation (OpenAI only)
- `IN_PROGRESS`: Processing requests
- `COMPLETED`: All requests processed
- `FAILED`: Batch failed
- `CANCELLED`: Batch was cancelled
- `EXPIRED`: Batch expired before completion

### Retrieving Results

```python
async for result in provider.get_results(batch_id):
    print(f"Request ID: {result.custom_id}")
    print(f"Status: {result.status.value}")

    if result.status == ResultStatus.SUCCEEDED:
        # Access response based on provider format
        response = result.response
        print(f"Response: {response}")
    elif result.status == ResultStatus.ERRORED:
        print(f"Error: {result.error}")
```

### Cancelling a Batch

```python
cancelled = await provider.cancel_batch(batch_id)
if cancelled:
    print("Batch cancelled successfully")
else:
    print("Batch already complete")
```

### Listing Batches

```python
batches = await provider.list_batches(limit=10)
for batch_status in batches:
    print(f"{batch_status.batch_id}: {batch_status.status.value}")
```

## File Management

Batch Router maintains transparency by saving all data as JSONL files:

```
.batch_router/
└── generated/
    ├── openai/
    │   ├── batch_<batch_id>_unified.jsonl      # Unified format (reference)
    │   ├── batch_<batch_id>_provider.jsonl     # Provider format (sent)
    │   ├── batch_<batch_id>_output.jsonl       # Raw provider output
    │   └── batch_<batch_id>_results.jsonl      # Unified results
    ├── anthropic/
    ├── google/
    └── vllm/
```

### File Types

1. **unified.jsonl**: Your original requests in unified format
2. **provider.jsonl**: Converted to provider-specific format (what gets sent)
3. **output.jsonl**: Raw results from the provider
4. **results.jsonl**: Results converted back to unified format

## Advanced Usage

### Provider-Specific Options

Use `provider_kwargs` to pass provider-specific parameters:

```python
# Anthropic: Enable thinking/reasoning
request = UnifiedRequest(
    custom_id="reasoning-task",
    model="claude-sonnet-4-5",
    messages=[...],
    provider_kwargs={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2000
        }
    }
)

# OpenAI: Enable structured outputs
request = UnifiedRequest(
    custom_id="structured-task",
    model="gpt-4o",
    messages=[...],
    provider_kwargs={
        "response_format": {
            "type": "json_schema",
            "json_schema": {...}
        }
    }
)
```

### Switching Providers

The unified format makes it easy to switch providers:

```python
# Same requests, different providers
requests = [...]  # List of UnifiedRequest objects

# Try with OpenAI
openai_batch = UnifiedBatchMetadata(provider="openai", requests=requests)
openai_provider = OpenAIProvider()
batch_id = await openai_provider.send_batch(openai_batch)

# Or try with Anthropic
anthropic_batch = UnifiedBatchMetadata(provider="anthropic", requests=requests)
anthropic_provider = AnthropicProvider()
batch_id = await anthropic_provider.send_batch(anthropic_batch)
```

### Custom Base URLs

For custom endpoints or proxies:

```python
# OpenAI with custom base URL
provider = OpenAIProvider(
    api_key="...",
    base_url="https://custom-proxy.example.com/v1"
)

# Anthropic with custom base URL
provider = AnthropicProvider(
    api_key="...",
    base_url="https://custom-endpoint.example.com"
)
```

### vLLM with Custom Arguments

```python
provider = VLLMProvider(
    vllm_command="vllm",
    additional_args=[
        "--tensor-parallel-size", "4",
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "8192"
    ]
)
```

## API Reference

### Core Classes

#### `UnifiedRequest`
- `custom_id`: str - Unique identifier for the request
- `model`: str - Model identifier
- `messages`: list[UnifiedMessage] - Conversation messages
- `system_prompt`: Optional[str | list[str]] - System prompt
- `generation_config`: Optional[GenerationConfig] - Generation parameters
- `provider_kwargs`: dict[str, Any] - Provider-specific options

#### `UnifiedMessage`
- `role`: str - Message role ("user" or "assistant")
- `content`: list[TextContent | ImageContent | DocumentContent] - Message content

#### `GenerationConfig`
- `max_tokens`: Optional[int]
- `temperature`: Optional[float]
- `top_p`: Optional[float]
- `top_k`: Optional[int]
- `stop_sequences`: Optional[list[str]]
- `presence_penalty`: Optional[float]
- `frequency_penalty`: Optional[float]

#### `UnifiedBatchMetadata`
- `provider`: str - Provider name ("openai", "anthropic", "google", "mistral", "vllm")
- `requests`: list[UnifiedRequest] - List of requests
- `metadata`: dict[str, Any] - Optional metadata

### Provider Methods

All providers implement:

#### `async send_batch(batch: UnifiedBatchMetadata) -> str`
Submit a batch for processing. Returns batch ID.

#### `async get_status(batch_id: str) -> BatchStatusResponse`
Get current status of a batch.

#### `async get_results(batch_id: str) -> AsyncIterator[UnifiedResult]`
Stream results from a completed batch.

#### `async cancel_batch(batch_id: str) -> bool`
Cancel a running batch. Returns True if cancelled, False if already complete.

#### `async list_batches(limit: int = 20) -> list[BatchStatusResponse]`
List recent batches (not supported by all providers).

## Environment Variables

Set API keys via environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."
```

Then initialize providers without explicit keys:

```python
provider = OpenAIProvider()  # Uses OPENAI_API_KEY
provider = AnthropicProvider()  # Uses ANTHROPIC_API_KEY
provider = GoogleProvider()  # Uses GOOGLE_API_KEY
provider = MistralProvider()  # Uses MISTRAL_API_KEY
```

## Error Handling

```python
from batch_router.exceptions import (
    ProviderError,
    BatchNotFoundError,
    BatchNotCompleteError,
)

try:
    batch_id = await provider.send_batch(batch)
except ProviderError as e:
    print(f"Provider error: {e}")
except ValueError as e:
    print(f"Invalid request: {e}")

try:
    status = await provider.get_status(batch_id)
except BatchNotFoundError:
    print("Batch not found")

try:
    async for result in provider.get_results(batch_id):
        process_result(result)
except BatchNotCompleteError:
    print("Batch is still processing")
```

## Best Practices

1. **Use Unique Custom IDs**: Always provide unique `custom_id` values to track individual requests
2. **Monitor Status**: Poll `get_status()` before retrieving results
3. **Handle Partial Failures**: Some requests may succeed while others fail - check individual result statuses
4. **Leverage Local Files**: Use saved JSONL files for debugging and audit trails
5. **Start Small**: Test with a small batch before scaling up
6. **Set Appropriate Timeouts**: Cloud providers typically complete within 24 hours

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Links

- **GitHub**: [https://github.com/javi22020/batch-router](https://github.com/javi22020/batch-router)
- **PyPI**: [https://pypi.org/project/batch-router](https://pypi.org/project/batch-router)

## Support

For issues, questions, or contributions, please visit the GitHub repository.
