# Batch Router

A Python package designed to facilitate batch LLM requests efficiently across multiple providers with a unified interface.

## Overview

`batch-router` abstracts away the complexities of different LLM providers' batch APIs. It provides a unified object model for requests, messages, and content, allowing you to define your inputs once and route them to any supported provider (OpenAI, Anthropic, Google, vLLM).

## Installation

You can install the package using pip:

```bash
pip install batch-router
```

## Key Concepts

- **Unified Interface**: Use standard classes like `InputRequest`, `InputMessage`, and `TextContent` regardless of the underlying provider.
- **Batch Processing**: Designed specifically for high-volume, asynchronous batch processing.
- **Provider Agnostic**: Switch between OpenAI, Anthropic, Google GenAI, and vLLM with minimal code changes.

## Supported Providers

- **OpenAI**: Supports Chat Completions batch API.
- **Anthropic**: Supports Message Batches API.
- **Google GenAI**: Supports Batch API.
- **vLLM**: Supports local batch inference.

## Basic Usage

Here is a simple example of how to use `batch-router` with Google GenAI:

```python
import time
from dotenv import load_dotenv
load_dotenv()

from batch_router.core.base import BatchStatus, ProviderId
from batch_router.providers import GoogleGenAIProvider
from batch_router.core.input import InputBatch, InputRequest, InputMessage, InputMessageRole
from batch_router.core.base import TextContent, InferenceParams

# 1. Initialize the provider
# Ensure you have GOOGLE_API_KEY in your environment variables or pass it explicitly.
provider = GoogleGenAIProvider()

# 2. Create requests
# Define a list of requests. Each request has a unique custom_id and a list of messages.
requests = [
    InputRequest(
        custom_id=f"request_{i+1}",
        messages=[
            InputMessage(
                role=InputMessageRole.USER,
                contents=[
                    TextContent(text="Hello, how are you?")
                ]
            )
        ]
    )
    for i in range(5)
]

# 3. Create an InputBatch
input_batch = InputBatch(
    requests=requests
)

# 4. Configure Inference Parameters
# Define model, provider, system prompt, etc.
google_params = InferenceParams(
    model_id="gemini-1.5-flash",
    provider_id=ProviderId.GOOGLE,
    system_prompt="Answer like a pirate.",
    max_output_tokens=128
)

# 5. Apply parameters to the batch
google_genai_batch = input_batch.with_params(google_params)

# 6. Send the batch
batch_id = provider.send_batch(google_genai_batch)
print(f"Batch sent! ID: {batch_id}")

# 7. Poll for completion
while provider.poll_status(batch_id) != BatchStatus.COMPLETED:
    status = provider.poll_status(batch_id)
    print(f"Batch {batch_id} is {status}")
    if status in [BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED]:
        print("Batch failed or was cancelled.")
        break
    time.sleep(5)

# 8. Retrieve results
if provider.poll_status(batch_id) == BatchStatus.COMPLETED:
    results = provider.get_results(batch_id)
    for request in results.requests:
        print(f"Request {request.custom_id}:")
        for msg in request.messages:
            for content in msg.contents:
                if hasattr(content, 'text'):
                    print(f"  Response: {content.text}")
```

## Usage with Other Providers

### OpenAI

```python
from batch_router.providers import OpenAIChatCompletionsProvider

provider = OpenAIChatCompletionsProvider(api_key="your-api-key")
# Use InferenceParams with provider_id=ProviderId.OPENAI and an OpenAI model (e.g., "gpt-4o")
```

### Anthropic

```python
from batch_router.providers import AnthropicProvider

provider = AnthropicProvider(api_key="your-api-key")
# Use InferenceParams with provider_id=ProviderId.ANTHROPIC and an Anthropic model (e.g., "claude-3-5-sonnet-20241022")
```

### vLLM

```python
from batch_router.providers import vLLMProvider

# Requires local path to model and running vLLM environment
provider = vLLMProvider(model_path="/path/to/model")
# Use InferenceParams with provider_id=ProviderId.VLLM
```

## Advanced Features

### Multi-Modal Inputs

You can send images and audio (depending on provider support) using `ImageContent` and `AudioContent`.

```python
from batch_router.core.base import ImageContent

# Create from file
image_content = ImageContent.from_file("path/to/image.png")

# Or from base64 string
# image_content = ImageContent(image_base64="...")

message = InputMessage(
    role=InputMessageRole.USER,
    contents=[
        TextContent(text="What is in this image?"),
        image_content
    ]
)
```

### Reusing Batches

You can define a generic `InputBatch` and configure it for different providers using `.with_params()`. This allows you to run the same set of inputs against multiple models easily.

```python
# Define generic batch
input_batch = InputBatch(requests=...)

# Configure for OpenAI
openai_batch = input_batch.with_params(
    InferenceParams(provider_id=ProviderId.OPENAI, model_id="gpt-4o")
)

# Configure for Anthropic
anthropic_batch = input_batch.with_params(
    InferenceParams(provider_id=ProviderId.ANTHROPIC, model_id="claude-3-5-sonnet-20241022")
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
