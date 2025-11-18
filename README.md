# Batch Router

A Python package designed to facilitate batch LLM requests efficiently across multiple providers with a unified interface.

## Installation

You can install the package using pip:

```bash
pip install batch-router
```

## Basic Usage

Here is a simple example of how to use `batch-router` with vLLM:

```python
import time
from batch_router.core.base import BatchConfig, BatchStatus, ProviderId
from batch_router.providers import vLLMProvider
from batch_router.core.input import InputBatch, InputRequest, InputMessage, InputMessageRole
from batch_router.core.base import TextContent, InferenceParams

# Initialize the provider
provider = vLLMProvider(model_path="./gemma_3_270m")

# Create requests
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
        ],
        params=InferenceParams(
            max_output_tokens=100
        )
    )
    for i in range(5)
]

# Create an input batch
input_batch = InputBatch(
    requests=requests
)

# Configure the batch for vLLM
vllm_batch = input_batch.with_config(
    config=BatchConfig(
        provider_id=ProviderId.VLLM,
        model_id="./gemma_3_270m"
    )
)

# Send the batch
batch_id = provider.send_batch(vllm_batch)

# Poll for completion
while provider.poll_status(batch_id) != BatchStatus.COMPLETED:
    time.sleep(5)
    print(f"Batch {batch_id} is {provider.poll_status(batch_id)}")

# Retrieve results
results = provider.get_results(batch_id)
for request in results.requests:
    print(f"Request {request.custom_id} response: {request.messages[0].contents[0].text}")
```

## Available Providers

`batch-router` supports multiple providers. You can import them from `batch_router.providers` (or their specific submodules if not exposed directly).

### OpenAI

```python
from batch_router.providers import OpenAIChatProvider

provider = OpenAIChatProvider(api_key="your-api-key")
```

### Anthropic

```python
from batch_router.providers import AnthropicProvider

provider = AnthropicProvider(api_key="your-api-key")
```

### Google GenAI

```python
from batch_router.providers.google.google_genai_provider import GoogleGenAIProvider

provider = GoogleGenAIProvider(api_key="your-api-key")
```

### vLLM

```python
from batch_router.providers import vLLMProvider

provider = vLLMProvider(model_path="/path/to/model")
```

## Advanced Usage

### Multi-Modal Inputs

You can send images and audio (depending on provider support) using `ImageContent` and `AudioContent`.

```python
from batch_router.core.base import ImageContent, Modality

image_content = ImageContent(image_base64="base64_encoded_image_string")
# Add to InputMessage contents...
```

### Reusing Batches across Providers

You can define a generic `InputBatch` and configure it for different providers using `.with_config()`.

```python
# Define generic batch
input_batch = InputBatch(requests=...)

# Configure for OpenAI
openai_batch = input_batch.with_config(
    config=BatchConfig(
        provider_id=ProviderId.OPENAI,
        model_id="gpt-4o"
    )
)

# Configure for Anthropic
anthropic_batch = input_batch.with_config(
    config=BatchConfig(
        provider_id=ProviderId.ANTHROPIC,
        model_id="claude-3-5-sonnet-20241022"
    )
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
