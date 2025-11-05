1. C (Common + provider specific kwargs)
2. B (Automatic upload, but to Google Generative AI API, not using GCP; I believe you're thinking of the deprecated way)
3. B (Async subprocess)
4. We should download each provider's files, then convert them to the unified format.
5. Batch size splitting is up to the user; the library does not handle it automatically, even if it throws an error.
6. A (Blocking poll with exponential backoff)
7. All providers must return both successful and failed results.
8. In a `.batch_router/generated/` subdirectory within the project root (all library-generated files must be placed inside ``).
9. A (User must provide custom IDs for each request)
10. User must give both provider name and model id for the provider API. For example, if using Anthropic, the user must provide "anthropic" as the provider name and the specific model id as "claude-4-5-sonnet-latest".
11. The message format is the main problem of this project. I believe the best approach is to create a unified message format with optional provider-specific kwargs. What I'm certain of is that the system prompt must be defined at the request level, not inside the messages.
12. A (Fail immediately with clear error message)

I believe the next thing to do is to sketch out the main classes needed for the library. I've already implemented these:

```python
class UnifiedMessage:
    role: str  # Role of the message, either "user" or "assistant"
    content: str  # Content of the message.
    provider_kwargs: dict  # Provider-specific kwargs.

class UnifiedRequest:
    custom_id: str  # Unique identifier for the request.
    model: str  # Model identifier, e.g., "gpt-4", "claude-4", etc.
    messages: list[UnifiedMessage]  # List of messages in a unified format.
    provider_kwargs: dict  # Provider-specific kwargs.

class UnifiedBatch:
    batch_id: str  # Unique identifier for the batch.
    provider: str  # Name of the provider, e.g., "openai", "anthropic", etc.
    requests: list[UnifiedRequest] # List of unified requests.

class BaseProvider: # Abstract class for all providers.
    name: str # Name of the provider, e.g., "openai", "anthropic", etc.
    async def send_batch(self, batch: UnifiedBatch) -> UnifiedBatchResponse: # Send a batch of requests.
        pass
    async def poll_batch(self, batch_id: str) -> UnifiedBatchResponse: # Poll for batch results.
        pass

```

Reason about these classes and whether they make sense or not. Propose and compare different designs. Check any necessary documentation for the providers to ensure that the design is sound. It's crucial that the design is flexible enough to accommodate the specific requirements of each provider while maintaining a unified interface.
