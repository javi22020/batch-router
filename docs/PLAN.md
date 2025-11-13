# Batch LLM Router - Implementation Plan

## Overview

A unified Python library for batch processing LLM requests across multiple providers (OpenAI, Anthropic, Google GenAI, Mistral, vLLM). The library provides a single interface for sending batch requests, polling for completion, and retrieving results in a unified format.

**Core Principle:** All providers save JSONL files (both provider-specific and unified formats) to `.batch_router/generated/<provider>/` for transparency and debugging.

---

## Project Structure

```
batch_router/
├── __init__.py                      # Main exports
├── core/                            # Core data models (no business logic)
│   ├── __init__.py
│   ├── content.py                   # Multimodal content types
│   ├── messages.py                  # UnifiedMessage
│   ├── requests.py                  # UnifiedRequest, UnifiedBatchMetadata  
│   ├── responses.py                 # BatchStatusResponse, UnifiedResult
│   ├── config.py                    # GenerationConfig
│   ├── enums.py                     # Status enums
│   ├── types.py                     # Type aliases
│   └── base.py                      # BaseProvider abstract class
├── providers/                       # Provider implementations
│   ├── __init__.py
│   ├── openai_provider.py                    # OpenAI Batch API implementation
│   ├── anthropic_provider.py                 # Anthropic Message Batches API
│   ├── google_provider.py                    # Google GenAI Batch API
│   ├── mistral_provider.py                   # Mistral Batch API
│   └── vllm_provider.py                      # vLLM run-batch subprocess
├── router.py                        # BatchRouter orchestration
├── utilities/                           # Shared utilities
│   ├── __init__.py
│   ├── file_manager.py              # JSONL file operations
│   ├── logging.py                   # Structured logging
│   └── validation.py                # Request validation
├── exceptions.py                    # Custom exceptions
└── constants.py                     # Shared constants
```

---

## Requirements

### Python Dependencies
```
# Core dependencies
python >= 3.11  # For native union types (X | Y)
openai >= 1.0.0
anthropic >= 0.40.0
google-genai >= 1.0.0  # New unified SDK
mistralai >= 1.0.0     # Mistral AI SDK

# Async support
aiofiles >= 23.0.0  # For async file operations
httpx >= 0.27.0     # For async HTTP (used by provider SDKs)

# Optional
pydantic >= 2.0.0   # For enhanced validation (optional)
```

### External Tools
- **vLLM**: `vllm` CLI must be available in PATH for vLLM provider
  - Check with: `subprocess.run(['vllm', '--version'], capture_output=True)`

---

## Core Data Models

### File: `core/content.py`

**Purpose:** Define multimodal content types for messages.

**Classes to implement:**

```python
@dataclass
class TextContent:
    """Plain text content in a message."""
    type: Literal["text"] = "text"
    text: str = ""

@dataclass
class ImageContent:
    """Image content (base64, URL, or file URI)."""
    type: Literal["image"] = "image"
    source_type: Literal["base64", "url", "file_uri"]
    media_type: str  # "image/jpeg", "image/png", etc.
    data: str  # base64 string, URL, or gs:// URI

@dataclass
class DocumentContent:
    """PDF/document content (base64, URL, or file URI)."""
    type: Literal["document"] = "document"
    source_type: Literal["base64", "url", "file_uri"]
    media_type: str  # "application/pdf", etc.
    data: str
```

**Key Points:**
- Use `@dataclass` decorator from `dataclasses`
- All fields should have type hints
- `type` field is literal for runtime type checking
- `source_type` determines how `data` should be interpreted

---

### File: `core/types.py`

**Purpose:** Type aliases and protocols for better IDE support.

**Contents:**
```python
from typing import Union
from .content import TextContent, ImageContent, DocumentContent

# Type alias for any message content
MessageContent = Union[TextContent, ImageContent, DocumentContent]

# Type alias for system prompt (can be string or list of strings)
SystemPrompt = Union[str, list[str], None]
```

---

### File: `core/messages.py`

**Purpose:** Unified message representation.

**Class to implement:**

```python
@dataclass
class UnifiedMessage:
    """
    Unified message format across all providers.
    
    Important: System messages should NOT be in the messages array.
    Use UnifiedRequest.system_prompt instead.
    Only 'user' and 'assistant' roles are allowed here.
    """
    role: Literal["user", "assistant"]
    content: list[MessageContent]  # List to support multimodal
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_text(cls, role: str, text: str, **kwargs) -> "UnifiedMessage":
        """Convenience constructor for text-only messages."""
        return cls(
            role=role,
            content=[TextContent(text=text)],
            provider_kwargs=kwargs
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Implementation needed
        pass
```

**Validation Requirements:**
- `role` must be exactly "user" or "assistant" (no "system")
- `content` list cannot be empty
- Each item in `content` must be a valid content object

---

### File: `core/config.py`

**Purpose:** Common generation parameters across providers.

**Class to implement:**

```python
@dataclass
class GenerationConfig:
    """
    Common generation parameters that map to provider-specific params.
    
    Providers will convert these to their native parameter names:
    - OpenAI: max_tokens, temperature, top_p, presence_penalty, frequency_penalty
    - Anthropic: max_tokens, temperature, top_p, top_k
    - Google: maxOutputTokens, temperature, topP, topK
    - vLLM: max_tokens, temperature, top_p, top_k
    """
    # Core parameters (supported by all)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    
    # Optional parameters (not all providers support)
    top_k: Optional[int] = None  # Not in OpenAI
    stop_sequences: Optional[list[str]] = None
    presence_penalty: Optional[float] = None  # OpenAI, Google only
    frequency_penalty: Optional[float] = None  # OpenAI only
    
    def __post_init__(self):
        """Validate parameter ranges."""
        if self.temperature is not None:
            if not (0 <= self.temperature <= 2):
                raise ValueError("temperature must be between 0 and 2")
        
        if self.max_tokens is not None:
            if self.max_tokens < 1:
                raise ValueError("max_tokens must be positive")
        
        if self.top_p is not None:
            if not (0 <= self.top_p <= 1):
                raise ValueError("top_p must be between 0 and 1")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
```

**Important:** These are COMMON parameters. Providers convert them to their specific names.

---

### File: `core/requests.py`

**Purpose:** Request structures for batch operations.

**Classes to implement:**

```python
@dataclass
class UnifiedRequest:
    """
    A single request in unified format.
    
    System prompt is at REQUEST level, not in messages array.
    This design choice allows proper handling across all providers:
    - OpenAI: Converts to message with role="system"
    - Anthropic: Uses 'system' parameter
    - Google: Uses 'systemInstruction' in config
    - Mistral: Converts to message with role="system"
    """
    custom_id: str  # REQUIRED: User must provide unique ID
    model: str  # Provider-specific model ID (e.g., "gpt-4o", "claude-sonnet-4-5")
    messages: list[UnifiedMessage]
    
    # System prompt at request level
    system_prompt: Optional[str | list[str]] = None
    
    # Generation parameters
    generation_config: Optional[GenerationConfig] = None
    
    # Provider-specific advanced features
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate request structure."""
        if not self.custom_id:
            raise ValueError("custom_id is required and cannot be empty")
        
        if not self.messages:
            raise ValueError("messages list cannot be empty")
        
        # Ensure no system messages in messages array
        for msg in self.messages:
            if msg.role not in ["user", "assistant"]:
                raise ValueError(
                    f"Invalid role '{msg.role}' in messages. "
                    "Use system_prompt field for system instructions."
                )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        # Implementation needed
        pass


@dataclass
class UnifiedBatchMetadata:
    """
    Batch metadata before sending to provider.
    This is NOT the response - just the input specification.
    """
    provider: str  # "openai", "anthropic", "google", "mistral", "vllm"
    requests: list[UnifiedRequest]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate batch metadata."""
        valid_providers = ["openai", "anthropic", "google", "mistral", "vllm"]
        if self.provider not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}")
        
        if not self.requests:
            raise ValueError("requests list cannot be empty")
        
        # Check for duplicate custom_ids
        custom_ids = [req.custom_id for req in self.requests]
        if len(custom_ids) != len(set(custom_ids)):
            raise ValueError("Duplicate custom_id values found in requests")
```

---

### File: `core/enums.py`

**Purpose:** Status enumerations.

**Enums to implement:**

```python
from enum import Enum

class BatchStatus(Enum):
    """
    Unified batch status across all providers.
    
    Provider mapping:
    - OpenAI: validating, in_progress, completed, failed, expired, cancelled
    - Anthropic: in_progress, ended (then check request_counts)
    - Google: JOB_STATE_PENDING, JOB_STATE_RUNNING, JOB_STATE_SUCCEEDED,
              JOB_STATE_FAILED, JOB_STATE_CANCELLED
    - Mistral: QUEUED, RUNNING, SUCCESS, FAILED, TIMEOUT_EXCEEDED, CANCELLATION_REQUESTED, CANCELLED
    - vLLM: File-based (processing if file not ready, completed when exists)
    """
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ResultStatus(Enum):
    """
    Status of individual request within a batch.
    All providers support these statuses.
    """
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
```

---

### File: `core/responses.py`

**Purpose:** Response structures from batch operations.

**Classes to implement:**

```python
@dataclass
class RequestCounts:
    """
    Breakdown of request statuses within a batch.
    Used to show progress and completion statistics.
    """
    total: int
    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    cancelled: int = 0
    expired: int = 0
    
    def is_complete(self) -> bool:
        """Check if all requests have finished processing."""
        return self.processing == 0
    
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.succeeded / self.total) * 100


@dataclass
class BatchStatusResponse:
    """
    Response from checking batch status.
    Does NOT contain actual results - only status info.
    """
    batch_id: str
    provider: str
    status: BatchStatus
    request_counts: RequestCounts
    
    # Timestamps (ISO 8601 format)
    created_at: str
    completed_at: Optional[str] = None
    expires_at: Optional[str] = None
    
    # Provider-specific additional data
    provider_data: dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if batch has finished processing."""
        return self.status in [
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED
        ]


@dataclass
class UnifiedResult:
    """
    Individual request result within a batch.
    
    Results from all providers are converted to this unified format.
    """
    custom_id: str
    status: ResultStatus
    
    # If succeeded: full response object
    response: Optional[dict[str, Any]] = None
    
    # If errored: error details
    error: Optional[dict[str, Any]] = None
    
    # Provider-specific raw data (for debugging)
    provider_data: dict[str, Any] = field(default_factory=dict)
    
    def get_text_response(self) -> Optional[str]:
        """
        Extract text response from successful result.
        Handles different provider response formats.
        """
        if self.status != ResultStatus.SUCCEEDED or not self.response:
            return None
        
        # Providers will implement format-specific extraction
        # This is a helper that should work for most cases
        # Implementation needed based on provider response structure
        pass
```

---

### File: `core/base.py`

**Purpose:** Abstract base class defining provider interface.

**Class to implement:**

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

class BaseProvider(ABC):
    """
    Abstract base class for all batch providers.
    
    Each provider must implement:
    1. Conversion from unified format to provider-specific format
    2. Sending batch requests to the provider API
    3. Polling for batch status
    4. Retrieving and converting results back to unified format
    5. File management for JSONL inputs/outputs
    
    File Management:
    - All providers MUST save JSONL files to .batch_router/generated/<provider>/
    - Format: batch_<batch_id>_input_<format>.jsonl
      - _unified.jsonl: Unified format (for reference)
      - _provider.jsonl: Provider-specific format (what gets sent)
      - _output.jsonl: Raw provider output
      - _results.jsonl: Converted to unified format
    """
    
    def __init__(self, name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize provider.
        
        Args:
            name: Provider name (e.g., "openai")
            api_key: API key for the provider (if needed)
            **kwargs: Provider-specific configuration
        """
        self.name = name
        self.api_key = api_key
        self.config = kwargs
        self._validate_configuration()
    
    @abstractmethod
    def _validate_configuration(self) -> None:
        """
        Validate provider configuration.
        Should check for required credentials, tools, etc.
        Raise ValueError if configuration is invalid.
        """
        pass
    
    # ========================================================================
    # FORMAT CONVERSION (must be implemented by each provider)
    # ========================================================================
    
    @abstractmethod
    def _convert_to_provider_format(
        self, 
        requests: list[UnifiedRequest]
    ) -> list[dict[str, Any]]:
        """
        Convert unified requests to provider-specific format.
        
        This is where system_prompt gets converted to provider format:
        - OpenAI: Add as message with role="system"
        - Anthropic: Add as 'system' field in params
        - Google: Add as 'systemInstruction' in config
        - vLLM: Add as message with role="system" (OpenAI-compatible)
        
        Args:
            requests: List of unified requests
            
        Returns:
            List of provider-specific request dictionaries
        """
        pass
    
    @abstractmethod
    def _convert_from_provider_format(
        self, 
        provider_results: list[dict[str, Any]]
    ) -> list[UnifiedResult]:
        """
        Convert provider-specific results to unified format.
        
        Args:
            provider_results: Raw results from provider
            
        Returns:
            List of unified results
        """
        pass
    
    # ========================================================================
    # BATCH OPERATIONS (must be implemented by each provider)
    # ========================================================================
    
    @abstractmethod
    async def send_batch(
        self, 
        batch: UnifiedBatchMetadata
    ) -> str:
        """
        Send batch to provider.
        
        Implementation steps:
        1. Convert requests to provider format
        2. Save unified format JSONL to .batch_router/generated/<provider>/
        3. Save provider format JSONL
        4. Upload/send to provider API
        5. Return batch_id for tracking
        
        Args:
            batch: Batch metadata with unified requests
            
        Returns:
            batch_id: Unique identifier for tracking
            
        Raises:
            ValidationError: If requests are invalid
            ProviderError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_status(
        self, 
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a batch.
        
        Does NOT retrieve results - only status information.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Status information including request counts
            
        Raises:
            BatchNotFoundError: If batch_id doesn't exist
        """
        pass
    
    @abstractmethod
    async def get_results(
        self, 
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed batch.
        
        Implementation steps:
        1. Download/fetch results from provider
        2. Save raw results to .batch_router/generated/<provider>/
        3. Convert to unified format
        4. Save unified results JSONL
        5. Yield each result
        
        Args:
            batch_id: Batch identifier
            
        Yields:
            UnifiedResult objects (order NOT guaranteed)
            
        Raises:
            BatchNotCompleteError: If batch is still processing
            BatchNotFoundError: If batch_id doesn't exist
        """
        pass
    
    @abstractmethod
    async def cancel_batch(
        self, 
        batch_id: str
    ) -> bool:
        """
        Cancel a running batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            True if cancelled successfully, False if already complete
            
        Raises:
            BatchNotFoundError: If batch_id doesn't exist
        """
        pass
    
    async def list_batches(
        self, 
        limit: int = 20
    ) -> list[BatchStatusResponse]:
        """
        List recent batches.
        
        Optional method - providers may not implement if API doesn't support.
        
        Args:
            limit: Maximum number of batches to return
            
        Returns:
            List of batch status responses
        """
        raise NotImplementedError(f"{self.name} provider does not support listing batches")
    
    # ========================================================================
    # HELPER METHODS (can be overridden if needed)
    # ========================================================================
    
    def get_batch_file_path(
        self, 
        batch_id: str, 
        file_type: str
    ) -> Path:
        """
        Get path for batch file.
        
        Args:
            batch_id: Batch identifier
            file_type: One of "unified", "provider", "output", "results"
            
        Returns:
            Path to the file
        """
        from pathlib import Path
        from .constants import BATCH_DIR_PATH
        
        base_dir = Path(BATCH_DIR_PATH) / self.name
        base_dir.mkdir(parents=True, exist_ok=True)
        
        return base_dir / f"batch_{batch_id}_{file_type}.jsonl"
```

**Critical Points:**
- Providers MUST implement all abstract methods
- File management is standardized across all providers
- System prompt conversion is provider-specific
- Results must always be converted to unified format

---

## Utilities

### File: `utils/file_manager.py`

**Purpose:** JSONL file operations and directory management.

**Class to implement:**

```python
class FileManager:
    """
    Manages JSONL file operations for batch processing.
    
    All batch files are stored in:
    .batch_router/generated/<provider>/batch_<id>_<type>.jsonl
    """
    
    @staticmethod
    def ensure_batch_directory(provider: str) -> Path:
        """
        Ensure batch directory exists for provider.
        
        Creates: .batch_router/generated/<provider>/
        """
        pass
    
    @staticmethod
    async def write_jsonl(
        file_path: Path | str,
        data: list[dict[str, Any]]
    ) -> None:
        """
        Write data to JSONL file asynchronously.
        
        Each dict in data becomes one line in the file.
        """
        pass
    
    @staticmethod
    async def read_jsonl(
        file_path: Path | str
    ) -> list[dict[str, Any]]:
        """
        Read JSONL file asynchronously.
        
        Returns list of dictionaries, one per line.
        """
        pass
    
    @staticmethod
    async def stream_jsonl(
        file_path: Path | str
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream JSONL file line by line.
        
        Memory efficient for large files.
        Yields one dictionary per line.
        """
        pass
    
    @staticmethod
    def get_batch_files(
        provider: str,
        batch_id: str
    ) -> dict[str, Path]:
        """
        Get all file paths for a batch.
        
        Returns dict with keys: unified, provider, output, results
        """
        pass
```

**Implementation notes:**
- Use `aiofiles` for async file operations
- Handle file encoding (UTF-8)
- Proper error handling for file I/O

---

### File: `utils/validation.py`

**Purpose:** Request validation utilities.

**Functions to implement:**

```python
def validate_custom_ids(requests: list[UnifiedRequest]) -> None:
    """
    Ensure all custom_ids are unique within the batch.
    
    Raises:
        ValidationError: If duplicates found
    """
    pass

def validate_provider_compatibility(
    provider: str,
    requests: list[UnifiedRequest]
) -> None:
    """
    Check if requests are compatible with provider.
    
    For example:
    - Check if multimodal content is supported
    - Check if generation params are supported
    - Check batch size limits
    
    Raises:
        ValidationError: If incompatible features detected
    """
    pass

def validate_batch_size(
    provider: str,
    requests: list[UnifiedRequest]
) -> None:
    """
    Check if batch size is within provider limits.
    
    Limits:
    - OpenAI: 50,000 requests or 200 MB
    - Anthropic: 100,000 requests or 256 MB
    - Google: No hard limit documented
    - vLLM: No hard limit (local processing)
    
    Raises:
        ValidationError: If limits exceeded
    """
    pass
```

---

### File: `utils/logging.py`

**Purpose:** Structured logging configuration.

**Setup:**

```python
import logging
from pathlib import Path

def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure logger with structured output.
    
    Args:
        name: Logger name (typically module name)
        log_file: Optional file path for logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
```

**Usage in providers:**
```python
from .utils.logging import setup_logger

logger = setup_logger(__name__)
logger.info(f"Sending batch {batch_id} to {self.name}")
```

---

## Exceptions

### File: `exceptions.py`

**Purpose:** Custom exception hierarchy.

**Exceptions to implement:**

```python
class BatchRouterError(Exception):
    """Base exception for all batch router errors."""
    pass

class ProviderNotFoundError(BatchRouterError):
    """Raised when provider is not registered."""
    pass

class ValidationError(BatchRouterError):
    """Raised when request validation fails."""
    pass

class BatchTimeoutError(BatchRouterError):
    """Raised when batch doesn't complete within timeout."""
    pass

class BatchNotFoundError(BatchRouterError):
    """Raised when batch_id doesn't exist."""
    pass

class BatchNotCompleteError(BatchRouterError):
    """Raised when trying to get results from incomplete batch."""
    pass

class FileOperationError(BatchRouterError):
    """Raised when file operations fail."""
    pass

class ProviderError(BatchRouterError):
    """Raised when provider API call fails."""
    def __init__(self, provider: str, message: str, original_error: Optional[Exception] = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
```

---

## Constants

### File: `constants.py`

**Purpose:** Shared constants across the library.

**Constants to define:**

```python
from pathlib import Path

# Directory for all generated batch files
BATCH_DIR_PATH = ".batch_router/generated"

# Provider names
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GOOGLE = "google"
PROVIDER_MISTRAL = "mistral"
PROVIDER_vLLM = "vllm"

VALID_PROVIDERS = [
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_MISTRAL,
    PROVIDER_vLLM
]

# Batch size limits (used for validation)
BATCH_LIMITS = {
    PROVIDER_OPENAI: {
        "max_requests": 50_000,
        "max_size_mb": 200
    },
    PROVIDER_ANTHROPIC: {
        "max_requests": 100_000,
        "max_size_mb": 256
    },
    PROVIDER_GOOGLE: {
        "max_requests": None,  # No documented limit
        "max_size_mb": None
    },
    PROVIDER_MISTRAL: {
        "max_requests": None,  # No documented limit
        "max_size_mb": None
    },
    PROVIDER_vLLM: {
        "max_requests": None,  # Local processing
        "max_size_mb": None
    }
}

# Polling defaults
DEFAULT_POLL_INTERVAL = 60  # seconds
DEFAULT_TIMEOUT = 86400  # 24 hours
MAX_POLL_INTERVAL = 300  # 5 minutes (cap for exponential backoff)
```

---

## Main Router

### File: `router.py`

**Purpose:** Main orchestration class.

**Class to implement:**

```python
import asyncio
import time
from typing import Optional, AsyncIterator
from .core.base import BaseProvider
from .core.requests import UnifiedRequest, UnifiedBatchMetadata
from .core.responses import BatchStatusResponse, UnifiedResult
from .core.enums import BatchStatus
from .exceptions import ProviderNotFoundError, BatchTimeoutError
from .utils.logging import setup_logger
from .constants import DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT, MAX_POLL_INTERVAL

logger = setup_logger(__name__)

class BatchRouter:
    """
    Main entry point for unified batch processing.
    
    Usage:
        router = BatchRouter()
        router.register_provider(OpenAIProvider(api_key="..."))
        router.register_provider(AnthropicProvider(api_key="..."))
        
        batch_id = await router.send_batch(
            provider="openai",
            requests=[...],
            metadata={"project": "evaluation"}
        )
        
        status = await router.wait_for_completion(
            provider="openai",
            batch_id=batch_id
        )
        
        async for result in router.get_results("openai", batch_id):
            print(result.custom_id, result.status)
    """
    
    def __init__(self):
        """Initialize router with empty provider registry."""
        self.providers: dict[str, BaseProvider] = {}
        logger.info("BatchRouter initialized")
    
    def register_provider(self, provider: BaseProvider) -> None:
        """
        Register a provider implementation.
        
        Args:
            provider: Provider instance
            
        Raises:
            ValueError: If provider with same name already registered
        """
        if provider.name in self.providers:
            raise ValueError(f"Provider '{provider.name}' already registered")
        
        self.providers[provider.name] = provider
        logger.info(f"Registered provider: {provider.name}")
    
    def _get_provider(self, name: str) -> BaseProvider:
        """Get provider by name, raise if not found."""
        if name not in self.providers:
            raise ProviderNotFoundError(
                f"Provider '{name}' not registered. "
                f"Available: {list(self.providers.keys())}"
            )
        return self.providers[name]
    
    async def send_batch(
        self,
        provider: str,
        requests: list[UnifiedRequest],
        metadata: Optional[dict] = None
    ) -> str:
        """
        Send batch to specified provider.
        
        Args:
            provider: Provider name (e.g., "openai")
            requests: List of unified requests
            metadata: Optional user metadata
            
        Returns:
            batch_id for tracking
            
        Raises:
            ProviderNotFoundError: If provider not registered
            ValidationError: If requests are invalid
        """
        logger.info(f"Sending batch with {len(requests)} requests to {provider}")
        
        provider_impl = self._get_provider(provider)
        batch = UnifiedBatchMetadata(
            provider=provider,
            requests=requests,
            metadata=metadata or {}
        )
        
        batch_id = await provider_impl.send_batch(batch)
        logger.info(f"Batch {batch_id} sent to {provider}")
        return batch_id
    
    async def get_status(
        self,
        provider: str,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a batch.
        
        Args:
            provider: Provider name
            batch_id: Batch identifier
            
        Returns:
            Current batch status
        """
        provider_impl = self._get_provider(provider)
        return await provider_impl.get_status(batch_id)
    
    async def wait_for_completion(
        self,
        provider: str,
        batch_id: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
        verbose: bool = True
    ) -> BatchStatusResponse:
        """
        Poll batch status until completion with exponential backoff.
        
        Args:
            provider: Provider name
            batch_id: Batch identifier
            poll_interval: Initial polling interval in seconds
            timeout: Maximum time to wait in seconds
            verbose: Print status updates
            
        Returns:
            Final batch status
            
        Raises:
            BatchTimeoutError: If batch doesn't complete within timeout
        """
        logger.info(f"Waiting for batch {batch_id} on {provider} (timeout={timeout}s)")
        
        provider_impl = self._get_provider(provider)
        start_time = time.time()
        current_interval = poll_interval
        
        while True:
            status = await provider_impl.get_status(batch_id)
            
            if verbose:
                logger.info(
                    f"Batch {batch_id}: {status.status.value} "
                    f"({status.request_counts.succeeded}/{status.request_counts.total} succeeded)"
                )
            
            # Check if complete
            if status.is_complete():
                elapsed = time.time() - start_time
                logger.info(
                    f"Batch {batch_id} completed in {elapsed:.1f}s "
                    f"with status {status.status.value}"
                )
                return status
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise BatchTimeoutError(
                    f"Batch {batch_id} did not complete within {timeout}s"
                )
            
            # Wait with exponential backoff
            await asyncio.sleep(current_interval)
            current_interval = min(current_interval * 1.5, MAX_POLL_INTERVAL)
    
    async def get_results(
        self,
        provider: str,
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed batch.
        
        Args:
            provider: Provider name
            batch_id: Batch identifier
            
        Yields:
            UnifiedResult objects (order NOT guaranteed)
            
        Raises:
            BatchNotCompleteError: If batch still processing
        """
        logger.info(f"Retrieving results for batch {batch_id} from {provider}")
        
        provider_impl = self._get_provider(provider)
        async for result in provider_impl.get_results(batch_id):
            yield result
    
    async def cancel_batch(
        self,
        provider: str,
        batch_id: str
    ) -> bool:
        """
        Cancel a running batch.
        
        Args:
            provider: Provider name
            batch_id: Batch identifier
            
        Returns:
            True if cancelled, False if already complete
        """
        logger.info(f"Cancelling batch {batch_id} on {provider}")
        provider_impl = self._get_provider(provider)
        return await provider_impl.cancel_batch(batch_id)
    
    async def list_batches(
        self,
        provider: str,
        limit: int = 20
    ) -> list[BatchStatusResponse]:
        """
        List recent batches for a provider.
        
        Args:
            provider: Provider name
            limit: Max number to return
            
        Returns:
            List of batch statuses
            
        Note: Not all providers support this operation
        """
        provider_impl = self._get_provider(provider)
        return await provider_impl.list_batches(limit)
```

---

## Package Exports

### File: `__init__.py`

**Purpose:** Main package exports for public API.

**Exports:**

```python
"""
Batch LLM Router - Unified batch processing for multiple LLM providers.

Example usage:
    from batch_router import BatchRouter, UnifiedRequest, UnifiedMessage
    
    router = BatchRouter()
    router.register_provider(OpenAIProvider(api_key="..."))
    
    requests = [
        UnifiedRequest(
            custom_id="req1",
            model="gpt-4o",
            messages=[UnifiedMessage.from_text("user", "Hello")],
            system_prompt="You are a helpful assistant"
        )
    ]
    
    batch_id = await router.send_batch("openai", requests)
    status = await router.wait_for_completion("openai", batch_id)
    
    async for result in router.get_results("openai", batch_id):
        print(result.custom_id, result.status)
"""

# Main router
from .router import BatchRouter

# Core data models
from .core.messages import UnifiedMessage
from .core.requests import UnifiedRequest, UnifiedBatchMetadata
from .core.responses import BatchStatusResponse, UnifiedResult, RequestCounts
from .core.config import GenerationConfig
from .core.content import TextContent, ImageContent, DocumentContent
from .core.enums import BatchStatus, ResultStatus

# Providers (will be implemented)
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.mistral import MistralProvider
from .providers.vllm import vLLMProvider

# Exceptions
from .exceptions import (
    BatchRouterError,
    ProviderNotFoundError,
    ValidationError,
    BatchTimeoutError,
    BatchNotFoundError,
    BatchNotCompleteError,
    ProviderError
)

# Version
__version__ = "0.1.0"

__all__ = [
    # Main
    "BatchRouter",
    # Requests
    "UnifiedRequest",
    "UnifiedMessage",
    "GenerationConfig",
    "UnifiedBatchMetadata",
    # Content types
    "TextContent",
    "ImageContent",
    "DocumentContent",
    # Responses
    "BatchStatusResponse",
    "UnifiedResult",
    "RequestCounts",
    # Enums
    "BatchStatus",
    "ResultStatus",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MistralProvider",
    "vLLMProvider",
    # Exceptions
    "BatchRouterError",
    "ProviderNotFoundError",
    "ValidationError",
    "BatchTimeoutError",
    "BatchNotFoundError",
    "BatchNotCompleteError",
    "ProviderError",
]
```

---

## Provider Implementation Guidelines

### File Structure
Each provider file (`providers/<provider>.py`) should contain:

1. **Provider class** inheriting from `BaseProvider`
2. **Format converter functions** (private methods)
3. **Provider-specific constants** (endpoints, limits)
4. **SDK client initialization**

### Required Methods

Each provider MUST implement:

```python
class <Provider>Provider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(name="<provider>", api_key=api_key, **kwargs)
        # Initialize provider SDK client
    
    def _validate_configuration(self) -> None:
        # Check API key, credentials, tools
        pass
    
    def _convert_to_provider_format(self, requests) -> list[dict]:
        # Convert UnifiedRequest -> Provider format
        # Handle system_prompt conversion
        # Handle multimodal content
        pass
    
    def _convert_from_provider_format(self, results) -> list[UnifiedResult]:
        # Convert Provider format -> UnifiedResult
        pass
    
    async def send_batch(self, batch) -> str:
        # 1. Convert requests
        # 2. Save JSONL files
        # 3. Upload/send to API
        # 4. Return batch_id
        pass
    
    async def get_status(self, batch_id) -> BatchStatusResponse:
        # Query provider API for status
        # Convert to unified BatchStatusResponse
        pass
    
    async def get_results(self, batch_id) -> AsyncIterator[UnifiedResult]:
        # 1. Download results
        # 2. Save to JSONL
        # 3. Convert to unified format
        # 4. Yield each result
        pass
    
    async def cancel_batch(self, batch_id) -> bool:
        # Call provider cancellation API
        pass
```

### System Prompt Conversion

Each provider handles system prompts differently:

**OpenAI/Mistral/vLLM:**
```python
# Convert system_prompt to message with role="system"
if request.system_prompt:
    messages.insert(0, {
        "role": "system",
        "content": request.system_prompt if isinstance(request.system_prompt, str)
                  else "\n".join(request.system_prompt)
    })
```

**Anthropic:**
```python
# Add as separate 'system' field in params
params = {
    "model": request.model,
    "messages": [...],
    "system": request.system_prompt  # Can be string or list
}
```

**Google:**
```python
# Add as 'systemInstruction' in config
request_dict = {
    "contents": [...],
    "config": {
        "systemInstruction": {
            "parts": [{"text": request.system_prompt}]
        }
    }
}
```

### File Management

All providers MUST follow this pattern:

```python
async def send_batch(self, batch):
    batch_id = self._generate_batch_id()
    
    # Convert to provider format
    provider_requests = self._convert_to_provider_format(batch.requests)
    
    # Save unified format
    unified_path = self.get_batch_file_path(batch_id, "unified")
    await FileManager.write_jsonl(unified_path, [r.to_dict() for r in batch.requests])
    
    # Save provider format
    provider_path = self.get_batch_file_path(batch_id, "provider")
    await FileManager.write_jsonl(provider_path, provider_requests)
    
    # Send to provider API
    # ... provider-specific code ...
    
    return batch_id
```

---

## Testing Strategy

### Unit Tests
Create `tests/` directory with:
- `test_core/`: Test all data models
- `test_utils/`: Test file operations, validation
- `test_router.py`: Test BatchRouter with mock providers

### Integration Tests
- `test_providers/`: Test each provider with real APIs (requires credentials)
- Use environment variables for API keys
- Mark as `pytest.mark.integration` for optional execution

### Mock Provider
Create `tests/mock_provider.py` for testing router without real APIs:

```python
class MockProvider(BaseProvider):
    """Mock provider for testing."""
    def __init__(self):
        super().__init__("mock")
        self.batches = {}
    
    async def send_batch(self, batch):
        batch_id = f"mock_{len(self.batches)}"
        self.batches[batch_id] = {
            "status": BatchStatus.IN_PROGRESS,
            "requests": batch.requests
        }
        return batch_id
    
    # ... implement other methods ...
```

---

## Implementation Priority

### Phase 1: Core (No provider dependencies)
1. ✅ All `core/` modules
2. ✅ `utils/` modules
3. ✅ `exceptions.py`
4. ✅ `constants.py`
5. ✅ `router.py`
6. ✅ Unit tests for core

### Phase 2: OpenAI Provider
1. Implement `OpenAIProvider`
2. Test with real API
3. Verify JSONL file generation

### Phase 3: Anthropic Provider
1. Implement `AnthropicProvider`
2. Test with real API
3. Verify format conversion

### Phase 4: Google Provider
1. ✅ Implement `GoogleProvider`
2. ✅ Handle Google GenAI SDK specifics
3. ✅ Test file upload flow

### Phase 5: Mistral Provider
1. ✅ Implement `MistralProvider`
2. ✅ Handle Mistral SDK specifics
3. ✅ Test with real API

### Phase 6: vLLM Provider
1. ✅ Implement `vLLMProvider`
2. ✅ Handle subprocess execution
3. ✅ Test with local model

### Phase 7: Documentation & Examples
1. README.md with usage examples
2. API documentation
3. Example scripts

---

## Key Design Principles

1. **Async First**: All I/O is async using `asyncio` and `aiofiles`
2. **Type Safety**: Heavy use of type hints, dataclasses, and enums
3. **Provider Isolation**: Each provider is self-contained
4. **File Transparency**: All JSONL files saved for debugging
5. **Unified Interface**: Single API regardless of provider
6. **Error Handling**: Custom exceptions for different failure modes
7. **System Prompt at Request Level**: Not in messages array
8. **No Automatic Splitting**: User manages batch sizes
9. **Explicit Custom IDs**: User must provide unique identifiers
10. **Format Conversion**: Bidirectional conversion for all providers

---

## Critical Implementation Notes

### System Prompt Handling
**DO NOT** add system messages to the messages array. The `system_prompt` field in `UnifiedRequest` is separate and each provider converts it appropriately.

### Multimodal Content
`UnifiedMessage.content` is a LIST of content objects to support:
- Text only: `[TextContent(text="...")]`
- Text + image: `[TextContent(...), ImageContent(...)]`
- Multiple images: `[ImageContent(...), ImageContent(...)]`

### Result Ordering
Results from batches are **NOT** guaranteed to be in the same order as input. Always use `custom_id` to match results to requests.

### Batch Status vs Results
- `get_status()`: Returns status info (processing, succeeded, failed counts)
- `get_results()`: Returns actual response content from each request

### File Paths
All files go to: `.batch_router/generated/<provider>/batch_<id>_<type>.jsonl`
Types: `unified`, `provider`, `output`, `results`

### Async Iteration
Use `async for` when streaming results:
```python
async for result in router.get_results(provider, batch_id):
    print(result.custom_id, result.status)
```

### Provider Configuration
Providers can be configured with kwargs:
```python
OpenAIProvider(
    api_key="...",
    base_url="...",  # For custom endpoints
    timeout=120
)
```

---

## Success Criteria

A successful implementation should:

1. ✅ Pass all unit tests without provider API calls
2. ✅ Support all four providers with consistent interface
3. ✅ Generate all JSONL files in correct locations
4. ✅ Handle system prompts correctly for each provider
5. ✅ Convert multimodal content (at least text + images)
6. ✅ Implement exponential backoff polling
7. ✅ Handle errors gracefully with custom exceptions
8. ✅ Provide clear logging throughout
9. ✅ Work with async/await throughout
10. ✅ Be well-typed with proper hints

---

## Getting Started

To begin implementation:

1. **Start with Phase 1** (Core modules)
2. **Implement in this order:**
   - `core/enums.py`
   - `core/content.py`
   - `core/types.py`
   - `core/config.py`
   - `core/messages.py`
   - `core/requests.py`
   - `core/responses.py`
   - `core/base.py`
   - `exceptions.py`
   - `constants.py`
   - `utils/file_manager.py`
   - `utils/validation.py`
   - `utils/logging.py`
   - `router.py`
   - `__init__.py`

3. **Test core functionality** with mock provider
4. **Implement real providers** one at a time
5. **Test each provider** with actual API calls
