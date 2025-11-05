# Batch LLM Router - Implementation Plan

## Project Overview

**Name:** `batch_router`  
**Purpose:** Unified Python library for sending batch requests to multiple LLM providers (OpenAI, Anthropic, Google GenAI, vLLM) with a consistent interface.

**Key Features:**
- Unified request/response format across all providers
- Automatic format conversion between unified and provider-specific formats
- File-based batch management (JSONL)
- Async/await support with exponential backoff polling
- Multimodal content support (text, images, PDFs)
- System prompt at request level (not in messages)
- Explicit generation parameters (not buried in kwargs)

---

## File Structure

```
batch_router/
├── __init__.py                      # Main package exports
├── core/                            # Core data models (no business logic)
│   ├── __init__.py
│   ├── content.py                   # Content type dataclasses
│   ├── messages.py                  # UnifiedMessage
│   ├── requests.py                  # UnifiedRequest, UnifiedBatchMetadata
│   ├── responses.py                 # BatchStatusResponse, UnifiedResult
│   ├── config.py                    # GenerationConfig
│   ├── enums.py                     # Status enums
│   ├── types.py                     # Type aliases
│   └── base.py                      # BaseProvider abstract class
├── providers/                       # Provider implementations
│   ├── __init__.py
│   ├── openai.py                    # OpenAI implementation
│   ├── anthropic.py                 # Anthropic implementation
│   ├── google.py                    # Google GenAI implementation
│   └── vllm.py                      # vLLM implementation
├── router.py                        # BatchRouter orchestration
├── utils/                           # Utilities
│   ├── __init__.py
│   ├── file_manager.py              # File operations
│   ├── logging.py                   # Logging configuration
│   └── validation.py                # Validation functions
├── exceptions.py                    # Custom exceptions
└── constants.py                     # Package constants
```

---

## Implementation Order

### Phase 1: Core Data Models (No Dependencies)
1. `constants.py`
2. `exceptions.py`
3. `core/enums.py`
4. `core/content.py`
5. `core/config.py`
6. `core/messages.py`
7. `core/requests.py`
8. `core/responses.py`
9. `core/types.py`
10. `core/base.py`

### Phase 2: Utilities
11. `utils/logging.py`
12. `utils/validation.py`
13. `utils/file_manager.py`

### Phase 3: Provider Implementations (Can be parallel)
14. `providers/openai.py`
15. `providers/anthropic.py`
16. `providers/google.py`
17. `providers/vllm.py`

### Phase 4: Router & Package
18. `router.py`
19. `__init__.py` (package level)
20. `core/__init__.py`
21. `providers/__init__.py`
22. `utils/__init__.py`

---

## Detailed Module Specifications

### 1. `constants.py`

**Purpose:** Define package-wide constants.

**Contents:**
```python
from pathlib import Path

# File management
BATCH_DIR_NAME = ".batch_router"
GENERATED_DIR_NAME = "generated"
BATCH_DIR_PATH = Path.cwd() / BATCH_DIR_NAME / GENERATED_DIR_NAME

# Provider names
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GOOGLE = "google"
PROVIDER_VLLM = "vllm"

SUPPORTED_PROVIDERS = [PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GOOGLE, PROVIDER_VLLM]

# File naming patterns
INPUT_FILE_PATTERN = "batch_{batch_id}_input.jsonl"
OUTPUT_FILE_PATTERN = "batch_{batch_id}_output.jsonl"
UNIFIED_FILE_PATTERN = "batch_{batch_id}_unified.jsonl"

# Polling defaults
DEFAULT_POLL_INTERVAL = 60  # seconds
DEFAULT_TIMEOUT = 86400  # 24 hours
MAX_POLL_INTERVAL = 300  # 5 minutes (cap for exponential backoff)
BACKOFF_MULTIPLIER = 1.5

# Provider limits (for validation)
OPENAI_MAX_REQUESTS = 50000
OPENAI_MAX_FILE_SIZE_MB = 200
ANTHROPIC_MAX_REQUESTS = 100000
ANTHROPIC_MAX_FILE_SIZE_MB = 256
GOOGLE_MAX_REQUESTS = 1000000  # No documented limit, estimate
VLLM_MAX_REQUESTS = None  # No hard limit for local
```

**Tests:**
- Verify all constants are accessible
- Verify Path objects are correctly constructed

---

### 2. `exceptions.py`

**Purpose:** Define custom exception hierarchy.

**Classes:**

```python
class BatchRouterError(Exception):
    """Base exception for all batch_router errors"""
    pass

class ProviderNotFoundError(BatchRouterError):
    """Raised when specified provider is not registered"""
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        super().__init__(f"Provider '{provider_name}' not found")

class ValidationError(BatchRouterError):
    """Raised when request validation fails"""
    pass

class BatchTimeoutError(BatchRouterError):
    """Raised when batch processing exceeds timeout"""
    def __init__(self, batch_id: str, timeout: int):
        self.batch_id = batch_id
        self.timeout = timeout
        super().__init__(f"Batch {batch_id} timed out after {timeout}s")

class BatchFailedError(BatchRouterError):
    """Raised when batch processing fails"""
    def __init__(self, batch_id: str, error_details: dict):
        self.batch_id = batch_id
        self.error_details = error_details
        super().__init__(f"Batch {batch_id} failed: {error_details}")

class FileOperationError(BatchRouterError):
    """Raised when file operations fail"""
    pass

class ProviderAPIError(BatchRouterError):
    """Raised when provider API returns an error"""
    def __init__(self, provider: str, status_code: int, message: str):
        self.provider = provider
        self.status_code = status_code
        self.message = message
        super().__init__(f"{provider} API error ({status_code}): {message}")

class UnsupportedContentTypeError(ValidationError):
    """Raised when content type is not supported by provider"""
    pass
```

**Tests:**
- Instantiate each exception type
- Verify error messages contain expected information
- Test inheritance chain

---

### 3. `core/enums.py`

**Purpose:** Define enumerations for status types.

**Enums:**

```python
from enum import Enum

class BatchStatus(Enum):
    """Unified batch status across all providers"""
    VALIDATING = "validating"      # OpenAI: validating
    IN_PROGRESS = "in_progress"    # All: in_progress/processing
    COMPLETED = "completed"        # All: completed/ended/succeeded
    FAILED = "failed"              # All: failed
    CANCELLED = "cancelled"        # All: cancelled/canceled
    EXPIRED = "expired"            # OpenAI, Anthropic: expired
    CANCELLING = "cancelling"      # Transition state

class ResultStatus(Enum):
    """Status of individual request within a batch"""
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
```

**Provider Mapping Reference:**
- OpenAI: validating, failed, in_progress, finalizing, completed, expired, cancelling, cancelled
- Anthropic: in_progress, ended (with result.type: succeeded/errored/cancelled/expired)
- Google: JOB_STATE_PENDING, JOB_STATE_RUNNING, JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_CANCELLED
- vLLM: Determined by output file existence (no status API)

**Tests:**
- Verify enum values
- Test string conversion

---

### 4. `core/content.py`

**Purpose:** Define content type dataclasses for multimodal messages.

**Classes:**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class TextContent:
    """Text content in a message"""
    type: Literal["text"] = "text"
    text: str = ""
    
    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")

@dataclass
class ImageContent:
    """Image content in a message (base64 or URL)"""
    type: Literal["image"] = "image"
    source_type: Literal["base64", "url"]  # "url" includes file_uri for Google
    media_type: str  # "image/jpeg", "image/png", "image/gif", "image/webp"
    data: str  # base64 string or URL/URI
    
    def __post_init__(self):
        if self.source_type not in ["base64", "url"]:
            raise ValueError("source_type must be 'base64' or 'url'")
        if not self.media_type.startswith("image/"):
            raise ValueError(f"Invalid image media_type: {self.media_type}")

@dataclass
class DocumentContent:
    """PDF/document content in a message"""
    type: Literal["document"] = "document"
    source_type: Literal["base64", "url"]
    media_type: str  # "application/pdf"
    data: str  # base64 string or URL/URI
    
    def __post_init__(self):
        if self.source_type not in ["base64", "url"]:
            raise ValueError("source_type must be 'base64' or 'url'")
        # Only Anthropic supports documents in batch mode
```

**Tests:**
- Create instances of each content type
- Validate __post_init__ raises errors for invalid data
- Test serialization/deserialization

---

### 5. `core/config.py`

**Purpose:** Define generation configuration dataclass.

**Class:**

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GenerationConfig:
    """
    Common generation parameters across providers.
    Providers will map these to their specific names.
    
    Note: Not all parameters are supported by all providers.
    Provider implementations will ignore unsupported parameters.
    """
    # Universal parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None  # REQUIRED for Anthropic
    top_p: Optional[float] = None
    top_k: Optional[int] = None  # Not supported by OpenAI
    
    # Stop sequences
    stop_sequences: Optional[list[str]] = None
    
    # Penalties (OpenAI and Google only)
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    
    def __post_init__(self):
        """Validate parameter ranges"""
        if self.temperature is not None:
            if not (0 <= self.temperature <= 2):
                raise ValueError("temperature must be between 0 and 2")
        
        if self.max_tokens is not None:
            if self.max_tokens < 1:
                raise ValueError("max_tokens must be positive")
        
        if self.top_p is not None:
            if not (0 <= self.top_p <= 1):
                raise ValueError("top_p must be between 0 and 1")
        
        if self.top_k is not None:
            if self.top_k < 1:
                raise ValueError("top_k must be positive")
        
        if self.presence_penalty is not None:
            if not (-2 <= self.presence_penalty <= 2):
                raise ValueError("presence_penalty must be between -2 and 2")
        
        if self.frequency_penalty is not None:
            if not (-2 <= self.frequency_penalty <= 2):
                raise ValueError("frequency_penalty must be between -2 and 2")
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
```

**Tests:**
- Create config with valid parameters
- Test validation errors for out-of-range values
- Test to_dict() excludes None values

---

### 6. `core/messages.py`

**Purpose:** Define UnifiedMessage dataclass.

**Class:**

```python
from dataclasses import dataclass, field
from typing import Literal, Any
from .content import TextContent, ImageContent, DocumentContent

# Type alias for content union
MessageContent = TextContent | ImageContent | DocumentContent

@dataclass
class UnifiedMessage:
    """
    Unified message format across providers.
    
    IMPORTANT: System messages should NOT be in messages array.
    Use UnifiedRequest.system_prompt instead.
    Only user/assistant roles are allowed here.
    """
    role: Literal["user", "assistant"]
    content: list[MessageContent]
    
    # Provider-specific extensions (for advanced features)
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate message"""
        if self.role not in ["user", "assistant"]:
            raise ValueError(f"Invalid role '{self.role}'. Use 'user' or 'assistant' only. "
                           "System prompts go in UnifiedRequest.system_prompt.")
        
        if not self.content:
            raise ValueError("content cannot be empty")
        
        if not all(isinstance(c, (TextContent, ImageContent, DocumentContent)) 
                   for c in self.content):
            raise TypeError("All content items must be TextContent, ImageContent, or DocumentContent")
    
    @classmethod
    def from_text(cls, role: str, text: str, **provider_kwargs):
        """
        Convenience constructor for text-only messages.
        
        Example:
            msg = UnifiedMessage.from_text("user", "Hello, world!")
        """
        return cls(
            role=role,
            content=[TextContent(text=text)],
            provider_kwargs=provider_kwargs
        )
    
    def is_text_only(self) -> bool:
        """Check if message contains only text content"""
        return len(self.content) == 1 and isinstance(self.content[0], TextContent)
    
    def get_text(self) -> str:
        """Get text content if text-only message, else raise error"""
        if not self.is_text_only():
            raise ValueError("Message is not text-only")
        return self.content[0].text
```

**Tests:**
- Create text-only message using from_text()
- Create multimodal message with text + image
- Validate role restrictions
- Test is_text_only() and get_text()

---

### 7. `core/requests.py`

**Purpose:** Define request dataclasses.

**Classes:**

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from .messages import UnifiedMessage
from .config import GenerationConfig

@dataclass
class UnifiedRequest:
    """
    Unified request format for batch processing.
    
    System prompt is at request level, NOT in messages array.
    This ensures consistency across providers with different system prompt handling.
    """
    custom_id: str  # User MUST provide this (used to match results)
    model: str  # Provider-specific model ID (e.g., "gpt-4", "claude-sonnet-4-5")
    messages: list[UnifiedMessage]
    
    # System prompt at request level (can be string or list of strings)
    system_prompt: Optional[str | list[str]] = None
    
    # Generation parameters (explicit, not in provider_kwargs)
    generation_config: Optional[GenerationConfig] = None
    
    # Provider-specific extensions (for advanced features like tools, caching, etc.)
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate request"""
        if not self.custom_id:
            raise ValueError("custom_id is required and cannot be empty")
        
        if not self.model:
            raise ValueError("model is required and cannot be empty")
        
        if not self.messages:
            raise ValueError("messages cannot be empty")
        
        if not isinstance(self.messages, list):
            raise TypeError("messages must be a list")
        
        # Ensure all items are UnifiedMessage instances
        if not all(isinstance(msg, UnifiedMessage) for msg in self.messages):
            raise TypeError("All messages must be UnifiedMessage instances")
        
        # Validate no system role in messages
        for msg in self.messages:
            if msg.role not in ["user", "assistant"]:
                raise ValueError(
                    f"Invalid role '{msg.role}' in messages. "
                    "Use system_prompt field for system instructions."
                )
        
        # Validate alternating user/assistant pattern (optional, provider-dependent)
        # OpenAI and Anthropic require this, Google does not
        # We'll validate in provider-specific converters
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "custom_id": self.custom_id,
            "model": self.model,
            "messages": [
                {
                    "role": msg.role,
                    "content": [
                        {"type": c.type, **{k: v for k, v in c.__dict__.items() if k != "type"}}
                        for c in msg.content
                    ],
                    "provider_kwargs": msg.provider_kwargs
                }
                for msg in self.messages
            ],
            "system_prompt": self.system_prompt,
            "generation_config": self.generation_config.to_dict() if self.generation_config else None,
            "provider_kwargs": self.provider_kwargs
        }


@dataclass
class UnifiedBatchMetadata:
    """
    Batch metadata before sending to provider.
    This is NOT a response - just what we're preparing to send.
    """
    provider: str  # "openai", "anthropic", "google", "vllm"
    requests: list[UnifiedRequest]
    metadata: dict[str, Any] = field(default_factory=dict)  # User-defined metadata
    
    def __post_init__(self):
        """Validate batch metadata"""
        from ..constants import SUPPORTED_PROVIDERS
        
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        if not self.requests:
            raise ValueError("requests cannot be empty")
        
        # Check for duplicate custom_ids
        custom_ids = [req.custom_id for req in self.requests]
        duplicates = [cid for cid in custom_ids if custom_ids.count(cid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate custom_ids found: {set(duplicates)}")
    
    def get_request_count(self) -> int:
        """Get total number of requests in batch"""
        return len(self.requests)
```

**Tests:**
- Create valid UnifiedRequest
- Test validation errors (missing fields, invalid roles)
- Test duplicate custom_id detection in UnifiedBatchMetadata
- Test to_dict() serialization

---

### 8. `core/responses.py`

**Purpose:** Define response dataclasses.

**Classes:**

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from .enums import BatchStatus, ResultStatus

@dataclass
class RequestCounts:
    """Breakdown of request statuses in a batch"""
    total: int
    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    cancelled: int = 0
    expired: int = 0
    
    def __post_init__(self):
        """Validate counts"""
        if self.total < 0:
            raise ValueError("total must be non-negative")
        
        sum_counts = (self.processing + self.succeeded + self.errored + 
                     self.cancelled + self.expired)
        if sum_counts > self.total:
            raise ValueError(f"Sum of status counts ({sum_counts}) exceeds total ({self.total})")


@dataclass
class BatchStatusResponse:
    """
    Response from checking batch status.
    Does NOT contain results - only status information.
    Use get_results() to retrieve actual responses.
    """
    batch_id: str
    provider: str
    status: BatchStatus
    request_counts: RequestCounts
    
    # Timestamps (ISO 8601 format)
    created_at: str
    completed_at: Optional[str] = None
    expires_at: Optional[str] = None
    
    # Provider-specific fields (raw provider data)
    provider_data: dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if batch has finished processing"""
        return self.status in [
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED
        ]
    
    def is_successful(self) -> bool:
        """Check if batch completed successfully"""
        return self.status == BatchStatus.COMPLETED


@dataclass
class UnifiedResult:
    """
    Individual request result within a batch.
    Order is NOT guaranteed - use custom_id to match with requests.
    """
    custom_id: str
    status: ResultStatus
    
    # If succeeded: contains full response
    response: Optional[dict[str, Any]] = None
    
    # If errored: contains error details
    error: Optional[dict[str, Any]] = None
    
    # Provider-specific data (raw result from provider)
    provider_data: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result"""
        if self.status == ResultStatus.SUCCEEDED and self.response is None:
            raise ValueError("response must be provided for succeeded results")
        
        if self.status == ResultStatus.ERRORED and self.error is None:
            raise ValueError("error must be provided for errored results")
    
    def is_successful(self) -> bool:
        """Check if request succeeded"""
        return self.status == ResultStatus.SUCCEEDED
    
    def get_content(self) -> Optional[str]:
        """
        Extract text content from successful response.
        Provider-agnostic helper method.
        """
        if not self.is_successful() or not self.response:
            return None
        
        # Try to extract content from common response structures
        # OpenAI: response['choices'][0]['message']['content']
        # Anthropic: response['content'][0]['text']
        # Google: response['candidates'][0]['content']['parts'][0]['text']
        
        if 'choices' in self.response:  # OpenAI format
            return self.response['choices'][0]['message']['content']
        elif 'content' in self.response:  # Anthropic format
            if isinstance(self.response['content'], list):
                return self.response['content'][0]['text']
        elif 'candidates' in self.response:  # Google format
            return self.response['candidates'][0]['content']['parts'][0]['text']
        
        return None
```

**Tests:**
- Create BatchStatusResponse with various statuses
- Test is_complete() and is_successful()
- Test RequestCounts validation
- Test UnifiedResult validation
- Test get_content() extraction for different provider formats

---

### 9. `core/types.py`

**Purpose:** Define type aliases for better IDE support.

**Contents:**

```python
from typing import TypeAlias
from .content import TextContent, ImageContent, DocumentContent

# Content type union
MessageContentType: TypeAlias = TextContent | ImageContent | DocumentContent

# System prompt can be string or list of strings
SystemPromptType: TypeAlias = str | list[str]
```

**Tests:**
- Type checking with mypy/pyright

---

### 10. `core/base.py`

**Purpose:** Define abstract base class for providers.

**Class:**

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from .requests import UnifiedBatchMetadata
from .responses import BatchStatusResponse, UnifiedResult

class BaseProvider(ABC):
    """
    Abstract base class for all batch providers.
    
    Each provider must implement:
    1. Format conversion (unified -> provider-specific)
    2. API communication (send, poll, retrieve)
    3. Result parsing (provider-specific -> unified)
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize provider.
        
        Args:
            api_key: API key for the provider (if required)
            **kwargs: Provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')"""
        pass
    
    @abstractmethod
    async def send_batch(self, batch: UnifiedBatchMetadata) -> str:
        """
        Send a batch of requests to the provider.
        
        Steps:
        1. Convert unified requests to provider format
        2. Write to JSONL file (if needed)
        3. Upload file to provider (if needed)
        4. Create batch job via provider API
        5. Save generated files to .batch_router/generated/<provider>/
        
        Args:
            batch: UnifiedBatchMetadata with requests
        
        Returns:
            batch_id: Unique identifier to track this batch
        
        Raises:
            ProviderAPIError: If provider API returns an error
            ValidationError: If requests don't meet provider requirements
        """
        pass
    
    @abstractmethod
    async def get_status(self, batch_id: str) -> BatchStatusResponse:
        """
        Get current status of a batch.
        
        Args:
            batch_id: Batch identifier returned by send_batch()
        
        Returns:
            BatchStatusResponse with current status
        
        Raises:
            ProviderAPIError: If provider API returns an error
        """
        pass
    
    @abstractmethod
    async def get_results(self, batch_id: str) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed batch.
        
        Steps:
        1. Download results file from provider
        2. Parse provider-specific format
        3. Convert to UnifiedResult format
        4. Yield results one by one
        
        Args:
            batch_id: Batch identifier
        
        Yields:
            UnifiedResult objects for each request in batch
        
        Note: Order is NOT guaranteed to match input order.
        Always use custom_id to match results to requests.
        
        Raises:
            ProviderAPIError: If provider API returns an error
            FileOperationError: If result file cannot be read
        """
        pass
    
    @abstractmethod
    async def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel a running batch.
        
        Args:
            batch_id: Batch identifier
        
        Returns:
            True if cancellation successful
        
        Raises:
            ProviderAPIError: If provider API returns an error
        """
        pass
    
    async def list_batches(self, limit: int = 20) -> list[BatchStatusResponse]:
        """
        List recent batches (optional, not all providers support this).
        
        Args:
            limit: Maximum number of batches to return
        
        Returns:
            List of BatchStatusResponse objects
        """
        raise NotImplementedError(f"{self.name} does not support listing batches")
    
    # Helper methods that can be overridden
    
    def validate_batch(self, batch: UnifiedBatchMetadata) -> None:
        """
        Validate batch before sending (provider-specific constraints).
        Raises ValidationError if batch doesn't meet requirements.
        """
        pass
    
    def get_provider_directory(self) -> Path:
        """Get directory for this provider's generated files"""
        from ..constants import BATCH_DIR_PATH
        provider_dir = BATCH_DIR_PATH / self.name
        provider_dir.mkdir(parents=True, exist_ok=True)
        return provider_dir
```

**Tests:**
- Cannot instantiate abstract class
- Test get_provider_directory() creates correct path

---

### 11. `utils/logging.py`

**Purpose:** Configure structured logging for the package.

**Implementation:**

```python
import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "batch_router",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logs
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format: [2025-01-01 12:00:00] INFO - batch_router.openai - Message
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Package-level logger
logger = setup_logger()
```

**Usage in other modules:**
```python
from ..utils.logging import logger

logger.info("Sending batch to OpenAI")
logger.error("Failed to parse response", exc_info=True)
```

---

### 12. `utils/validation.py`

**Purpose:** Validate requests before sending.

**Functions:**

```python
from typing import Optional
from ..core.requests import UnifiedRequest, UnifiedBatchMetadata
from ..core.messages import UnifiedMessage
from ..core.content import ImageContent, DocumentContent
from ..exceptions import ValidationError

def validate_message_pattern(messages: list[UnifiedMessage], provider: str) -> None:
    """
    Validate message alternation pattern (provider-specific).
    
    OpenAI/Anthropic: Must alternate user/assistant, starting with user
    Google: No strict alternation required
    """
    if provider in ["openai", "anthropic"]:
        if not messages:
            raise ValidationError("Messages cannot be empty")
        
        # Must start with user
        if messages[0].role != "user":
            raise ValidationError(f"{provider} requires first message to be from user")
        
        # Check alternation
        for i in range(len(messages) - 1):
            if messages[i].role == messages[i + 1].role:
                raise ValidationError(
                    f"{provider} requires alternating user/assistant messages"
                )

def validate_content_support(request: UnifiedRequest, provider: str) -> None:
    """
    Validate that content types are supported by provider.
    
    Text: All providers
    Images: All providers (in batch mode)
    Documents: Only Anthropic
    """
    for message in request.messages:
        for content in message.content:
            if isinstance(content, DocumentContent):
                if provider != "anthropic":
                    raise ValidationError(
                        f"Document content not supported by {provider} in batch mode"
                    )
            
            if isinstance(content, ImageContent):
                # All providers support images, but validate format
                if content.source_type not in ["base64", "url"]:
                    raise ValidationError(
                        f"Invalid image source_type: {content.source_type}"
                    )

def validate_generation_config(request: UnifiedRequest, provider: str) -> None:
    """
    Validate generation config parameters for provider.
    
    Anthropic: max_tokens is REQUIRED
    OpenAI: top_k not supported
    """
    config = request.generation_config
    if not config:
        if provider == "anthropic":
            raise ValidationError("Anthropic requires generation_config.max_tokens")
        return
    
    if provider == "anthropic" and config.max_tokens is None:
        raise ValidationError("Anthropic requires generation_config.max_tokens")
    
    if provider == "openai" and config.top_k is not None:
        raise ValidationError("OpenAI does not support top_k parameter")

def validate_batch_size(batch: UnifiedBatchMetadata) -> None:
    """
    Validate batch doesn't exceed provider limits.
    """
    from ..constants import (
        OPENAI_MAX_REQUESTS, ANTHROPIC_MAX_REQUESTS,
        PROVIDER_OPENAI, PROVIDER_ANTHROPIC
    )
    
    request_count = batch.get_request_count()
    
    if batch.provider == PROVIDER_OPENAI:
        if request_count > OPENAI_MAX_REQUESTS:
            raise ValidationError(
                f"OpenAI batch limit is {OPENAI_MAX_REQUESTS} requests, "
                f"got {request_count}"
            )
    
    elif batch.provider == PROVIDER_ANTHROPIC:
        if request_count > ANTHROPIC_MAX_REQUESTS:
            raise ValidationError(
                f"Anthropic batch limit is {ANTHROPIC_MAX_REQUESTS} requests, "
                f"got {request_count}"
            )

def validate_request(request: UnifiedRequest, provider: str) -> None:
    """
    Run all validations on a single request.
    """
    validate_message_pattern(request.messages, provider)
    validate_content_support(request, provider)
    validate_generation_config(request, provider)

def validate_batch(batch: UnifiedBatchMetadata) -> None:
    """
    Run all validations on a batch.
    """
    validate_batch_size(batch)
    
    for request in batch.requests:
        validate_request(request, batch.provider)
```

**Tests:**
- Test each validation function with valid/invalid inputs
- Test provider-specific validation rules

---

### 13. `utils/file_manager.py`

**Purpose:** Manage JSONL files and directory structure.

**Class:**

```python
import json
import uuid
from pathlib import Path
from typing import Any, Iterator
from ..constants import (
    BATCH_DIR_PATH, INPUT_FILE_PATTERN, OUTPUT_FILE_PATTERN,
    UNIFIED_FILE_PATTERN
)
from ..exceptions import FileOperationError
from ..utils.logging import logger

class FileManager:
    """
    Manages batch file operations:
    - Directory structure (.batch_router/generated/<provider>/)
    - JSONL file creation and parsing
    - File downloads from providers
    """
    
    @staticmethod
    def ensure_directory_structure() -> None:
        """Create .batch_router/generated/ directory structure"""
        try:
            BATCH_DIR_PATH.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {BATCH_DIR_PATH}")
        except Exception as e:
            raise FileOperationError(f"Failed to create directory: {e}")
    
    @staticmethod
    def get_provider_directory(provider: str) -> Path:
        """Get directory for provider's generated files"""
        provider_dir = BATCH_DIR_PATH / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        return provider_dir
    
    @staticmethod
    def generate_batch_id() -> str:
        """Generate unique batch ID"""
        return f"batch_{uuid.uuid4().hex[:16]}"
    
    @staticmethod
    def get_input_file_path(provider: str, batch_id: str) -> Path:
        """Get path for input JSONL file"""
        provider_dir = FileManager.get_provider_directory(provider)
        filename = INPUT_FILE_PATTERN.format(batch_id=batch_id)
        return provider_dir / filename
    
    @staticmethod
    def get_output_file_path(provider: str, batch_id: str) -> Path:
        """Get path for output JSONL file"""
        provider_dir = FileManager.get_provider_directory(provider)
        filename = OUTPUT_FILE_PATTERN.format(batch_id=batch_id)
        return provider_dir / filename
    
    @staticmethod
    def get_unified_file_path(provider: str, batch_id: str) -> Path:
        """Get path for unified format file"""
        provider_dir = FileManager.get_provider_directory(provider)
        filename = UNIFIED_FILE_PATTERN.format(batch_id=batch_id)
        return provider_dir / filename
    
    @staticmethod
    def write_jsonl(file_path: Path, data: list[dict]) -> None:
        """
        Write list of dicts to JSONL file.
        
        Args:
            file_path: Output file path
            data: List of dictionaries to write
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Wrote {len(data)} lines to {file_path}")
        except Exception as e:
            raise FileOperationError(f"Failed to write JSONL file: {e}")
    
    @staticmethod
    def read_jsonl(file_path: Path) -> Iterator[dict]:
        """
        Read JSONL file line by line.
        
        Args:
            file_path: Input file path
        
        Yields:
            Parsed JSON objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            raise FileOperationError(f"File not found: {file_path}")
        except Exception as e:
            raise FileOperationError(f"Failed to read JSONL file: {e}")
    
    @staticmethod
    async def download_file(url: str, destination: Path) -> None:
        """
        Download file from URL to destination.
        
        Args:
            url: URL to download from
            destination: Local file path
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    content = await response.read()
                    
                    with open(destination, 'wb') as f:
                        f.write(content)
                    
                    logger.info(f"Downloaded {len(content)} bytes to {destination}")
        except Exception as e:
            raise FileOperationError(f"Failed to download file: {e}")
    
    @staticmethod
    def file_size_mb(file_path: Path) -> float:
        """Get file size in MB"""
        return file_path.stat().st_size / (1024 * 1024)
```

**Tests:**
- Test directory creation
- Test JSONL write/read roundtrip
- Test batch_id generation uniqueness
- Test file path construction

---

### 14. `providers/openai.py`

**Purpose:** OpenAI Batch API implementation.

**Key Requirements:**
- Upload JSONL file to OpenAI Files API
- Create batch job referencing uploaded file
- Poll batch status
- Download output file
- Convert between unified and OpenAI formats

**OpenAI Format:**
```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4", "messages": [...]}}
```

**Implementation Outline:**

```python
import asyncio
from openai import AsyncOpenAI
from ..core.base import BaseProvider
from ..core.requests import UnifiedBatchMetadata
from ..core.responses import BatchStatusResponse, UnifiedResult, RequestCounts
from ..core.enums import BatchStatus, ResultStatus
from ..utils.file_manager import FileManager
from ..utils.logging import logger
from ..exceptions import ProviderAPIError

class OpenAIProvider(BaseProvider):
    """OpenAI Batch API implementation"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
    
    @property
    def name(self) -> str:
        return "openai"
    
    # Conversion methods
    
    def _convert_system_prompt(self, system_prompt: str | list[str] | None) -> list[dict]:
        """Convert system prompt to OpenAI message format"""
        if not system_prompt:
            return []
        
        if isinstance(system_prompt, str):
            return [{"role": "system", "content": system_prompt}]
        else:
            # Multiple system prompts - combine into one
            combined = "\n\n".join(system_prompt)
            return [{"role": "system", "content": combined}]
    
    def _convert_content(self, content_list: list) -> str | list[dict]:
        """Convert unified content to OpenAI format"""
        # If text-only, return string
        if len(content_list) == 1 and content_list[0].type == "text":
            return content_list[0].text
        
        # Multimodal: return list of content objects
        result = []
        for content in content_list:
            if content.type == "text":
                result.append({"type": "text", "text": content.text})
            elif content.type == "image":
                if content.source_type == "base64":
                    result.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content.media_type};base64,{content.data}"
                        }
                    })
                else:  # url
                    result.append({
                        "type": "image_url",
                        "image_url": {"url": content.data}
                    })
        return result
    
    def _convert_to_openai_format(self, batch: UnifiedBatchMetadata) -> list[dict]:
        """Convert UnifiedBatchMetadata to OpenAI JSONL format"""
        openai_requests = []
        
        for request in batch.requests:
            # Build messages array
            messages = []
            
            # Add system prompt as first message (if present)
            messages.extend(self._convert_system_prompt(request.system_prompt))
            
            # Add user/assistant messages
            for msg in request.messages:
                messages.append({
                    "role": msg.role,
                    "content": self._convert_content(msg.content)
                })
            
            # Build body
            body = {
                "model": request.model,
                "messages": messages
            }
            
            # Add generation config
            if request.generation_config:
                config = request.generation_config
                if config.temperature is not None:
                    body["temperature"] = config.temperature
                if config.max_tokens is not None:
                    body["max_tokens"] = config.max_tokens
                if config.top_p is not None:
                    body["top_p"] = config.top_p
                if config.stop_sequences is not None:
                    body["stop"] = config.stop_sequences
                if config.presence_penalty is not None:
                    body["presence_penalty"] = config.presence_penalty
                if config.frequency_penalty is not None:
                    body["frequency_penalty"] = config.frequency_penalty
            
            # Add provider_kwargs
            body.update(request.provider_kwargs)
            
            # Build OpenAI request
            openai_requests.append({
                "custom_id": request.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            })
        
        return openai_requests
    
    async def send_batch(self, batch: UnifiedBatchMetadata) -> str:
        """Send batch to OpenAI"""
        # Generate batch ID
        batch_id = FileManager.generate_batch_id()
        
        # Convert to OpenAI format
        openai_requests = self._convert_to_openai_format(batch)
        
        # Write to local JSONL file
        input_file_path = FileManager.get_input_file_path(self.name, batch_id)
        FileManager.write_jsonl(input_file_path, openai_requests)
        
        # Upload to OpenAI
        try:
            with open(input_file_path, 'rb') as f:
                file_response = await self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            logger.info(f"Uploaded file to OpenAI: {file_response.id}")
            
            # Create batch
            batch_response = await self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=batch.metadata
            )
            
            logger.info(f"Created OpenAI batch: {batch_response.id}")
            
            return batch_response.id
            
        except Exception as e:
            raise ProviderAPIError("openai", 0, str(e))
    
    async def get_status(self, batch_id: str) -> BatchStatusResponse:
        """Get batch status from OpenAI"""
        try:
            batch = await self.client.batches.retrieve(batch_id)
            
            # Map OpenAI status to unified status
            status_map = {
                "validating": BatchStatus.VALIDATING,
                "failed": BatchStatus.FAILED,
                "in_progress": BatchStatus.IN_PROGRESS,
                "finalizing": BatchStatus.IN_PROGRESS,
                "completed": BatchStatus.COMPLETED,
                "expired": BatchStatus.EXPIRED,
                "cancelling": BatchStatus.CANCELLING,
                "cancelled": BatchStatus.CANCELLED
            }
            
            status = status_map.get(batch.status, BatchStatus.IN_PROGRESS)
            
            # Build request counts
            counts = RequestCounts(
                total=batch.request_counts.total if batch.request_counts else 0,
                processing=batch.request_counts.processing if batch.request_counts else 0,
                succeeded=batch.request_counts.completed if batch.request_counts else 0,
                errored=batch.request_counts.failed if batch.request_counts else 0,
                cancelled=0,  # OpenAI doesn't separate cancelled
                expired=0     # OpenAI doesn't separate expired
            )
            
            return BatchStatusResponse(
                batch_id=batch_id,
                provider=self.name,
                status=status,
                request_counts=counts,
                created_at=str(batch.created_at),
                completed_at=str(batch.completed_at) if batch.completed_at else None,
                expires_at=str(batch.expires_at) if batch.expires_at else None,
                provider_data=batch.model_dump()
            )
            
        except Exception as e:
            raise ProviderAPIError("openai", 0, str(e))
    
    async def get_results(self, batch_id: str) -> AsyncIterator[UnifiedResult]:
        """Get results from OpenAI"""
        try:
            # Get batch info
            batch = await self.client.batches.retrieve(batch_id)
            
            if not batch.output_file_id:
                logger.warning(f"No output file for batch {batch_id}")
                return
            
            # Download output file
            output_file_path = FileManager.get_output_file_path(self.name, batch_id)
            file_content = await self.client.files.content(batch.output_file_id)
            
            with open(output_file_path, 'wb') as f:
                f.write(file_content.content)
            
            logger.info(f"Downloaded results to {output_file_path}")
            
            # Parse and yield results
            for line in FileManager.read_jsonl(output_file_path):
                custom_id = line.get("custom_id")
                response_data = line.get("response")
                error_data = line.get("error")
                
                if error_data:
                    yield UnifiedResult(
                        custom_id=custom_id,
                        status=ResultStatus.ERRORED,
                        error=error_data,
                        provider_data=line
                    )
                else:
                    yield UnifiedResult(
                        custom_id=custom_id,
                        status=ResultStatus.SUCCEEDED,
                        response=response_data.get("body"),
                        provider_data=line
                    )
        
        except Exception as e:
            raise ProviderAPIError("openai", 0, str(e))
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel batch"""
        try:
            await self.client.batches.cancel(batch_id)
            logger.info(f"Cancelled batch {batch_id}")
            return True
        except Exception as e:
            raise ProviderAPIError("openai", 0, str(e))
```

**Tests:**
- Test format conversion (unified -> OpenAI)
- Test send_batch() flow
- Test status polling
- Test results parsing
- Mock OpenAI API responses

---

### 15. `providers/anthropic.py`

**Purpose:** Anthropic Message Batches API implementation.

**Key Requirements:**
- Send requests directly (no file upload)
- System prompt is separate field
- max_tokens is REQUIRED
- Stream results via API (no file download)

**Anthropic Format:**
```python
{
    "requests": [
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "system": "You are helpful",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        }
    ]
}
```

**Implementation Outline:**

```python
from anthropic import AsyncAnthropic
from ..core.base import BaseProvider

class AnthropicProvider(BaseProvider):
    """Anthropic Message Batches API implementation"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    def _convert_to_anthropic_format(self, batch: UnifiedBatchMetadata) -> list[dict]:
        """Convert to Anthropic format"""
        anthropic_requests = []
        
        for request in batch.requests:
            # Build params
            params = {
                "model": request.model,
                "messages": [
                    {
                        "role": msg.role,
                        "content": self._convert_content(msg.content)
                    }
                    for msg in request.messages
                ]
            }
            
            # Add system prompt (separate field!)
            if request.system_prompt:
                if isinstance(request.system_prompt, list):
                    params["system"] = request.system_prompt
                else:
                    params["system"] = request.system_prompt
            
            # Add generation config (max_tokens is REQUIRED)
            if request.generation_config:
                config = request.generation_config
                if config.max_tokens:
                    params["max_tokens"] = config.max_tokens
                else:
                    raise ValueError("Anthropic requires max_tokens")
                
                if config.temperature is not None:
                    params["temperature"] = config.temperature
                if config.top_p is not None:
                    params["top_p"] = config.top_p
                if config.top_k is not None:
                    params["top_k"] = config.top_k
                if config.stop_sequences is not None:
                    params["stop_sequences"] = config.stop_sequences
            else:
                raise ValueError("Anthropic requires generation_config.max_tokens")
            
            # Add provider kwargs
            params.update(request.provider_kwargs)
            
            anthropic_requests.append({
                "custom_id": request.custom_id,
                "params": params
            })
        
        return anthropic_requests
    
    def _convert_content(self, content_list: list) -> str | list[dict]:
        """Convert content to Anthropic format"""
        # Similar to OpenAI but Anthropic has specific format for documents
        # Text-only: return string
        if len(content_list) == 1 and content_list[0].type == "text":
            return content_list[0].text
        
        # Multimodal: return list
        result = []
        for content in content_list:
            if content.type == "text":
                result.append({"type": "text", "text": content.text})
            elif content.type == "image":
                result.append({
                    "type": "image",
                    "source": {
                        "type": content.source_type,
                        "media_type": content.media_type,
                        "data": content.data
                    }
                })
            elif content.type == "document":
                result.append({
                    "type": "document",
                    "source": {
                        "type": content.source_type,
                        "media_type": content.media_type,
                        "data": content.data
                    }
                })
        return result
    
    async def send_batch(self, batch: UnifiedBatchMetadata) -> str:
        """Send batch to Anthropic"""
        # Convert to Anthropic format
        anthropic_requests = self._convert_to_anthropic_format(batch)
        
        # Save to local file for record-keeping
        batch_id = FileManager.generate_batch_id()
        input_file_path = FileManager.get_input_file_path(self.name, batch_id)
        FileManager.write_jsonl(input_file_path, anthropic_requests)
        
        try:
            # Send directly to API (no file upload needed)
            response = await self.client.messages.batches.create(
                requests=anthropic_requests
            )
            
            logger.info(f"Created Anthropic batch: {response.id}")
            
            # Map Anthropic batch ID to our batch ID
            # Store mapping for later retrieval
            return response.id
            
        except Exception as e:
            raise ProviderAPIError("anthropic", 0, str(e))
    
    async def get_status(self, batch_id: str) -> BatchStatusResponse:
        """Get status from Anthropic"""
        # Similar pattern to OpenAI but use Anthropic client
        # Map processing_status: in_progress, ended
        pass
    
    async def get_results(self, batch_id: str) -> AsyncIterator[UnifiedResult]:
        """Stream results from Anthropic"""
        # Use client.messages.batches.results(batch_id)
        # This streams results directly, no file download
        async for result in self.client.messages.batches.results(batch_id):
            # Convert Anthropic result to UnifiedResult
            if result.result.type == "succeeded":
                yield UnifiedResult(
                    custom_id=result.custom_id,
                    status=ResultStatus.SUCCEEDED,
                    response=result.result.message.model_dump(),
                    provider_data=result.model_dump()
                )
            elif result.result.type == "errored":
                yield UnifiedResult(
                    custom_id=result.custom_id,
                    status=ResultStatus.ERRORED,
                    error=result.result.error.model_dump(),
                    provider_data=result.model_dump()
                )
            # Handle expired, cancelled similarly
```

---

### 16. `providers/google.py`

**Purpose:** Google GenAI Batch API implementation.

**Key Requirements:**
- Upload JSONL to Google Files API
- Create batch job with file reference
- Poll for completion
- Results may be in response or separate file
- systemInstruction in config object

**Google Format:**
```jsonl
{"key":"request_1", "request": {"contents": [{"parts": [{"text": "Hello"}]}], "generationConfig": {...}}}
```

**Implementation Outline:**

```python
from google import genai
from google.genai import types

class GoogleProvider(BaseProvider):
    """Google GenAI Batch API implementation"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = genai.Client(api_key=api_key)
    
    @property
    def name(self) -> str:
        return "google"
    
    def _convert_to_google_format(self, batch: UnifiedBatchMetadata) -> list[dict]:
        """Convert to Google format"""
        google_requests = []
        
        for request in batch.requests:
            # Build contents array
            contents = []
            for msg in request.messages:
                parts = []
                for content in msg.content:
                    if content.type == "text":
                        parts.append({"text": content.text})
                    elif content.type == "image":
                        # Google uses file_uri or inline_data
                        if content.source_type == "url":
                            parts.append({
                                "file_data": {
                                    "file_uri": content.data,
                                    "mime_type": content.media_type
                                }
                            })
                        else:
                            parts.append({
                                "inline_data": {
                                    "data": content.data,
                                    "mime_type": content.media_type
                                }
                            })
                
                contents.append({"parts": parts})
            
            # Build request object
            request_obj = {
                "contents": contents
            }
            
            # Add config (includes systemInstruction and generationConfig)
            config = {}
            
            if request.system_prompt:
                if isinstance(request.system_prompt, list):
                    config["systemInstruction"] = {
                        "parts": [{"text": p} for p in request.system_prompt]
                    }
                else:
                    config["systemInstruction"] = {
                        "parts": [{"text": request.system_prompt}]
                    }
            
            if request.generation_config:
                gen_config = {}
                if request.generation_config.temperature is not None:
                    gen_config["temperature"] = request.generation_config.temperature
                if request.generation_config.max_tokens is not None:
                    gen_config["maxOutputTokens"] = request.generation_config.max_tokens
                if request.generation_config.top_p is not None:
                    gen_config["topP"] = request.generation_config.top_p
                if request.generation_config.top_k is not None:
                    gen_config["topK"] = request.generation_config.top_k
                if request.generation_config.stop_sequences is not None:
                    gen_config["stopSequences"] = request.generation_config.stop_sequences
                
                if gen_config:
                    config["generationConfig"] = gen_config
            
            if config:
                request_obj["config"] = config
            
            # Add provider kwargs
            request_obj.update(request.provider_kwargs)
            
            google_requests.append({
                "key": request.custom_id,
                "request": request_obj
            })
        
        return google_requests
    
    async def send_batch(self, batch: UnifiedBatchMetadata) -> str:
        """Send batch to Google"""
        # Convert format
        google_requests = self._convert_to_google_format(batch)
        
        # Write to local file
        batch_id = FileManager.generate_batch_id()
        input_file_path = FileManager.get_input_file_path(self.name, batch_id)
        FileManager.write_jsonl(input_file_path, google_requests)
        
        try:
            # Upload file
            file_response = self.client.files.upload(
                file=str(input_file_path),
                config=types.UploadFileConfig(display_name=batch_id)
            )
            
            logger.info(f"Uploaded file to Google: {file_response.name}")
            
            # Get first request to extract model
            model = batch.requests[0].model
            
            # Create batch job
            batch_job = self.client.batches.create(
                model=model,
                src=f"files/{file_response.name}"
            )
            
            logger.info(f"Created Google batch: {batch_job.name}")
            
            return batch_job.name
            
        except Exception as e:
            raise ProviderAPIError("google", 0, str(e))
    
    async def get_status(self, batch_id: str) -> BatchStatusResponse:
        """Get status from Google"""
        # Map JOB_STATE_* to BatchStatus
        pass
    
    async def get_results(self, batch_id: str) -> AsyncIterator[UnifiedResult]:
        """Get results from Google"""
        # Results might be inline or in a file
        # Check batch_job.response for inlinedResponses or responsesFile
        pass
```

---

### 17. `providers/vllm.py`

**Purpose:** vLLM batch inference implementation.

**Key Requirements:**
- Use `vllm run-batch` subprocess
- OpenAI-compatible format
- No status API (check file existence)
- Async subprocess management

**Implementation Outline:**

```python
import asyncio
from pathlib import Path

class VLLMProvider(BaseProvider):
    """vLLM batch inference implementation"""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(api_key=None, **kwargs)
        self.model_path = model_path
        self.vllm_args = kwargs.get('vllm_args', {})
    
    @property
    def name(self) -> str:
        return "vllm"
    
    async def send_batch(self, batch: UnifiedBatchMetadata) -> str:
        """Send batch to vLLM"""
        # Convert to OpenAI format (vLLM uses same format)
        from .openai import OpenAIProvider
        converter = OpenAIProvider(api_key="dummy")
        vllm_requests = converter._convert_to_openai_format(batch)
        
        # Write to file
        batch_id = FileManager.generate_batch_id()
        input_file_path = FileManager.get_input_file_path(self.name, batch_id)
        output_file_path = FileManager.get_output_file_path(self.name, batch_id)
        
        FileManager.write_jsonl(input_file_path, vllm_requests)
        
        # Build vllm command
        cmd = [
            "vllm", "run-batch",
            "-i", str(input_file_path),
            "-o", str(output_file_path),
            "--model", self.model_path
        ]
        
        # Add additional vllm args
        for key, value in self.vllm_args.items():
            cmd.extend([f"--{key}", str(value)])
        
        # Check vllm is available
        try:
            result = await asyncio.create_subprocess_exec(
                "vllm", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            if result.returncode != 0:
                raise FileNotFoundError("vllm command not found")
        except FileNotFoundError:
            raise ProviderAPIError(
                "vllm", 0,
                "vllm command not found. Please install vllm: pip install vllm"
            )
        
        # Start vllm subprocess (async, non-blocking)
        logger.info(f"Starting vLLM batch: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Store process info for polling
        # In real implementation, save to file or database
        
        return batch_id
    
    async def get_status(self, batch_id: str) -> BatchStatusResponse:
        """Get status from vLLM"""
        # vLLM has no status API - check if output file exists
        output_file_path = FileManager.get_output_file_path(self.name, batch_id)
        
        if output_file_path.exists():
            # Check if file is complete (no longer being written)
            # Simple heuristic: check file size twice with delay
            size1 = output_file_path.stat().st_size
            await asyncio.sleep(2)
            size2 = output_file_path.stat().st_size
            
            if size1 == size2 and size1 > 0:
                # File is complete
                # Count results
                result_count = sum(1 for _ in FileManager.read_jsonl(output_file_path))
                
                return BatchStatusResponse(
                    batch_id=batch_id,
                    provider=self.name,
                    status=BatchStatus.COMPLETED,
                    request_counts=RequestCounts(
                        total=result_count,
                        succeeded=result_count,
                        processing=0
                    ),
                    created_at="",  # Not tracked for vLLM
                    completed_at=""
                )
            else:
                # Still writing
                return BatchStatusResponse(
                    batch_id=batch_id,
                    provider=self.name,
                    status=BatchStatus.IN_PROGRESS,
                    request_counts=RequestCounts(total=0, processing=0),
                    created_at=""
                )
        else:
            # Not started yet or still processing
            return BatchStatusResponse(
                batch_id=batch_id,
                provider=self.name,
                status=BatchStatus.IN_PROGRESS,
                request_counts=RequestCounts(total=0, processing=0),
                created_at=""
            )
    
    async def get_results(self, batch_id: str) -> AsyncIterator[UnifiedResult]:
        """Get results from vLLM"""
        output_file_path = FileManager.get_output_file_path(self.name, batch_id)
        
        if not output_file_path.exists():
            raise FileOperationError(f"Output file not found: {output_file_path}")
        
        # Parse results (same format as OpenAI)
        for line in FileManager.read_jsonl(output_file_path):
            custom_id = line.get("custom_id")
            response_data = line.get("response")
            error_data = line.get("error")
            
            if error_data:
                yield UnifiedResult(
                    custom_id=custom_id,
                    status=ResultStatus.ERRORED,
                    error=error_data,
                    provider_data=line
                )
            else:
                yield UnifiedResult(
                    custom_id=custom_id,
                    status=ResultStatus.SUCCEEDED,
                    response=response_data,
                    provider_data=line
                )
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel vLLM batch"""
        # Would need to track process PID and kill it
        # For simplicity, not implemented
        raise NotImplementedError("vLLM does not support batch cancellation")
```

---

### 18. `router.py`

**Purpose:** Main BatchRouter orchestration class.

**Class:**

```python
import asyncio
import time
from typing import Optional, AsyncIterator
from .core.base import BaseProvider
from .core.requests import UnifiedRequest, UnifiedBatchMetadata
from .core.responses import BatchStatusResponse, UnifiedResult
from .core.enums import BatchStatus
from .exceptions import ProviderNotFoundError, BatchTimeoutError
from .utils.file_manager import FileManager
from .utils.validation import validate_batch
from .utils.logging import logger
from .constants import (
    DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT,
    MAX_POLL_INTERVAL, BACKOFF_MULTIPLIER
)

class BatchRouter:
    """
    Main entry point for unified batch processing.
    
    Usage:
        router = BatchRouter()
        router.register_provider(OpenAIProvider(api_key="..."))
        router.register_provider(AnthropicProvider(api_key="..."))
        
        batch_id = await router.send_batch("openai", requests)
        status = await router.wait_for_completion("openai", batch_id)
        
        async for result in router.get_results("openai", batch_id):
            print(result.custom_id, result.response)
    """
    
    def __init__(self):
        """Initialize router"""
        self.providers: dict[str, BaseProvider] = {}
        
        # Ensure directory structure exists
        FileManager.ensure_directory_structure()
        
        logger.info("BatchRouter initialized")
    
    def register_provider(self, provider: BaseProvider) -> None:
        """
        Register a provider implementation.
        
        Args:
            provider: Provider instance
        """
        self.providers[provider.name] = provider
        logger.info(f"Registered provider: {provider.name}")
    
    def get_provider(self, name: str) -> BaseProvider:
        """Get provider by name"""
        if name not in self.providers:
            raise ProviderNotFoundError(name)
        return self.providers[name]
    
    async def send_batch(
        self,
        provider_name: str,
        requests: list[UnifiedRequest],
        metadata: Optional[dict] = None
    ) -> str:
        """
        Send batch through specified provider.
        
        Args:
            provider_name: Provider name ("openai", "anthropic", etc.)
            requests: List of UnifiedRequest objects
            metadata: Optional user metadata
        
        Returns:
            batch_id: Unique identifier to track this batch
        
        Raises:
            ProviderNotFoundError: If provider not registered
            ValidationError: If batch doesn't meet requirements
            ProviderAPIError: If provider API returns error
        """
        provider = self.get_provider(provider_name)
        
        # Create batch metadata
        batch = UnifiedBatchMetadata(
            provider=provider_name,
            requests=requests,
            metadata=metadata or {}
        )
        
        # Validate batch
        validate_batch(batch)
        
        # Send to provider
        logger.info(f"Sending batch with {len(requests)} requests to {provider_name}")
        
        batch_id = await provider.send_batch(batch)
        
        logger.info(f"Batch sent successfully: {batch_id}")
        
        return batch_id
    
    async def get_status(
        self,
        provider_name: str,
        batch_id: str
    ) -> BatchStatusResponse:
        """
        Get current status of a batch.
        
        Args:
            provider_name: Provider name
            batch_id: Batch identifier
        
        Returns:
            BatchStatusResponse with current status
        """
        provider = self.get_provider(provider_name)
        return await provider.get_status(batch_id)
    
    async def wait_for_completion(
        self,
        provider_name: str,
        batch_id: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
        verbose: bool = True
    ) -> BatchStatusResponse:
        """
        Poll batch status until completion with exponential backoff.
        
        Args:
            provider_name: Provider name
            batch_id: Batch identifier
            poll_interval: Initial poll interval in seconds
            timeout: Maximum time to wait in seconds
            verbose: Whether to log progress
        
        Returns:
            Final BatchStatusResponse
        
        Raises:
            BatchTimeoutError: If batch doesn't complete within timeout
            BatchFailedError: If batch fails
        """
        provider = self.get_provider(provider_name)
        
        start_time = time.time()
        current_interval = poll_interval
        iteration = 0
        
        logger.info(f"Waiting for batch {batch_id} to complete (timeout: {timeout}s)")
        
        while True:
            iteration += 1
            
            # Get status
            status = await provider.get_status(batch_id)
            
            if verbose and iteration % 5 == 0:
                logger.info(
                    f"Batch {batch_id} status: {status.status.value} "
                    f"({status.request_counts.succeeded}/{status.request_counts.total} succeeded)"
                )
            
            # Check if complete
            if status.is_complete():
                elapsed = time.time() - start_time
                logger.info(
                    f"Batch {batch_id} completed in {elapsed:.1f}s with status: {status.status.value}"
                )
                
                if status.status == BatchStatus.FAILED:
                    from .exceptions import BatchFailedError
                    raise BatchFailedError(batch_id, status.provider_data)
                
                return status
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise BatchTimeoutError(batch_id, timeout)
            
            # Wait before next poll (exponential backoff)
            await asyncio.sleep(current_interval)
            current_interval = min(current_interval * BACKOFF_MULTIPLIER, MAX_POLL_INTERVAL)
    
    async def get_results(
        self,
        provider_name: str,
        batch_id: str
    ) -> AsyncIterator[UnifiedResult]:
        """
        Stream results from a completed batch.
        
        Args:
            provider_name: Provider name
            batch_id: Batch identifier
        
        Yields:
            UnifiedResult objects for each request
        
        Note: Results may not be in the same order as input requests.
        Use custom_id to match results to requests.
        """
        provider = self.get_provider(provider_name)
        
        logger.info(f"Retrieving results for batch {batch_id}")
        
        count = 0
        async for result in provider.get_results(batch_id):
            count += 1
            yield result
        
        logger.info(f"Retrieved {count} results from batch {batch_id}")
    
    async def cancel_batch(
        self,
        provider_name: str,
        batch_id: str
    ) -> bool:
        """
        Cancel a running batch.
        
        Args:
            provider_name: Provider name
            batch_id: Batch identifier
        
        Returns:
            True if cancellation successful
        """
        provider = self.get_provider(provider_name)
        result = await provider.cancel_batch(batch_id)
        
        if result:
            logger.info(f"Cancelled batch {batch_id}")
        
        return result
    
    def list_providers(self) -> list[str]:
        """List registered provider names"""
        return list(self.providers.keys())
```

**Tests:**
- Test provider registration
- Test routing to correct provider
- Test exponential backoff logic
- Test timeout handling
- Mock provider responses

---

### 19. Package `__init__.py` files

**`batch_router/__init__.py`:**

```python
"""
Batch LLM Router - Unified interface for batch requests across multiple LLM providers.

Example usage:
    from batch_router import BatchRouter, UnifiedRequest, UnifiedMessage
    from batch_router.providers import OpenAIProvider, AnthropicProvider
    
    # Initialize router
    router = BatchRouter()
    router.register_provider(OpenAIProvider(api_key="sk-..."))
    router.register_provider(AnthropicProvider(api_key="sk-ant-..."))
    
    # Create requests
    requests = [
        UnifiedRequest(
            custom_id="req-1",
            model="gpt-4",
            messages=[UnifiedMessage.from_text("user", "Hello!")]
        )
    ]
    
    # Send batch
    batch_id = await router.send_batch("openai", requests)
    
    # Wait for completion
    status = await router.wait_for_completion("openai", batch_id)
    
    # Get results
    async for result in router.get_results("openai", batch_id):
        print(result.custom_id, result.get_content())
"""

from .router import BatchRouter
from .core.requests import UnifiedRequest, UnifiedBatchMetadata
from .core.messages import UnifiedMessage
from .core.content import TextContent, ImageContent, DocumentContent
from .core.config import GenerationConfig
from .core.responses import BatchStatusResponse, UnifiedResult
from .core.enums import BatchStatus, ResultStatus
from .exceptions import (
    BatchRouterError,
    ProviderNotFoundError,
    ValidationError,
    BatchTimeoutError,
    BatchFailedError,
    FileOperationError,
    ProviderAPIError
)

__version__ = "0.1.0"

__all__ = [
    # Main router
    "BatchRouter",
    
    # Core models
    "UnifiedRequest",
    "UnifiedBatchMetadata",
    "UnifiedMessage",
    "TextContent",
    "ImageContent",
    "DocumentContent",
    "GenerationConfig",
    
    # Responses
    "BatchStatusResponse",
    "UnifiedResult",
    
    # Enums
    "BatchStatus",
    "ResultStatus",
    
    # Exceptions
    "BatchRouterError",
    "ProviderNotFoundError",
    "ValidationError",
    "BatchTimeoutError",
    "BatchFailedError",
    "FileOperationError",
    "ProviderAPIError",
]
```

**`batch_router/core/__init__.py`:**

```python
from .content import TextContent, ImageContent, DocumentContent
from .messages import UnifiedMessage
from .requests import UnifiedRequest, UnifiedBatchMetadata
from .responses import BatchStatusResponse, UnifiedResult, RequestCounts
from .config import GenerationConfig
from .enums import BatchStatus, ResultStatus
from .base import BaseProvider

__all__ = [
    "TextContent",
    "ImageContent",
    "DocumentContent",
    "UnifiedMessage",
    "UnifiedRequest",
    "UnifiedBatchMetadata",
    "BatchStatusResponse",
    "UnifiedResult",
    "RequestCounts",
    "GenerationConfig",
    "BatchStatus",
    "ResultStatus",
    "BaseProvider",
]
```

**`batch_router/providers/__init__.py`:**

```python
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .vllm import VLLMProvider

__all__ = [
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "VLLMProvider",
]
```

**`batch_router/utils/__init__.py`:**

```python
from .file_manager import FileManager
from .validation import validate_batch, validate_request
from .logging import logger, setup_logger

__all__ = [
    "FileManager",
    "validate_batch",
    "validate_request",
    "logger",
    "setup_logger",
]
```

---

## Testing Strategy

### Unit Tests
- Test each dataclass with valid/invalid inputs
- Test validation functions
- Test format converters (unified ↔ provider)
- Test file operations
- Mock provider APIs

### Integration Tests
- Test full flow: send → poll → retrieve
- Test with multiple providers
- Test error handling paths
- Test file cleanup

### Example Test Structure:

```python
# tests/test_openai_provider.py
import pytest
from batch_router.providers import OpenAIProvider
from batch_router.core import UnifiedRequest, UnifiedMessage

@pytest.mark.asyncio
async def test_openai_format_conversion():
    """Test unified to OpenAI format conversion"""
    provider = OpenAIProvider(api_key="test")
    
    request = UnifiedRequest(
        custom_id="test-1",
        model="gpt-4",
        messages=[UnifiedMessage.from_text("user", "Hello")],
        system_prompt="You are helpful"
    )
    
    batch = UnifiedBatchMetadata(
        provider="openai",
        requests=[request]
    )
    
    openai_format = provider._convert_to_openai_format(batch)
    
    assert openai_format[0]["custom_id"] == "test-1"
    assert openai_format[0]["body"]["model"] == "gpt-4"
    assert openai_format[0]["body"]["messages"][0]["role"] == "system"
    assert openai_format[0]["body"]["messages"][1]["role"] == "user"
```

---

## Example Usage

### Basic Example:

```python
import asyncio
from batch_router import (
    BatchRouter, UnifiedRequest, UnifiedMessage,
    GenerationConfig
)
from batch_router.providers import OpenAIProvider, AnthropicProvider

async def main():
    # Initialize router
    router = BatchRouter()
    router.register_provider(OpenAIProvider(api_key="sk-..."))
    router.register_provider(AnthropicProvider(api_key="sk-ant-..."))
    
    # Create requests
    requests = [
        UnifiedRequest(
            custom_id="req-1",
            model="gpt-4o",
            messages=[
                UnifiedMessage.from_text("user", "What is 2+2?")
            ],
            system_prompt="You are a helpful math tutor.",
            generation_config=GenerationConfig(
                temperature=0.7,
                max_tokens=100
            )
        ),
        UnifiedRequest(
            custom_id="req-2",
            model="gpt-4o",
            messages=[
                UnifiedMessage.from_text("user", "What is the capital of France?")
            ],
            system_prompt="You are a geography expert.",
            generation_config=GenerationConfig(
                temperature=0.7,
                max_tokens=100
            )
        )
    ]
    
    # Send batch
    batch_id = await router.send_batch("openai", requests)
    print(f"Batch sent: {batch_id}")
    
    # Wait for completion
    status = await router.wait_for_completion("openai", batch_id)
    print(f"Batch completed: {status.status}")
    
    # Get results
    async for result in router.get_results("openai", batch_id):
        if result.is_successful():
            print(f"{result.custom_id}: {result.get_content()}")
        else:
            print(f"{result.custom_id}: ERROR - {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Example with Multimodal:

```python
import asyncio
import base64
from batch_router import (
    BatchRouter, UnifiedRequest, UnifiedMessage,
    TextContent, ImageContent
)
from batch_router.providers import AnthropicProvider

async def main():
    router = BatchRouter()
    router.register_provider(AnthropicProvider(api_key="sk-ant-..."))
    
    # Read image and encode to base64
    with open("image.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Create multimodal request
    request = UnifiedRequest(
        custom_id="vision-1",
        model="claude-sonnet-4-5",
        messages=[
            UnifiedMessage(
                role="user",
                content=[
                    ImageContent(
                        source_type="base64",
                        media_type="image/jpeg",
                        data=image_data
                    ),
                    TextContent(text="What's in this image?")
                ]
            )
        ],
        generation_config=GenerationConfig(max_tokens=500)
    )
    
    batch_id = await router.send_batch("anthropic", [request])
    status = await router.wait_for_completion("anthropic", batch_id)
    
    async for result in router.get_results("anthropic", batch_id):
        print(result.get_content())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Dependencies

Add to `pyproject.toml` or `requirements.txt`:

```toml
[project]
name = "batch_router"
version = "0.1.0"
dependencies = [
    "openai>=1.0.0",
    "anthropic>=0.34.0",
    "google-genai>=0.1.0",
    "aiohttp>=3.9.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
vllm = ["vllm>=0.7.0"]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
    "black>=24.0.0",
    "mypy>=1.8.0",
    "ruff>=0.1.0",
]
```

---

## Implementation Checklist

### Phase 1: Foundation
- [ ] Create project structure
- [ ] Implement `constants.py`
- [ ] Implement `exceptions.py`
- [ ] Implement all `core/` modules
- [ ] Write unit tests for core models

### Phase 2: Utilities
- [ ] Implement `utils/logging.py`
- [ ] Implement `utils/validation.py`
- [ ] Implement `utils/file_manager.py`
- [ ] Write unit tests for utilities

### Phase 3: Providers
- [ ] Implement `OpenAIProvider`
- [ ] Implement `AnthropicProvider`
- [ ] Implement `GoogleProvider`
- [ ] Implement `VLLMProvider`
- [ ] Write unit tests for each provider
- [ ] Write integration tests with mocked APIs

### Phase 4: Router
- [ ] Implement `BatchRouter`
- [ ] Write unit tests for router
- [ ] Write integration tests for full flow

### Phase 5: Documentation & Examples
- [ ] Write README.md
- [ ] Write API documentation
- [ ] Create example scripts
- [ ] Write usage guide

### Phase 6: Polish
- [ ] Type checking with mypy
- [ ] Code formatting with black
- [ ] Linting with ruff
- [ ] Final testing
- [ ] Package for distribution

---

## Next Steps

1. **Start with Phase 1**: Implement core data models first as they have no dependencies
2. **Test incrementally**: Write tests as you implement each module
3. **Mock provider APIs**: Use `pytest-mock` or `responses` library for testing
4. **Iterate on design**: The design may need adjustments during implementation

---

## Notes

- All file paths should use `Path` objects from `pathlib`
- All I/O operations should be async
- Use structured logging throughout
- Provider implementations should be independent (no cross-dependencies)
- Keep error messages descriptive and actionable
- Document all public APIs with docstrings
- Follow PEP 8 style guide
- Use type hints everywhere
