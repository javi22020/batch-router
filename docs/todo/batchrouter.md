# BatchRouter Implementation Plan

## Overview

Implement a `BatchRouter` class in `src/batch_router/router.py` that can run multiple batches in parallel across different providers, with proper validation and flexible result retrieval.

## Design Decisions

- **Parallel Execution**: All batches start simultaneously using `asyncio.gather()`
- **Result Streaming**: Default method streams all results together; alternative method groups by batch
- **No Auto-Splitting**: Users manage batch sizes; router validates against provider limits
- **Early Validation**: Raises error before processing if required providers aren't registered

## Core Attributes

The `BatchRouter` class will have:

1. **`providers: dict[str, BaseProvider]`** - Registry mapping provider names to instances
2. **`_active_batches: dict[str, tuple[str, str]]`** - Tracking active batches (batch_id -> (provider_name, status))
3. **`_default_poll_interval: int`** - Default polling interval for status checks (60s)
4. **`_default_timeout: int`** - Default timeout for batch completion (24 hours)
5. **`_max_poll_interval: int`** - Maximum polling interval for exponential backoff (5 minutes)

## Key Methods to Implement

### 1. Provider Management

**`register_provider(provider: BaseProvider) -> None`**

- Adds a provider instance to the registry
- Validates provider isn't already registered
- Logs registration

**`unregister_provider(provider_name: str) -> None`**

- Removes a provider from the registry
- Useful for cleanup or switching providers

**`get_provider(provider_name: str) -> BaseProvider`**

- Retrieves provider instance
- Raises `ProviderNotFoundError` if not found

**`list_providers() -> list[str]`**

- Returns list of registered provider names

### 2. Batch Validation

**`_validate_batches(batches: list[UnifiedBatchMetadata]) -> None`**

- Checks all required providers are registered
- Validates batch sizes against provider limits
- Checks for duplicate custom_ids across all batches
- Raises `ValidationError` or `ProviderNotFoundError` if issues found

### 3. Parallel Batch Execution

**`run_batches(batches: list[UnifiedBatchMetadata], wait_for_completion: bool = True, poll_interval: Optional[int] = None, timeout: Optional[int] = None, verbose: bool = True) -> list[str]`**

- **Before execution**: Calls `_validate_batches()` to ensure all providers exist
- **Execution**: Uses `asyncio.gather()` to send all batches in parallel
- **Returns**: List of batch_ids in same order as input batches
- **If `wait_for_completion=True`**: Polls all batches until complete
- **Parameters**:
  - `poll_interval`: Override default polling interval
  - `timeout`: Override default timeout
  - `verbose`: Enable/disable progress logging

**`_send_batch_with_provider(batch: UnifiedBatchMetadata) -> str`**

- Helper method that looks up provider and calls `send_batch()`
- Tracks batch in `_active_batches`
- Returns batch_id

**`wait_for_all(batch_ids: list[str], poll_interval: Optional[int] = None, timeout: Optional[int] = None, verbose: bool = True) -> dict[str, BatchStatusResponse]`**

- Polls all batches until completion
- Uses exponential backoff for polling
- Returns final status for each batch
- Raises `BatchTimeoutError` if any batch times out

### 4. Status Checking

**`get_status(provider_name: str, batch_id: str) -> BatchStatusResponse`**

- Gets status for a single batch
- Delegates to provider's `get_status()` method

**`get_all_statuses(batch_ids: list[str]) -> dict[str, BatchStatusResponse]`**

- Gets status for multiple batches in parallel
- Returns mapping of batch_id to status

### 5. Result Retrieval (Default: Mixed Streaming)

**`get_all_results(batch_ids: list[str], provider_mapping: Optional[dict[str, str]] = None) -> AsyncIterator[tuple[str, UnifiedResult]]`**

- **Default method** for retrieving results
- Streams results from all batches as they arrive
- Yields `(batch_id, result)` tuples
- Uses `asyncio.as_completed()` to stream mixed results
- If `provider_mapping` is None, looks up providers from `_active_batches`

### 6. Result Retrieval (Grouped by Batch)

**`get_results_by_batch(batch_ids: list[str], provider_mapping: Optional[dict[str, str]] = None) -> dict[str, list[UnifiedResult]]`**

- Alternative method for grouped results
- Returns dictionary mapping batch_id to list of results
- Collects all results before returning
- Useful when you need complete batch results together

### 7. Batch Cancellation

**`cancel_batch(provider_name: str, batch_id: str) -> bool`**

- Cancels a single batch
- Removes from `_active_batches` if successful

**`cancel_all(batch_ids: list[str]) -> dict[str, bool]`**

- Cancels multiple batches in parallel
- Returns mapping of batch_id to success status

### 8. Convenience Methods

**`run_single_batch(batch: UnifiedBatchMetadata, wait_for_completion: bool = True, **kwargs) -> str`**

- Convenience wrapper for running a single batch
- Calls `run_batches([batch], ...)`
- Returns single batch_id

**`get_results(provider_name: str, batch_id: str) -> AsyncIterator[UnifiedResult]`**

- Convenience method for getting results from a single batch
- Delegates to provider's `get_results()` method

## Implementation File Structure

**File**: `src/batch_router/router.py`

```python
import asyncio
import time
from typing import Optional, AsyncIterator
from .core.base import BaseProvider
from .core.requests import UnifiedRequest, UnifiedBatchMetadata
from .core.responses import BatchStatusResponse, UnifiedResult
from .core.enums import BatchStatus
from .exceptions import (
    ProviderNotFoundError, 
    BatchTimeoutError,
    ValidationError
)

class BatchRouter:
    def __init__(
        self,
        default_poll_interval: int = 60,
        default_timeout: int = 86400,
        max_poll_interval: int = 300
    ):
        # Initialize attributes
        # Setup logging
    
    # [All methods listed above]
```

## Validation Logic Details

The `_validate_batches()` method should:

1. **Check provider availability**:

   - Extract unique provider names from all batches
   - Verify each provider exists in `self.providers`
   - Raise `ProviderNotFoundError` with list of missing providers

2. **Validate batch sizes** (from `src/batch_router/core/base.py` line 274-322):

   - Check each batch against provider-specific limits
   - OpenAI: max 50,000 requests
   - Anthropic: max 100,000 requests
   - Raise `ValidationError` if exceeded

3. **Check for duplicate custom_ids**:

   - Collect all custom_ids across all batches
   - Ensure global uniqueness (optional, could be per-batch)
   - Raise `ValidationError` if duplicates found

## Parallel Execution Pattern

```python
async def run_batches(self, batches, wait_for_completion=True, **kwargs):
    # 1. Validate first
    self._validate_batches(batches)
    
    # 2. Send all batches in parallel
    send_tasks = [
        self._send_batch_with_provider(batch) 
        for batch in batches
    ]
    batch_ids = await asyncio.gather(*send_tasks)
    
    # 3. Optionally wait for completion
    if wait_for_completion:
        await self.wait_for_all(batch_ids, **kwargs)
    
    return batch_ids
```

## Mixed Result Streaming Pattern

```python
async def get_all_results(self, batch_ids, provider_mapping=None):
    # Create async generators for each batch
    result_generators = []
    for batch_id in batch_ids:
        provider_name = self._get_provider_for_batch(batch_id, provider_mapping)
        provider = self.get_provider(provider_name)
        result_generators.append((batch_id, provider.get_results(batch_id)))
    
    # Use asyncio.as_completed() or similar to stream mixed results
    # Yield (batch_id, result) tuples as they arrive
```

## Update Module Exports

Update `src/batch_router/__init__.py` to export the new `BatchRouter` class.

## Testing Considerations

Create `tests/test_router.py` with:

- Tests for provider registration/unregistration
- Tests for batch validation (missing providers, size limits)
- Tests for parallel execution with mock providers
- Tests for mixed result streaming
- Tests for grouped result retrieval
- Tests for cancellation

## Files to Modify

1. **Create**: `src/batch_router/router.py` (~300-400 lines)
2. **Update**: `src/batch_router/__init__.py` (add BatchRouter export)
3. **Create**: `tests/test_router.py` (comprehensive test suite)

### To-dos

- [ ] Review existing codebase structure and confirm BatchRouter design fits with BaseProvider interface
- [ ] Create BatchRouter class in src/batch_router/router.py with initialization and core attributes
- [ ] Implement provider registration, unregistration, and lookup methods
- [ ] Implement _validate_batches() method with provider checks, size validation, and custom_id uniqueness
- [ ] Implement run_batches() and _send_batch_with_provider() for parallel batch execution
- [ ] Implement wait_for_all() with exponential backoff polling
- [ ] Implement get_status() and get_all_statuses() for batch status checking
- [ ] Implement get_all_results() for streaming mixed results from multiple batches
- [ ] Implement get_results_by_batch() for retrieving results grouped by batch
- [ ] Implement cancel_batch() and cancel_all() for batch cancellation
- [ ] Implement convenience methods: run_single_batch() and get_results()
- [ ] Update src/batch_router/__init__.py to export BatchRouter class
- [ ] Create comprehensive test suite in tests/test_router.py
- [ ] Add appropriate logging throughout BatchRouter methods
- [ ] Add docstrings and usage examples to all BatchRouter methods