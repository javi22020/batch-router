"""JSONL file operations and directory management."""

import json
import re
from pathlib import Path
from typing import Any, AsyncIterator
import aiofiles


def sanitize_filename_component(component: str) -> str:
    """
    Sanitize a filename component for safe file naming.

    Rules:
    - Only allows alphanumeric characters (a-z, A-Z, 0-9) and dashes (-)
    - Replaces underscores with dashes
    - Removes all other characters

    Args:
        component: String to sanitize

    Returns:
        Sanitized string safe for use in filenames

    Examples:
        >>> sanitize_filename_component("my_model")
        'my-model'
        >>> sanitize_filename_component("gpt-4o")
        'gpt-4o'
        >>> sanitize_filename_component("test@#$batch")
        'testbatch'
    """
    # First, replace underscores with dashes
    sanitized = component.replace("_", "-")
    # Then, keep only alphanumeric characters and dashes
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "", sanitized)
    return sanitized


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
        batch_dir = Path(".batch_router/generated") / provider
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    @staticmethod
    async def write_jsonl(
        file_path: Path | str,
        data: list[dict[str, Any]]
    ) -> None:
        """
        Write data to JSONL file asynchronously.

        Each dict in data becomes one line in the file.
        """
        file_path = Path(file_path)
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                line = json.dumps(item, ensure_ascii=False)
                await f.write(line + '\n')

    @staticmethod
    async def read_jsonl(
        file_path: Path | str
    ) -> list[dict[str, Any]]:
        """
        Read JSONL file asynchronously.

        Returns list of dictionaries, one per line.
        """
        file_path = Path(file_path)
        result = []

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            async for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    result.append(json.loads(line))

        return result

    @staticmethod
    async def stream_jsonl(
        file_path: Path | str
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream JSONL file line by line.

        Memory efficient for large files.
        Yields one dictionary per line.
        """
        file_path = Path(file_path)

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            async for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    yield json.loads(line)

    @staticmethod
    def get_batch_files(
        provider: str,
        batch_id: str
    ) -> dict[str, Path]:
        """
        Get all file paths for a batch.

        Returns dict with keys: unified, provider, output, results
        """
        base_dir = Path(".batch_router/generated") / provider
        return {
            "unified": base_dir / f"batch_{batch_id}_unified.jsonl",
            "provider": base_dir / f"batch_{batch_id}_provider.jsonl",
            "output": base_dir / f"batch_{batch_id}_output.jsonl",
            "results": base_dir / f"batch_{batch_id}_results.jsonl",
        }
