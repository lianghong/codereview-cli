"""File batching for managing context window limits."""

import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict  # type: ignore[attr-defined]


class FileBatch(BaseModel):
    """Represents a batch of files to analyze together."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    files: list[Path]
    batch_number: int
    total_batches: int


class FileBatcher:
    """Batches files into chunks for LLM analysis."""

    def __init__(self, max_files_per_batch: int = 10):
        """
        Initialize batcher.

        Args:
            max_files_per_batch: Maximum files per batch (default 10, must be >= 1)

        Raises:
            ValueError: If max_files_per_batch is less than 1
        """
        if max_files_per_batch < 1:
            raise ValueError("max_files_per_batch must be at least 1")
        self.max_files_per_batch = max_files_per_batch

    def create_batches(self, files: list[Path]) -> list[FileBatch]:
        """
        Create batches from file list.

        Args:
            files: List of file paths

        Returns:
            List of FileBatch objects
        """
        if not files:
            return []

        batches: list[FileBatch] = []
        total_batches = math.ceil(len(files) / self.max_files_per_batch)

        for i in range(0, len(files), self.max_files_per_batch):
            batch_files = files[i : i + self.max_files_per_batch]
            batch_number = len(batches) + 1

            batch = FileBatch(
                files=batch_files,
                batch_number=batch_number,
                total_batches=total_batches,
            )
            batches.append(batch)

        return batches
