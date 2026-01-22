"""Smart batching for managing context window limits."""
from pathlib import Path
from typing import List
from pydantic import BaseModel, ConfigDict


class FileBatch(BaseModel):
    """Represents a batch of files to analyze together."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    files: List[Path]
    batch_number: int
    total_batches: int


class SmartBatcher:
    """Batches files intelligently for LLM analysis."""

    def __init__(self, max_files_per_batch: int = 10):
        """
        Initialize batcher.

        Args:
            max_files_per_batch: Maximum files per batch (default 10)
        """
        self.max_files_per_batch = max_files_per_batch

    def create_batches(self, files: List[Path]) -> List[FileBatch]:
        """
        Create batches from file list.

        Args:
            files: List of file paths

        Returns:
            List of FileBatch objects
        """
        if not files:
            return []

        batches = []
        total_batches = (len(files) + self.max_files_per_batch - 1) // self.max_files_per_batch

        for i in range(0, len(files), self.max_files_per_batch):
            batch_files = files[i:i + self.max_files_per_batch]
            batch_number = len(batches) + 1

            batch = FileBatch(
                files=batch_files,
                batch_number=batch_number,
                total_batches=total_batches
            )
            batches.append(batch)

        return batches
