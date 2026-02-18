"""File batching for managing context window limits."""

import logging
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict  # type: ignore[attr-defined]

# Estimated token overhead per file for headers/separators in batch context
PER_FILE_OVERHEAD_TOKENS = 50


class FileBatch(BaseModel):
    """Represents a batch of files to analyze together."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    files: list[Path]
    batch_number: int
    total_batches: int


class FileBatcher:
    """Batches files into chunks for LLM analysis."""

    def __init__(
        self,
        max_files_per_batch: int = 10,
        token_budget: int | None = None,
    ):
        """
        Initialize batcher.

        Args:
            max_files_per_batch: Maximum files per batch (default 10, must be >= 1)
            token_budget: Maximum tokens available for file content per batch.
                If None, batching uses file-count only.

        Raises:
            ValueError: If max_files_per_batch < 1 or token_budget <= 0
        """
        if max_files_per_batch < 1:
            raise ValueError("max_files_per_batch must be at least 1")
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be greater than 0")
        self.max_files_per_batch = max_files_per_batch
        self.token_budget = token_budget

    @staticmethod
    def estimate_file_tokens(file_path: Path) -> int:
        """Estimate token count from file size (~4 bytes per token heuristic).

        Args:
            file_path: Path to the file

        Returns:
            Estimated token count (includes per-file overhead)
        """
        try:
            return file_path.stat().st_size // 4 + PER_FILE_OVERHEAD_TOKENS
        except OSError:
            return PER_FILE_OVERHEAD_TOKENS

    def create_batches(self, files: list[Path]) -> list[FileBatch]:
        """
        Create batches from file list.

        Uses token-budget-aware packing when token_budget is set,
        otherwise falls back to count-only batching.

        Args:
            files: List of file paths

        Returns:
            List of FileBatch objects
        """
        if not files:
            return []

        if self.token_budget is None:
            return self._batch_by_count(files)
        return self._batch_by_tokens(files)

    def _batch_by_count(self, files: list[Path]) -> list[FileBatch]:
        """Batch files by count only (original logic).

        Args:
            files: List of file paths

        Returns:
            List of FileBatch objects
        """
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

    def _batch_by_tokens(self, files: list[Path]) -> list[FileBatch]:
        """Batch files using greedy token-budget packing.

        Adds files to the current batch until the next file would exceed
        the token budget or the file-count cap, then starts a new batch.
        Files larger than the budget get their own single-file batch.

        Args:
            files: List of file paths

        Returns:
            List of FileBatch objects
        """
        assert self.token_budget is not None

        raw_batches: list[list[Path]] = []
        current_batch: list[Path] = []
        current_tokens = 0

        for file_path in files:
            file_tokens = self.estimate_file_tokens(file_path)

            # Oversized file: give it its own batch
            if file_tokens > self.token_budget:
                # Flush current batch first
                if current_batch:
                    raw_batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

                logging.warning(
                    "File %s estimated at %d tokens exceeds budget of %d tokens; "
                    "placing in single-file batch",
                    file_path,
                    file_tokens,
                    self.token_budget,
                )
                raw_batches.append([file_path])
                continue

            # Would exceed token budget or file-count cap: start new batch
            if (
                current_tokens + file_tokens > self.token_budget
                or len(current_batch) >= self.max_files_per_batch
            ):
                if current_batch:
                    raw_batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(file_path)
            current_tokens += file_tokens

        # Flush remaining files
        if current_batch:
            raw_batches.append(current_batch)

        # Convert to FileBatch objects
        total_batches = len(raw_batches)
        return [
            FileBatch(
                files=batch_files,
                batch_number=i + 1,
                total_batches=total_batches,
            )
            for i, batch_files in enumerate(raw_batches)
        ]
