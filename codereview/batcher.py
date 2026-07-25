"""File batching for managing context window limits."""

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

# Estimated token overhead per file for headers/separators in batch context
PER_FILE_OVERHEAD_TOKENS = 50

# Fallback bytes-per-token divisor used when tiktoken is unavailable. Modern
# BPE tokenizers average ~4 bytes/token on ASCII English but ~2-3 bytes/token
# on UTF-8 multi-byte content (CJK, emoji). A divisor of 3 is the conservative
# single-knob choice: overestimates ASCII slightly (safe — extra headroom)
# and roughly matches CJK (avoids the silent context-window overflows the
# original //4 heuristic produced). Only used as a fallback now — see
# count_tokens() for the real path.
BYTES_PER_TOKEN = 3

# Maximum file size we'll feed to tiktoken's encoder. Above this, the file is
# almost certainly going to be skipped or oversized anyway, so we save the
# encode pass and use the byte heuristic. 2 MB ≈ ~700K tokens, well past
# any provider's context window today.
_TIKTOKEN_MAX_BYTES = 2 * 1024 * 1024

# Cache slots for per-file token counts. One entry is three ints plus a path
# string — a few hundred bytes — so even 100K files of headroom costs a few
# tens of MB, and a run's working set is bounded by the number of files
# scanned, which is already all held in memory as a list of Paths.
#
# Sized to swallow a whole run rather than a "typical" repo. The bound has to
# exceed the *file count*, not some notion of a reasonable project: a single
# run estimates every file at least twice (--dry-run's table, then
# create_batches' packing loop), so at a bound below the file count the first
# pass evicts its own earliest entries before the second pass reaches them and
# every file is re-encoded — the memoization silently stops paying off exactly
# in the large repositories it was added for. 4096 used to be the bound, which
# is a plausible file count for a monorepo.
_TOKEN_CACHE_SIZE = 100_000


@lru_cache(maxsize=1)
def _get_encoder() -> Any | None:
    """Return a cached tiktoken encoder, or None if tiktoken is unavailable.

    Uses ``cl100k_base`` as a universal-enough encoding: it's exact for
    GPT-3.5/4/4o, within a few percent for Claude/Gemini/DeepSeek/Kimi for
    typical code, and always offline. Bedrock's Anthropic models would prefer
    the official Anthropic tokenizer, but that requires a network round-trip
    per file which would dominate batching cost. The estimate doesn't need
    to be exact — it just needs to be *much* better than bytes // 3, which
    misjudges by 30-60% on non-ASCII source.
    """
    try:
        import tiktoken
    except ImportError:
        logging.debug("tiktoken not installed; falling back to byte-based estimator")
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except (
        Exception
    ) as exc:  # pragma: no cover — only triggers if tiktoken's data files are missing
        logging.warning(
            "tiktoken encoder failed to load (%s); using byte fallback", exc
        )
        return None


def count_tokens(text: str) -> int:
    """Count tokens in *text* using tiktoken when available, byte heuristic otherwise.

    The result is an estimate, not a guarantee — different providers use
    different tokenizers and can disagree by ~10% on identical text. Used
    for batching decisions (where over-estimation is safe) and for cost
    fallback when the provider's response metadata lacks token counts.
    """
    if not text:
        return 0
    enc = _get_encoder()
    if enc is None:
        return len(text.encode("utf-8")) // BYTES_PER_TOKEN
    # disallowed_special=() — source files (e.g. llama.cpp, tokenizer code)
    # legitimately contain literals like "<|endoftext|>"; tiktoken's default
    # raises on them. We're counting, not feeding the model, so treat them
    # as ordinary text.
    return len(enc.encode(text, disallowed_special=()))


@lru_cache(maxsize=_TOKEN_CACHE_SIZE)
def _cached_file_tokens(path_key: str, size: int, _mtime_ns: int) -> int:
    """Token count for a file, memoized on (path, size, mtime).

    A single run estimates every file at least twice — ``--dry-run`` builds its
    per-file table and then ``create_batches`` packs the same list — and each
    estimate is a full tiktoken encode over the file's text. Keying on
    ``(size, _mtime_ns)`` alongside the path means an edit between calls
    invalidates the entry instead of returning a stale count.

    ``size`` and ``_mtime_ns`` are arguments rather than looked up here so the
    ``stat()`` happens exactly once per call in the caller, and so a caller
    holding a stat result can't race the cache key against the value.
    ``_mtime_ns`` is underscore-prefixed because it is *only* a cache-key
    component — nothing in the body reads it (vulture flags it otherwise).

    Only the resulting *count* is cached. Deliberately not the file's text:
    batches run concurrently in a ``ThreadPoolExecutor``, and holding every
    scanned file's contents alive for the duration of a run would trade a
    bounded number of re-reads for unbounded memory.
    """
    path = Path(path_key)
    encoder = _get_encoder()
    if encoder is not None and size <= _TIKTOKEN_MAX_BYTES:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return size // BYTES_PER_TOKEN + PER_FILE_OVERHEAD_TOKENS
        return (
            len(encoder.encode(text, disallowed_special=())) + PER_FILE_OVERHEAD_TOKENS
        )

    return size // BYTES_PER_TOKEN + PER_FILE_OVERHEAD_TOKENS


def clear_token_cache() -> None:
    """Drop all memoized per-file token counts.

    Only needed by tests that rewrite a file in place fast enough for its
    ``(size, mtime_ns)`` to be unchanged; real runs are invalidated by the
    cache key itself.
    """
    _cached_file_tokens.cache_clear()


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
        self.skipped_oversized: list[tuple[Path, int]] = []

    @staticmethod
    def estimate_file_tokens(file_path: Path) -> int:
        """Estimate token count for a file.

        Uses tiktoken (cl100k_base) for accuracy when available and the file
        is small enough to be worth encoding. Falls back to a UTF-8
        byte-based heuristic for huge files (avoids reading 2 MB+ files that
        would be skipped anyway) and when tiktoken is unavailable.

        Results are memoized per ``(path, size, mtime)`` — see
        :func:`_cached_file_tokens` — so the second and later estimates of the
        same unmodified file are free. ``--dry-run`` plus ``create_batches``
        already means two full encodes of every scanned file.

        Args:
            file_path: Path to the file

        Returns:
            Estimated token count (includes per-file overhead)
        """
        try:
            stat = file_path.stat()
        except OSError:
            return PER_FILE_OVERHEAD_TOKENS

        return _cached_file_tokens(str(file_path), stat.st_size, stat.st_mtime_ns)

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
        # Per-run state: clear stale skips from any earlier create_batches()
        # call on the same instance so reports don't conflate runs.
        self.skipped_oversized = []

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
        if self.token_budget is None:
            raise RuntimeError(
                "_batch_by_tokens called with token_budget=None; "
                "use _batch_by_count instead"
            )

        raw_batches: list[list[Path]] = []
        current_batch: list[Path] = []
        current_tokens = 0

        for file_path in files:
            file_tokens = self.estimate_file_tokens(file_path)

            # Oversized file: skip — it exceeds the token budget derived
            # from the model's context window and would fail at the API level.
            if file_tokens > self.token_budget:
                logging.warning(
                    "File %s estimated at %d tokens exceeds budget of %d tokens; "
                    "skipping (too large to review with this model)",
                    file_path,
                    file_tokens,
                    self.token_budget,
                )
                self.skipped_oversized.append((file_path, file_tokens))
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
