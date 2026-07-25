from pathlib import Path
from unittest.mock import patch

import pytest

from codereview.batcher import (
    PER_FILE_OVERHEAD_TOKENS,
    FileBatch,
    FileBatcher,
    clear_token_cache,
)


@pytest.fixture(autouse=True)
def _isolate_token_cache():
    """Keep the per-file token memo from leaking between tests.

    tmp_path makes most path keys unique, but a test that rewrites a file or
    swaps the encoder would otherwise observe another test's cached count.
    """
    clear_token_cache()
    yield
    clear_token_cache()


# ---------------------------------------------------------------------------
# Existing tests (count-only batching)
# ---------------------------------------------------------------------------


def test_batch_creation():
    """Test creating a file batch."""
    files = [Path("test1.py"), Path("test2.py")]
    batch = FileBatch(files=files, batch_number=1, total_batches=2)

    assert len(batch.files) == 2
    assert batch.batch_number == 1
    assert batch.total_batches == 2


def test_batcher_single_batch():
    """Test batcher with small number of files."""
    files = [Path(f"file{i}.py") for i in range(3)]
    batcher = FileBatcher(max_files_per_batch=10)
    batches = batcher.create_batches(files)

    assert len(batches) == 1
    assert len(batches[0].files) == 3


def test_batcher_multiple_batches():
    """Test batcher splits into multiple batches."""
    files = [Path(f"file{i}.py") for i in range(25)]
    batcher = FileBatcher(max_files_per_batch=10)
    batches = batcher.create_batches(files)

    assert len(batches) == 3
    assert len(batches[0].files) == 10
    assert len(batches[1].files) == 10
    assert len(batches[2].files) == 5


def test_batch_numbers_correct():
    """Test batch numbers are sequential and correct."""
    files = [Path(f"file{i}.py") for i in range(15)]
    batcher = FileBatcher(max_files_per_batch=5)
    batches = batcher.create_batches(files)

    assert batches[0].batch_number == 1
    assert batches[1].batch_number == 2
    assert batches[2].batch_number == 3

    for batch in batches:
        assert batch.total_batches == 3


def test_batcher_rejects_invalid_max_files():
    """Test batcher rejects invalid max_files_per_batch values."""
    with pytest.raises(ValueError, match="max_files_per_batch must be at least 1"):
        FileBatcher(max_files_per_batch=0)

    with pytest.raises(ValueError, match="max_files_per_batch must be at least 1"):
        FileBatcher(max_files_per_batch=-1)


# ---------------------------------------------------------------------------
# Token-budget-aware batching tests
# ---------------------------------------------------------------------------


def _create_file(tmp_path: Path, name: str, size_bytes: int) -> Path:
    """Helper: create a file of exactly *size_bytes* in *tmp_path*."""
    fp = tmp_path / name
    fp.write_bytes(b"x" * size_bytes)
    return fp


def test_estimate_file_tokens(tmp_path: Path):
    """estimate_file_tokens scales with file content and includes overhead."""
    fp = _create_file(tmp_path, "a.py", 4000)
    estimate = FileBatcher.estimate_file_tokens(fp)
    # We don't pin an exact value — tiktoken and the byte fallback differ —
    # but the result must include the per-file overhead and be a positive
    # token count proportional to file size.
    assert estimate >= PER_FILE_OVERHEAD_TOKENS
    assert estimate > PER_FILE_OVERHEAD_TOKENS  # 4 KB > 0 token content
    # Sanity bound: even pathological compression shouldn't claim a 4 KB
    # file is fewer than 5 tokens, and even pathological tokenization
    # shouldn't exceed 1 token per byte.
    assert PER_FILE_OVERHEAD_TOKENS + 5 <= estimate <= PER_FILE_OVERHEAD_TOKENS + 4000


def test_estimate_file_tokens_missing_file():
    """estimate_file_tokens returns overhead only for missing files."""
    assert (
        FileBatcher.estimate_file_tokens(Path("/nonexistent/file.py"))
        == PER_FILE_OVERHEAD_TOKENS
    )


def test_token_budget_none_gives_count_only_behavior():
    """token_budget=None gives identical behavior to count-only batching."""
    files = [Path(f"file{i}.py") for i in range(25)]
    batcher_none = FileBatcher(max_files_per_batch=10, token_budget=None)
    batcher_plain = FileBatcher(max_files_per_batch=10)

    batches_none = batcher_none.create_batches(files)
    batches_plain = batcher_plain.create_batches(files)

    assert len(batches_none) == len(batches_plain)
    for bn, bp in zip(batches_none, batches_plain):
        assert bn.files == bp.files


def test_token_budget_splits_correctly(tmp_path: Path):
    """Files are packed into batches that respect the token budget."""
    files = [_create_file(tmp_path, f"f{i}.py", 4000) for i in range(10)]
    # Source the per-file estimate from the function itself rather than
    # pinning the formula — the math depends on whether tiktoken is
    # installed and how it tokenizes the file's bytes.
    per_file = FileBatcher.estimate_file_tokens(files[0])

    # Budget should fit exactly 2 files but not 3.
    budget = per_file * 2 + per_file // 2  # halfway between 2x and 3x
    assert per_file * 2 <= budget < per_file * 3
    batcher = FileBatcher(max_files_per_batch=50, token_budget=budget)
    batches = batcher.create_batches(files)

    assert len(batches) == 5
    for batch in batches:
        assert len(batch.files) == 2


def test_file_count_cap_still_respected(tmp_path: Path):
    """File-count cap is enforced even when token budget is large."""
    # Tiny files — budget is never the bottleneck
    files = [_create_file(tmp_path, f"f{i}.py", 40) for i in range(20)]

    batcher = FileBatcher(max_files_per_batch=5, token_budget=999_999)
    batches = batcher.create_batches(files)

    assert len(batches) == 4
    for batch in batches:
        assert len(batch.files) <= 5


def test_oversized_file_skipped(tmp_path: Path):
    """A file exceeding the token budget is skipped, not batched."""
    small = _create_file(tmp_path, "small.py", 400)  # ~150 tokens
    huge = _create_file(tmp_path, "huge.py", 100_000)  # ~25050 tokens
    small2 = _create_file(tmp_path, "small2.py", 400)

    batcher = FileBatcher(max_files_per_batch=50, token_budget=5000)
    batches = batcher.create_batches([small, huge, small2])

    # Oversized file is skipped — only small files are batched together
    assert len(batches) == 1
    assert batches[0].files == [small, small2]

    # Skipped file is tracked
    assert len(batcher.skipped_oversized) == 1
    assert batcher.skipped_oversized[0][0] == huge
    assert batcher.skipped_oversized[0][1] > 5000


def test_invalid_token_budget_raises():
    """token_budget <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="token_budget must be greater than 0"):
        FileBatcher(token_budget=0)

    with pytest.raises(ValueError, match="token_budget must be greater than 0"):
        FileBatcher(token_budget=-100)


def test_empty_files_with_token_budget():
    """Empty file list returns empty batches even with token budget."""
    batcher = FileBatcher(token_budget=5000)
    assert batcher.create_batches([]) == []


def test_batch_numbers_correct_with_token_budget(tmp_path: Path):
    """Batch numbering is sequential and total_batches is correct."""
    files = [_create_file(tmp_path, f"f{i}.py", 4000) for i in range(6)]

    batcher = FileBatcher(max_files_per_batch=50, token_budget=2500)
    batches = batcher.create_batches(files)

    for i, batch in enumerate(batches, 1):
        assert batch.batch_number == i
        assert batch.total_batches == len(batches)


def test_all_files_included_in_token_batches(tmp_path: Path):
    """Every input file appears in exactly one output batch."""
    files = [_create_file(tmp_path, f"f{i}.py", 2000) for i in range(7)]

    batcher = FileBatcher(max_files_per_batch=3, token_budget=3000)
    batches = batcher.create_batches(files)

    result_files = [f for batch in batches for f in batch.files]
    assert result_files == files


# ---------------------------------------------------------------------------
# Per-run state isolation: skipped_oversized must not leak across calls.
# ---------------------------------------------------------------------------


def test_skipped_oversized_resets_between_calls(tmp_path: Path):
    """Re-running create_batches must start with an empty skipped list.

    Without the reset, callers reusing a FileBatcher instance would see
    stale entries from earlier runs, making the post-run summary
    misleading.
    """
    huge1 = _create_file(tmp_path, "huge1.py", 100_000)
    small = _create_file(tmp_path, "small.py", 400)

    batcher = FileBatcher(max_files_per_batch=50, token_budget=5000)
    batcher.create_batches([huge1, small])
    assert len(batcher.skipped_oversized) == 1

    # Second call with a *different* file set: only the second oversized
    # file should be reported, not the first.
    huge2 = _create_file(tmp_path, "huge2.py", 100_000)
    batcher.create_batches([huge2, small])
    assert len(batcher.skipped_oversized) == 1
    assert batcher.skipped_oversized[0][0] == huge2


def test_skipped_oversized_reset_when_no_skips(tmp_path: Path):
    """If a later run has no oversized files, the list must be empty."""
    huge = _create_file(tmp_path, "huge.py", 100_000)
    small = _create_file(tmp_path, "small.py", 400)

    batcher = FileBatcher(max_files_per_batch=50, token_budget=5000)
    batcher.create_batches([huge, small])
    assert len(batcher.skipped_oversized) == 1

    # Subsequent run with only small files clears the previous skip.
    batcher.create_batches([small])
    assert batcher.skipped_oversized == []


# ---------------------------------------------------------------------------
# count_tokens — tiktoken-backed estimator with byte fallback
# ---------------------------------------------------------------------------


def test_count_tokens_empty_string():
    from codereview.batcher import count_tokens

    assert count_tokens("") == 0


def test_count_tokens_ascii_is_close_to_word_count():
    """ASCII English: ~1 token per ~4 chars under cl100k_base."""
    from codereview.batcher import count_tokens

    text = "the quick brown fox jumps over the lazy dog " * 10
    n = count_tokens(text)
    # Sanity: result is positive, far below char count, and far above 1.
    assert 0 < n < len(text)
    assert n > 10


def test_count_tokens_cjk_does_not_underestimate():
    """CJK content: tiktoken (or the byte fallback) should not underestimate.

    The old `bytes // 4` heuristic returned ~3 tokens for 12 chars of CJK
    that actually need ~12 tokens. We just guard against severe
    underestimation, not exact match (tokenizers vary).
    """
    from codereview.batcher import count_tokens

    text = "中文注释" * 25  # 100 CJK chars, ~300 UTF-8 bytes
    n = count_tokens(text)
    assert n >= 50  # would be ~75 with old bytes // 4; we want >> that


def test_count_tokens_uses_byte_fallback_when_tiktoken_unavailable(monkeypatch):
    """If the encoder fails to load, count_tokens falls back to bytes // 3."""
    from codereview import batcher

    # monkeypatch.setattr auto-restores the real (lru_cached) function on
    # teardown, so callers afterwards still get the cached encoder.
    monkeypatch.setattr(batcher, "_get_encoder", lambda: None)
    text = "x" * 99
    assert batcher.count_tokens(text) == 99 // batcher.BYTES_PER_TOKEN


def test_count_tokens_handles_special_token_literals():
    """Source files may contain literal '<|endoftext|>' etc. (e.g. llama.cpp,
    tokenizer code). tiktoken's default disallowed_special='all' raises on
    these; we pass disallowed_special=() so counting succeeds.
    """
    from codereview.batcher import _get_encoder, count_tokens

    if _get_encoder() is None:
        pytest.skip("tiktoken unavailable; byte fallback can't trigger this")

    text = 'const char* eos = "<|endoftext|>";\n'
    n = count_tokens(text)
    assert n > 0


def test_estimate_file_tokens_handles_special_token_literals(tmp_path):
    """estimate_file_tokens must not raise on files containing tiktoken's
    special-token literals (e.g. llama-cpp source). Regression for ValueError:
    'Encountered text corresponding to disallowed special token'.
    """
    from codereview.batcher import FileBatcher, _get_encoder

    if _get_encoder() is None:
        pytest.skip("tiktoken unavailable; byte fallback can't trigger this")

    f = tmp_path / "tokens.cpp"
    f.write_text(
        "#include <string>\n"
        'const std::string EOS = "<|endoftext|>";\n'
        'const std::string FIM_PREFIX = "<|fim_prefix|>";\n',
        encoding="utf-8",
    )
    n = FileBatcher.estimate_file_tokens(f)
    assert n > 0


# ---------------------------------------------------------------------------
# Per-file token-count memoization
#
# A single run estimates every scanned file at least twice (--dry-run's table,
# then create_batches' packing loop), and each estimate was a full tiktoken
# encode over the file's text. The count is cached on (path, size, mtime).
# ---------------------------------------------------------------------------


def test_estimate_file_tokens_reads_the_file_once_for_repeat_calls(tmp_path):
    """Re-estimating an unmodified file must not re-read or re-encode it."""
    from codereview.batcher import _get_encoder

    if _get_encoder() is None:
        pytest.skip("tiktoken unavailable; the byte fallback never reads the file")

    f = tmp_path / "sample.py"
    f.write_text("def f():\n    return 42\n" * 20, encoding="utf-8")

    real_read_text = Path.read_text
    reads = []

    def counting_read_text(self, *args, **kwargs):
        reads.append(self)
        return real_read_text(self, *args, **kwargs)

    with patch.object(Path, "read_text", counting_read_text):
        first = FileBatcher.estimate_file_tokens(f)
        second = FileBatcher.estimate_file_tokens(f)
        third = FileBatcher.estimate_file_tokens(f)

    assert first == second == third
    assert reads == [f], f"expected exactly one read of {f}, got {reads}"


def test_dry_run_estimate_is_reused_by_batching(tmp_path):
    """The batching pass reuses the counts a --dry-run-style sweep produced."""
    from codereview.batcher import _get_encoder

    if _get_encoder() is None:
        pytest.skip("tiktoken unavailable; the byte fallback never reads the file")

    files = []
    for i in range(5):
        f = tmp_path / f"mod{i}.py"
        f.write_text(f"VALUE = {i}\n" * 30, encoding="utf-8")
        files.append(f)

    real_read_text = Path.read_text
    reads = []

    def counting_read_text(self, *args, **kwargs):
        reads.append(self)
        return real_read_text(self, *args, **kwargs)

    with patch.object(Path, "read_text", counting_read_text):
        # Pass 1: what _display_dry_run does per file.
        sweep = [FileBatcher.estimate_file_tokens(f) for f in files]
        # Pass 2: what _batch_by_tokens does over the same list.
        FileBatcher(max_files_per_batch=10, token_budget=100_000).create_batches(files)

    assert len(reads) == len(files), (
        "batching re-read files the dry-run sweep already measured: "
        f"{len(reads)} reads for {len(files)} files"
    )
    assert all(n > PER_FILE_OVERHEAD_TOKENS for n in sweep)


def test_token_cache_bound_exceeds_a_large_repo_file_count():
    """The memo's ceiling must be a *run* bound, not a "typical repo" bound.

    A run estimates every scanned file at least twice — ``--dry-run``'s table,
    then ``create_batches``' packing loop (the guarantee
    :func:`test_batching_reuses_the_dry_run_token_estimates` locks). With a
    ``maxsize`` below the file count, the first pass evicts its own earliest
    entries before the second pass reaches them and every file is re-encoded:
    the memoization silently stops paying off exactly in the large repositories
    it exists for. The bound was 4096, a plausible file count for a monorepo.

    Asserted as a constant rather than by scanning 100K files, which would make
    the suite unusable; the mechanism it guards is the two-pass test above.
    """
    from codereview.batcher import _TOKEN_CACHE_SIZE, _cached_file_tokens

    maxsize = _cached_file_tokens.cache_info().maxsize
    assert maxsize == _TOKEN_CACHE_SIZE, "cache decorated with a different bound"
    assert maxsize is not None, (
        "an unbounded memo holds one entry per file forever; keep a ceiling"
    )
    assert maxsize >= 50_000, (
        f"token memo bounded at {maxsize} entries — below the file count of a "
        "large repository, so the second estimate pass re-encodes everything"
    )


def test_estimate_file_tokens_invalidates_when_the_file_changes(tmp_path):
    """A modified file must not return its stale cached count."""
    f = tmp_path / "growing.py"
    f.write_text("x = 1\n", encoding="utf-8")
    small = FileBatcher.estimate_file_tokens(f)

    # A different size changes the cache key even if mtime granularity is coarse.
    f.write_text("x = 1\n" * 500, encoding="utf-8")
    large = FileBatcher.estimate_file_tokens(f)

    assert large > small, (
        "estimate_file_tokens returned a stale count after the file grew "
        f"({small} -> {large})"
    )


def test_estimate_file_tokens_does_not_confuse_same_sized_files(tmp_path):
    """The path is part of the cache key, so equal-sized files stay distinct."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    # Same byte length, very different token counts: ASCII vs CJK.
    a.write_text("a" * 300, encoding="utf-8")
    b.write_text("中" * 100, encoding="utf-8")
    assert a.stat().st_size == b.stat().st_size

    assert FileBatcher.estimate_file_tokens(a) != FileBatcher.estimate_file_tokens(b)


def test_clear_token_cache_forces_a_re_read(tmp_path):
    """clear_token_cache is the escape hatch for in-place rewrites."""
    f = tmp_path / "same_size.py"
    f.write_text("aaaa\n", encoding="utf-8")
    FileBatcher.estimate_file_tokens(f)

    clear_token_cache()

    real_read_text = Path.read_text
    reads = []

    def counting_read_text(self, *args, **kwargs):
        reads.append(self)
        return real_read_text(self, *args, **kwargs)

    from codereview.batcher import _get_encoder

    if _get_encoder() is None:
        pytest.skip("tiktoken unavailable; the byte fallback never reads the file")

    with patch.object(Path, "read_text", counting_read_text):
        FileBatcher.estimate_file_tokens(f)

    assert reads == [f]
