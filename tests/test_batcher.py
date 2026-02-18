from pathlib import Path

import pytest

from codereview.batcher import PER_FILE_OVERHEAD_TOKENS, FileBatch, FileBatcher

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
    """estimate_file_tokens returns file_size//4 + overhead."""
    fp = _create_file(tmp_path, "a.py", 4000)
    assert FileBatcher.estimate_file_tokens(fp) == 4000 // 4 + PER_FILE_OVERHEAD_TOKENS


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
    # Each file: 4000 bytes -> 1000 tokens + 50 overhead = 1050 tokens
    files = [_create_file(tmp_path, f"f{i}.py", 4000) for i in range(10)]

    # Budget of 2500 should fit 2 files (2*1050 = 2100 < 2500) but not 3 (3150 > 2500)
    batcher = FileBatcher(max_files_per_batch=50, token_budget=2500)
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


def test_oversized_file_gets_own_batch(tmp_path: Path):
    """A file exceeding the token budget is placed in a single-file batch."""
    small = _create_file(tmp_path, "small.py", 400)  # ~150 tokens
    huge = _create_file(tmp_path, "huge.py", 100_000)  # ~25050 tokens
    small2 = _create_file(tmp_path, "small2.py", 400)

    batcher = FileBatcher(max_files_per_batch=50, token_budget=5000)
    batches = batcher.create_batches([small, huge, small2])

    assert len(batches) == 3
    # First batch: small file
    assert batches[0].files == [small]
    # Second batch: oversized file alone
    assert batches[1].files == [huge]
    # Third batch: remaining small file
    assert batches[2].files == [small2]


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
