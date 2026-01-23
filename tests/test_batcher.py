from pathlib import Path

from codereview.batcher import FileBatch, FileBatcher


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
