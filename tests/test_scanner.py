# tests/test_scanner.py
import pytest
from pathlib import Path
from codereview.scanner import FileScanner


@pytest.fixture
def sample_dir():
    """Path to test fixtures."""
    return Path(__file__).parent / "fixtures" / "sample_code"


def test_scanner_finds_python_files(sample_dir):
    """Test scanner finds .py files."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    py_files = [f for f in files if f.suffix == ".py"]
    assert len(py_files) > 0
    assert any("main.py" in str(f) for f in py_files)


def test_scanner_finds_go_files(sample_dir):
    """Test scanner finds .go files."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    go_files = [f for f in files if f.suffix == ".go"]
    assert len(go_files) > 0
    assert any("main.go" in str(f) for f in go_files)


def test_scanner_excludes_json(sample_dir):
    """Test scanner excludes .json files."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    json_files = [f for f in files if f.suffix == ".json"]
    assert len(json_files) == 0


def test_scanner_excludes_venv(sample_dir):
    """Test scanner excludes .venv directory."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    venv_files = [f for f in files if ".venv" in str(f)]
    assert len(venv_files) == 0


def test_scanner_excludes_pycache(sample_dir):
    """Test scanner excludes __pycache__."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    cache_files = [f for f in files if "__pycache__" in str(f)]
    assert len(cache_files) == 0
