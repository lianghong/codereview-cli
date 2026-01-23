# tests/test_scanner.py
from pathlib import Path

import pytest

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


def test_scanner_finds_shell_scripts(sample_dir):
    """Test scanner finds .sh and .bash files."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    sh_files = [f for f in files if f.suffix in (".sh", ".bash")]
    assert len(sh_files) > 0
    assert any("setup.sh" in str(f) for f in sh_files)


def test_scanner_finds_cpp_files(sample_dir):
    """Test scanner finds C++ files (.cpp, .cc, .cxx, .h, .hpp)."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    cpp_extensions = {".cpp", ".cc", ".cxx", ".h", ".hpp"}
    cpp_files = [f for f in files if f.suffix in cpp_extensions]
    assert len(cpp_files) > 0
    assert any("example.cpp" in str(f) for f in cpp_files)


def test_scanner_finds_java_files(sample_dir):
    """Test scanner finds .java files."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    java_files = [f for f in files if f.suffix == ".java"]
    assert len(java_files) > 0
    assert any("Example.java" in str(f) for f in java_files)


def test_scanner_finds_javascript_files(sample_dir):
    """Test scanner finds JavaScript files (.js, .jsx, .mjs)."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    js_extensions = {".js", ".jsx", ".mjs"}
    js_files = [f for f in files if f.suffix in js_extensions]
    assert len(js_files) > 0
    assert any("example.js" in str(f) for f in js_files)


def test_scanner_finds_typescript_files(sample_dir):
    """Test scanner finds TypeScript files (.ts, .tsx)."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    ts_extensions = {".ts", ".tsx"}
    ts_files = [f for f in files if f.suffix in ts_extensions]
    assert len(ts_files) > 0
    assert any("example.ts" in str(f) for f in ts_files)
