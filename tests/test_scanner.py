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


def test_scanner_excludes_hidden_dirs_by_default(tmp_path):
    """Hidden directories like .github/ are skipped by default."""
    hidden = tmp_path / ".github" / "scripts"
    hidden.mkdir(parents=True)
    (hidden / "release.py").write_text("x = 1\n")

    scanner = FileScanner(tmp_path)
    files = scanner.scan()

    assert all(".github" not in p.parts for p in files)


def test_scanner_includes_hidden_dirs_when_opted_in(tmp_path):
    """exclude_hidden=False lets users scan inside .github/, .config/, etc."""
    hidden = tmp_path / ".github" / "scripts"
    hidden.mkdir(parents=True)
    target = hidden / "release.py"
    target.write_text("x = 1\n")

    scanner = FileScanner(tmp_path, exclude_hidden=False)
    files = scanner.scan()

    assert target.resolve() in [f.resolve() for f in files]


def test_finegrained_exclude_pattern_does_not_prune_directory(tmp_path):
    """A fine-grained exclude pattern must not prune an entire directory.

    Regression: ``_get_excluded_dir_names`` previously added every literal
    pattern segment to the pruned-directory set, so an exclude like
    ``src/generated.py`` would skip the whole ``src/`` tree, dropping
    unrelated source files from the review.
    """
    src = tmp_path / "src"
    src.mkdir()
    keep = src / "app.py"
    keep.write_text("x = 1\n")
    (src / "generated.py").write_text("y = 2\n")

    scanner = FileScanner(tmp_path, exclude_patterns=["src/generated.py"])
    files = scanner.scan()

    resolved = [f.resolve() for f in files]
    # The directory must NOT be pruned: app.py is still reviewed...
    assert keep.resolve() in resolved
    # ...while the specifically-excluded file is dropped.
    assert (src / "generated.py").resolve() not in resolved


def test_directory_exclude_pattern_still_prunes(tmp_path):
    """A ``**/dir/**`` pattern still prunes the whole directory (no regression)."""
    build = tmp_path / "build"
    build.mkdir()
    (build / "out.py").write_text("x = 1\n")
    keep = tmp_path / "main.py"
    keep.write_text("y = 2\n")

    scanner = FileScanner(tmp_path, exclude_patterns=["**/build/**"])
    files = scanner.scan()

    resolved = [f.resolve() for f in files]
    assert keep.resolve() in resolved
    assert all("build" not in p.parts for p in files)
