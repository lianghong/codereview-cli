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


def test_path_qualified_exclude_does_not_prune_same_named_dir_elsewhere(tmp_path):
    """``docs/api/*`` must not prune an unrelated ``app/api/``.

    Regression: ``_get_excluded_dir_names`` extracted the bare name of any
    segment followed by a wildcard, so a path-qualified pattern contributed its
    last directory name to the prune set. Pruning is by bare name and therefore
    matches at any depth, so ``docs/api/*`` silently skipped every ``api``
    directory in the tree — dropping files from the review with no warning.
    """
    excluded = tmp_path / "docs" / "api"
    excluded.mkdir(parents=True)
    (excluded / "generated.py").write_text("x = 1\n")

    unrelated = tmp_path / "app" / "api"
    unrelated.mkdir(parents=True)
    keep = unrelated / "service.py"
    keep.write_text("y = 2\n")

    scanner = FileScanner(tmp_path, exclude_patterns=["docs/api/*"])
    resolved = [f.resolve() for f in scanner.scan()]

    # The unrelated same-named directory is still reviewed...
    assert keep.resolve() in resolved
    # ...while the pattern's actual target is still excluded (by _is_excluded,
    # since the directory is now walked rather than pruned).
    assert (excluded / "generated.py").resolve() not in resolved


def test_deeply_qualified_exclude_does_not_prune_its_leaf_name(tmp_path):
    """``a/b/c/**`` must not prune every directory named ``c``."""
    keep_dir = tmp_path / "x" / "c"
    keep_dir.mkdir(parents=True)
    keep = keep_dir / "mod.py"
    keep.write_text("x = 1\n")

    scanner = FileScanner(tmp_path, exclude_patterns=["a/b/c/**"])
    resolved = [f.resolve() for f in scanner.scan()]

    assert keep.resolve() in resolved


def test_wildcard_prefixed_exclude_still_prunes(tmp_path):
    """A pattern whose prefix is all wildcards is unanchored, so it may prune.

    ``*/build/**`` names ``build`` at any location one level down, which a
    bare-name prune expresses correctly — the traversal saving must survive the
    fix above.
    """
    build = tmp_path / "pkg" / "build"
    build.mkdir(parents=True)
    (build / "out.py").write_text("x = 1\n")
    keep = tmp_path / "pkg" / "main.py"
    keep.write_text("y = 2\n")

    scanner = FileScanner(tmp_path, exclude_patterns=["*/build/**"])

    assert "build" in scanner._get_excluded_dir_names()
    resolved = [f.resolve() for f in scanner.scan()]
    assert keep.resolve() in resolved
    assert all("build" not in p.parts for p in resolved)


@pytest.mark.parametrize(
    "relative_path",
    [
        # No leading segment: PurePath.match("**/node_modules/**") requires one.
        "node_modules/index.py",
        # One leading segment: the only depth bare match() ever covered.
        "pkg/node_modules/index.py",
        # Deeper than one segment on either side — both were missed.
        "pkg/sub/node_modules/index.py",
        "pkg/node_modules/dep/lib/index.py",
        "a/b/c/node_modules/d/e/index.py",
    ],
)
def test_is_excluded_matches_a_recursive_pattern_at_every_depth(relative_path):
    """``**/node_modules/**`` must exclude the directory's contents at any depth.

    ``PurePath.match`` is right-anchored and treats ``**`` as a *single*
    segment, so the pattern read as "one segment, node_modules, one segment":
    only ``pkg/node_modules/index.py`` matched. Every other depth — including
    the top-level ``node_modules/`` and anything nested inside a vendored
    dependency — was *eligible for review*. The prune set masks this during a
    plain ``scan()``, which is exactly why it needs testing at this level: a
    path-qualified pattern contributes no prune name, so there the glob is the
    only defense.
    """
    scanner = FileScanner(".", exclude_patterns=["**/node_modules/**"])
    assert scanner._is_excluded(relative_path) is True


def test_is_excluded_does_not_widen_a_path_qualified_pattern():
    """Adding ``full_match`` must not make ``docs/api/**`` reach a sibling tree.

    ``full_match`` is whole-path, so it is *stricter* than ``match`` wherever
    the pattern has no recursive ``**``; the union can only add the recursive
    case. This pins that: the fix buys depth, not breadth.

    (``other/docs/api/x.py`` is deliberately not asserted here — ``match`` is
    right-anchored, so it excluded that path long before ``full_match`` was
    added. That is the pattern's pre-existing meaning, not a widening.)
    """
    scanner = FileScanner(".", exclude_patterns=["docs/api/**"])
    assert scanner._is_excluded("app/api/views.py") is False


@pytest.mark.parametrize(
    "relative_path",
    [
        "main.py",
        "src/main.py",
        "app/api/views.py",
        "pkg/service/handler.go",
        "cmd/server/main.go",
        "tests/test_api.py",
        "internal/build_config.py",
        "scripts/deploy.sh",
        "src/vendored_client.py",
    ],
)
def test_default_patterns_exclude_no_ordinary_source_file(relative_path):
    """The shipped defaults must not exclude a file a user expects reviewed.

    Widening ``_is_excluded`` to the ``match``/``full_match`` union buys depth
    on ``**/dir/**`` patterns; the failure mode to guard is that it also buys
    *breadth*. A false positive here is invisible at runtime — the file is
    never scanned, never counted, and never reported as skipped — so it is
    pinned against the real ``DEFAULT_EXCLUDE_PATTERNS`` rather than argued
    from the predicates' semantics. Note the deliberately adversarial names:
    ``build_config.py`` and ``vendored_client.py`` must survive ``**/build/**``
    and ``**/vendor/**`` (segment match, not substring).
    """
    scanner = FileScanner(".")
    assert scanner._is_excluded(relative_path) is False


def test_single_star_still_means_exactly_one_segment():
    """``docs/api/*`` keeps its literal meaning under the union.

    Documented behavior (see docs/architecture.md): there is no path-qualified spelling
    that covers an arbitrary-depth subtree, and ``docs/api/*`` deliberately
    does not reach ``docs/api/sub/x.py``. ``full_match`` treats a single ``*``
    as one segment too, so this must not change.
    """
    scanner = FileScanner(".", exclude_patterns=["docs/api/*"])

    assert scanner._is_excluded("docs/api/generated.py") is True
    assert scanner._is_excluded("docs/api/sub/x.py") is False


def test_deep_vendored_files_are_excluded_end_to_end(tmp_path):
    """The scan must not report a deeply nested vendored file.

    Drives ``scan()`` with a *path-qualified* pattern, so the directory is
    walked rather than pruned and ``_is_excluded`` is the only thing standing
    between the review and 40k lines of someone else's code.
    """
    deep = tmp_path / "web" / "node_modules" / "left-pad" / "src"
    deep.mkdir(parents=True)
    (deep / "index.py").write_text("x = 1\n")
    keep = tmp_path / "web" / "app.py"
    keep.write_text("y = 2\n")

    scanner = FileScanner(tmp_path, exclude_patterns=["web/node_modules/**"])
    resolved = [f.resolve() for f in scanner.scan()]

    assert keep.resolve() in resolved
    assert (deep / "index.py").resolve() not in resolved


def test_default_patterns_all_still_prune_their_directories():
    """Every default ``**/dir/**`` pattern must still contribute a prune name.

    The prune set is a pure traversal optimization, but losing an entry for a
    directory like ``node_modules`` would mean walking (and stat-ing) a huge
    tree on every scan. This pins the optimization to the defaults rather than
    to one hand-picked example.
    """
    from pathlib import PurePath

    from codereview.config import DEFAULT_EXCLUDE_PATTERNS

    pruned = FileScanner(".")._get_excluded_dir_names()
    expected = {
        PurePath(p).parts[1]
        for p in DEFAULT_EXCLUDE_PATTERNS
        if PurePath(p).parts[:1] == ("**",)
        and len(PurePath(p).parts) == 3
        and PurePath(p).parts[2] in ("**", "*")
    }
    assert expected, "no directory-style default patterns found; the scan is broken"
    assert expected <= pruned, f"stopped pruning {sorted(expected - pruned)}"
