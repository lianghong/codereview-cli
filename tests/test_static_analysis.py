"""Tests for static analysis integration."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codereview.static_analysis import StaticAnalysisResult, StaticAnalyzer


@pytest.fixture
def sample_directory(tmp_path):
    """Create a sample directory with Python files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create a simple Python file
    (src_dir / "test.py").write_text("""
def hello():
    print("Hello World")
""")

    return src_dir


def test_static_analyzer_initialization(sample_directory):
    """Test StaticAnalyzer initialization."""
    analyzer = StaticAnalyzer(sample_directory)
    assert analyzer.directory == sample_directory
    assert isinstance(analyzer.available_tools, list)


def test_check_available_tools():
    """Test checking available tools."""
    analyzer = StaticAnalyzer(Path("."))
    # Should return a list (might be empty if tools not installed)
    assert isinstance(analyzer.available_tools, list)


def test_resolve_tool_binary_rejects_in_repo_executable(tmp_path):
    """Binaries that resolve inside the analyzed directory must be refused.

    A repo could ship its own ``ruff``/``eslint`` (e.g. via node_modules/.bin)
    and have it run with the user's privileges. Defend against that.
    """
    # Plant a fake "evil-tool" inside the analyzed directory.
    fake_bin = tmp_path / "node_modules" / ".bin"
    fake_bin.mkdir(parents=True)
    evil = fake_bin / "evil-tool"
    evil.write_text("#!/bin/sh\necho pwned\n")
    evil.chmod(0o755)

    analyzer = StaticAnalyzer(tmp_path)

    with patch("shutil.which", return_value=str(evil)):
        assert analyzer._resolve_tool_binary("evil-tool") is None


def test_resolve_tool_binary_accepts_system_executable(tmp_path):
    """Binaries on the system PATH (outside the repo) are returned resolved."""
    # /bin/sh is on every POSIX system and is outside any tmp_path.
    analyzer = StaticAnalyzer(tmp_path)
    resolved = analyzer._resolve_tool_binary("sh")
    assert resolved is not None
    assert Path(resolved).is_absolute()


def test_resolve_tool_binary_returns_none_when_missing(tmp_path):
    """Tools not on PATH should resolve to None, not raise."""
    analyzer = StaticAnalyzer(tmp_path)
    assert analyzer._resolve_tool_binary("definitely-not-a-real-binary-xyz123") is None


def test_static_analysis_result_creation():
    """Test creating StaticAnalysisResult."""
    result = StaticAnalysisResult(
        tool="ruff", passed=True, issues_count=0, output="All checks passed", errors=[]
    )

    assert result.tool == "ruff"
    assert result.passed is True
    assert result.issues_count == 0
    assert result.output == "All checks passed"
    assert len(result.errors) == 0


def test_static_analysis_result_with_errors():
    """Test StaticAnalysisResult with errors."""
    result = StaticAnalysisResult(
        tool="mypy", passed=False, issues_count=5, output="", errors=["Tool not found"]
    )

    assert result.passed is False
    assert result.issues_count == 5
    assert len(result.errors) == 1


@patch("subprocess.run")
def test_run_tool_success(mock_subprocess, sample_directory):
    """Test running a tool successfully."""
    # Mock successful tool execution
    mock_subprocess.return_value = Mock(
        returncode=0, stdout="All checks passed", stderr=""
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["ruff"]  # Force tool to be available

    result = analyzer.run_tool("ruff")

    assert result.tool == "ruff"
    assert result.passed is True


@patch("subprocess.run")
def test_run_tool_with_issues(mock_subprocess, sample_directory):
    """Test running a tool that finds issues."""
    # Mock tool finding issues
    mock_subprocess.return_value = Mock(
        returncode=1, stdout="error: line too long\nwarning: unused import\n", stderr=""
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["ruff"]

    result = analyzer.run_tool("ruff")

    assert result.tool == "ruff"
    assert result.passed is False
    assert result.issues_count > 0


@patch("subprocess.run")
def test_gofmt_failure_is_not_reported_as_passed(mock_subprocess, tmp_path):
    """Regression: a gofmt error (non-zero exit, empty stdout) must NOT pass.

    Previously pass/fail keyed only on empty stdout, so a gofmt invocation that
    errored (writing to stderr and exiting non-zero with empty stdout) was
    falsely reported as formatted — letting broken/unformatted Go through CI.
    """
    (tmp_path / "main.go").write_text("package main\n")

    mock_subprocess.return_value = Mock(
        returncode=2, stdout="", stderr="gofmt: main.go: expected declaration\n"
    )

    analyzer = StaticAnalyzer(tmp_path)
    analyzer.available_tools = ["gofmt"]

    result = analyzer.run_tool("gofmt")

    assert result.tool == "gofmt"
    assert result.passed is False


@patch("subprocess.run")
def test_gofmt_clean_run_passes(mock_subprocess, tmp_path):
    """A clean gofmt run (exit 0, empty stdout) still passes."""
    (tmp_path / "main.go").write_text("package main\n")

    mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

    analyzer = StaticAnalyzer(tmp_path)
    analyzer.available_tools = ["gofmt"]

    result = analyzer.run_tool("gofmt")

    assert result.passed is True


@patch("subprocess.run")
def test_gofmt_invoked_on_directory_not_flat_file_list(mock_subprocess, tmp_path):
    """Guard: gofmt is run against the directory so it recurses into nested
    packages. `gofmt -l <dir>` walks the tree (verified separately), so passing
    the directory covers pkg/service/main.go etc. This locks that in: a future
    refactor to a flat file list (which would NOT recurse the same way and
    could hit command-line-length caps) is caught here.
    """
    nested = tmp_path / "pkg" / "service"
    nested.mkdir(parents=True)
    (nested / "main.go").write_text("package service\n")
    (tmp_path / "root.go").write_text("package main\n")

    mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

    analyzer = StaticAnalyzer(tmp_path)
    analyzer.available_tools = ["gofmt"]

    analyzer.run_tool("gofmt")

    # The command must include the analysis directory as a target, not the
    # individual .go file paths.
    command = mock_subprocess.call_args.args[0]
    assert str(tmp_path) in command
    assert not any(str(arg).endswith(".go") for arg in command), (
        "gofmt should receive the directory, not a flat .go file list"
    )


def test_count_issues_ruff_summary_line():
    """ruff prints `Found N errors.` — extract N exactly, don't double-count.

    Without the summary regex, the substring fallback would count both each
    diagnostic line AND the "Found 3 errors" summary line as 4 hits.
    """
    output = (
        "bad.py:1:8: F401 [*] `os` imported but unused\n"
        "bad.py:2:1: E302 expected 2 blank lines, found 1\n"
        "bad.py:3:5: E501 line too long\n"
        "Found 3 errors.\n"
    )
    assert StaticAnalyzer._count_issues("ruff", "python", output) == 3


def test_count_issues_mypy_summary_line():
    """mypy prints `Found N errors in M files`."""
    output = (
        "foo.py:2: error: Incompatible return value type  [return-value]\n"
        "Found 1 error in 1 file (checked 1 source file)\n"
    )
    assert StaticAnalyzer._count_issues("mypy", "python", output) == 1


def test_count_issues_mypy_no_summary_falls_back_to_regex():
    """When the summary is absent, count diagnostic lines via regex."""
    output = "foo.py:2: error: A\nfoo.py:5:8: error: B\nbar.py:10: warning: C\n"
    assert StaticAnalyzer._count_issues("mypy", "python", output) == 3


def test_count_issues_bandit_uses_issue_markers():
    """bandit counts >> Issue: markers, one per finding."""
    output = (
        ">> Issue: [B602:subprocess_popen_with_shell_equals_true] ...\n"
        "   Severity: High   Confidence: High\n"
        ">> Issue: [B608:hardcoded_sql_expressions] ...\n"
        "   Severity: Medium Confidence: Low\n"
    )
    assert StaticAnalyzer._count_issues("bandit", "python", output) == 2


def test_count_issues_unknown_tool_falls_back_to_substring():
    """For tools without a known summary format, substring counting is the floor."""
    output = "error: x\nwarning: y\nclean line\n"
    # Two indicator-bearing lines.
    assert StaticAnalyzer._count_issues("unknown-tool", "python", output) == 2


def test_run_tool_not_available(sample_directory):
    """Test running a tool that's not available."""
    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = []  # No tools available

    result = analyzer.run_tool("ruff")

    assert result.passed is False
    assert len(result.errors) > 0
    assert "not installed" in result.errors[0]


def test_run_unknown_tool(sample_directory):
    """Test running an unknown tool."""
    analyzer = StaticAnalyzer(sample_directory)

    result = analyzer.run_tool("unknown_tool")

    assert result.passed is False
    assert len(result.errors) > 0
    assert "Unknown tool" in result.errors[0]


@patch("subprocess.run")
def test_run_all_tools(mock_subprocess, sample_directory):
    """Test running all available tools."""
    # Mock all tools passing
    mock_subprocess.return_value = Mock(
        returncode=0, stdout="All checks passed", stderr=""
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["ruff", "mypy"]

    results = analyzer.run_all()

    assert len(results) == 2
    assert "ruff" in results
    assert "mypy" in results
    assert all(r.passed for r in results.values())


@patch("subprocess.run")
def test_run_all_tools_parallel(mock_subprocess, sample_directory):
    """Test running all tools in parallel."""
    mock_subprocess.return_value = Mock(
        returncode=0, stdout="All checks passed", stderr=""
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["ruff", "mypy", "black"]

    # Parallel execution (default)
    results = analyzer.run_all(parallel=True)

    assert len(results) == 3
    assert all(r.passed for r in results.values())


@patch("subprocess.run")
def test_run_all_tools_sequential(mock_subprocess, sample_directory):
    """Test running all tools sequentially."""
    mock_subprocess.return_value = Mock(
        returncode=0, stdout="All checks passed", stderr=""
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["ruff", "mypy"]

    # Sequential execution
    results = analyzer.run_all(parallel=False)

    assert len(results) == 2
    assert all(r.passed for r in results.values())


def test_get_summary_all_passed(sample_directory):
    """Test summary when all tools pass."""
    analyzer = StaticAnalyzer(sample_directory)

    results = {
        "ruff": StaticAnalysisResult("ruff", True, 0, "", []),
        "mypy": StaticAnalysisResult("mypy", True, 0, "", []),
    }

    summary = analyzer.get_summary(results)

    assert summary["tools_run"] == 2
    assert summary["tools_passed"] == 2
    assert summary["tools_failed"] == 0
    assert summary["total_issues"] == 0
    assert summary["passed"] is True


def test_get_summary_with_failures(sample_directory):
    """Test summary when some tools fail."""
    analyzer = StaticAnalyzer(sample_directory)

    results = {
        "ruff": StaticAnalysisResult("ruff", False, 5, "", []),
        "mypy": StaticAnalysisResult("mypy", True, 0, "", []),
    }

    summary = analyzer.get_summary(results)

    assert summary["tools_run"] == 2
    assert summary["tools_passed"] == 1
    assert summary["tools_failed"] == 1
    assert summary["total_issues"] == 5
    assert summary["passed"] is False


def test_tools_configuration():
    """Test that all tools have proper configuration."""
    analyzer = StaticAnalyzer(Path("."))

    for tool_name, config in analyzer.TOOLS.items():
        assert "name" in config
        assert "description" in config
        assert "command" in config
        assert isinstance(config["command"], list)
        assert len(config["command"]) > 0


@patch("subprocess.run")
def test_run_vulture_with_dead_code(mock_subprocess, sample_directory):
    """Test running vulture when it finds unused code."""
    # Mock vulture finding dead code
    mock_subprocess.return_value = Mock(
        returncode=1,
        stdout="test.py:10: unused function 'old_func' (60% confidence)\ntest.py:20: unused variable 'x' (100% confidence)\n",
        stderr="",
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["vulture"]

    result = analyzer.run_tool("vulture")

    assert result.tool == "vulture"
    assert result.passed is False
    assert result.issues_count == 2  # Two "unused" lines


def test_vulture_in_tools():
    """Test that vulture is included in TOOLS."""
    assert "vulture" in StaticAnalyzer.TOOLS
    vulture_config = StaticAnalyzer.TOOLS["vulture"]
    assert vulture_config["name"] == "Vulture"
    assert vulture_config["description"] == "Dead code finder"
    assert vulture_config["command"] == ["vulture", "--min-confidence", "80"]


# Go static analysis tests


def test_go_tools_in_tools():
    """Test that Go tools are included in TOOLS."""
    # golangci-lint
    assert "golangci-lint" in StaticAnalyzer.TOOLS
    golangci_config = StaticAnalyzer.TOOLS["golangci-lint"]
    assert golangci_config["name"] == "golangci-lint"
    assert golangci_config["description"] == "Go meta-linter"
    assert golangci_config["command"] == ["golangci-lint", "run"]
    assert golangci_config["language"] == "go"

    # go vet
    assert "go-vet" in StaticAnalyzer.TOOLS
    govet_config = StaticAnalyzer.TOOLS["go-vet"]
    assert govet_config["name"] == "go vet"
    assert govet_config["description"] == "Go static analyzer"
    assert govet_config["command"] == ["go", "vet"]
    assert govet_config["language"] == "go"
    assert govet_config["version_command"] == ["go", "version"]

    # gofmt
    assert "gofmt" in StaticAnalyzer.TOOLS
    gofmt_config = StaticAnalyzer.TOOLS["gofmt"]
    assert gofmt_config["name"] == "gofmt"
    assert gofmt_config["description"] == "Go formatter"
    assert gofmt_config["command"] == ["gofmt", "-l"]
    assert gofmt_config["language"] == "go"


@pytest.fixture
def go_directory(tmp_path):
    """Create a sample directory with Go files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create a simple Go file
    (src_dir / "main.go").write_text("""package main

func main() {
    println("Hello World")
}
""")

    return src_dir


@patch("subprocess.run")
def test_run_golangci_lint_with_issues(mock_subprocess, go_directory):
    """Test running golangci-lint when it finds issues."""
    mock_subprocess.return_value = Mock(
        returncode=1,
        stdout="main.go:10:5: ineffassign: assigned to x but never used\nmain.go:15:1: deadcode: unused function\n",
        stderr="",
    )

    analyzer = StaticAnalyzer(go_directory)
    analyzer.available_tools = ["golangci-lint"]

    result = analyzer.run_tool("golangci-lint")

    assert result.tool == "golangci-lint"
    assert result.passed is False
    assert result.issues_count == 2  # ineffassign + deadcode


@patch("subprocess.run")
def test_run_go_vet_with_issues(mock_subprocess, go_directory):
    """Test running go vet when it finds issues."""
    mock_subprocess.return_value = Mock(
        returncode=1,
        stdout="",
        stderr="./main.go:10:2: unreachable code\n./main.go:15:5: error: undefined variable\n",
    )

    analyzer = StaticAnalyzer(go_directory)
    analyzer.available_tools = ["go-vet"]

    result = analyzer.run_tool("go-vet")

    assert result.tool == "go-vet"
    assert result.passed is False
    assert result.issues_count >= 1  # At least one error


@patch("subprocess.run")
def test_run_gofmt_with_unformatted_files(mock_subprocess, go_directory):
    """Test running gofmt when it finds unformatted files."""
    mock_subprocess.return_value = Mock(
        returncode=0,  # gofmt returns 0 even with unformatted files
        stdout="main.go\nutils.go\n",
        stderr="",
    )

    analyzer = StaticAnalyzer(go_directory)
    analyzer.available_tools = ["gofmt"]

    result = analyzer.run_tool("gofmt")

    assert result.tool == "gofmt"
    assert result.passed is False  # Has unformatted files
    assert result.issues_count == 2  # Two unformatted files


@patch("subprocess.run")
def test_run_gofmt_all_formatted(mock_subprocess, go_directory):
    """Test running gofmt when all files are formatted."""
    mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

    analyzer = StaticAnalyzer(go_directory)
    analyzer.available_tools = ["gofmt"]

    result = analyzer.run_tool("gofmt")

    assert result.tool == "gofmt"
    assert result.passed is True
    assert result.issues_count == 0


def test_run_go_tools_no_go_files(sample_directory):
    """Test that Go tools return 'No Go files found' for non-Go directories."""
    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["go-vet", "golangci-lint", "gofmt"]

    for tool in ["go-vet", "golangci-lint", "gofmt"]:
        result = analyzer.run_tool(tool)
        assert result.passed is True
        assert result.output == "No Go files found"


def test_tools_have_language_field():
    """Test that all tools have a language field."""
    valid_languages = [
        "python",
        "go",
        "shell",
        "cpp",
        "java",
        "javascript",
        "typescript",
    ]
    for tool_name, config in StaticAnalyzer.TOOLS.items():
        assert "language" in config, f"{tool_name} missing language field"
        assert config["language"] in valid_languages, (
            f"{tool_name} has invalid language"
        )


# Shell script static analysis tests


def test_shellcheck_in_tools():
    """Test that shellcheck is included in TOOLS."""
    assert "shellcheck" in StaticAnalyzer.TOOLS
    shellcheck_config = StaticAnalyzer.TOOLS["shellcheck"]
    assert shellcheck_config["name"] == "ShellCheck"
    assert shellcheck_config["description"] == "Shell script static analyzer"
    assert shellcheck_config["command"] == ["shellcheck"]
    assert shellcheck_config["language"] == "shell"


@patch("subprocess.run")
def test_run_shellcheck_with_issues(mock_subprocess, sample_directory):
    """Test running shellcheck when it finds issues."""
    # Create a shell script file for testing
    shell_file = sample_directory / "test.sh"
    shell_file.write_text("#!/bin/bash\necho $VAR\n")

    mock_subprocess.return_value = Mock(
        returncode=1,
        stdout="",
        stderr="In test.sh line 2:\necho $VAR\n     ^-- SC2086: Double quote to prevent globbing and word splitting.\n",
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["shellcheck"]

    result = analyzer.run_tool("shellcheck")

    assert result.tool == "shellcheck"
    assert result.passed is False
    assert result.issues_count >= 1  # SC code found


@patch("subprocess.run")
def test_run_shellcheck_all_clean(mock_subprocess, sample_directory):
    """Test running shellcheck when all scripts pass."""
    # Create a shell script file for testing
    shell_file = sample_directory / "test.sh"
    shell_file.write_text('#!/bin/bash\necho "$VAR"\n')

    mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["shellcheck"]

    result = analyzer.run_tool("shellcheck")

    assert result.tool == "shellcheck"
    assert result.passed is True
    assert result.issues_count == 0


def test_run_shellcheck_no_scripts(sample_directory):
    """Test running shellcheck when no shell scripts exist."""
    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["shellcheck"]

    result = analyzer.run_tool("shellcheck")

    assert result.tool == "shellcheck"
    assert result.passed is True
    assert result.issues_count == 0
    assert "No shell scripts found" in result.output


# Path validation tests


def test_validate_file_path_within_directory(sample_directory):
    """Test that files within directory are validated as safe."""
    analyzer = StaticAnalyzer(sample_directory)

    # Create a test file
    test_file = sample_directory / "test.py"
    test_file.write_text("# test")

    assert analyzer._validate_file_path(test_file) is True


def test_validate_file_path_outside_directory(sample_directory, tmp_path):
    """Test that files outside directory are rejected."""
    analyzer = StaticAnalyzer(sample_directory)

    # Create a file outside the analysis directory
    outside_file = tmp_path / "outside" / "test.py"
    outside_file.parent.mkdir(parents=True, exist_ok=True)
    outside_file.write_text("# test")

    assert analyzer._validate_file_path(outside_file) is False


def test_filter_safe_files(sample_directory, tmp_path):
    """Test filtering file list to only include safe files."""
    analyzer = StaticAnalyzer(sample_directory)

    # Create files inside and outside directory
    inside_file = sample_directory / "inside.py"
    inside_file.write_text("# inside")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "outside.py"
    outside_file.write_text("# outside")

    files = [inside_file, outside_file]
    safe_files = analyzer._filter_safe_files(files)

    assert len(safe_files) == 1
    assert inside_file in safe_files
    assert outside_file not in safe_files


def test_validate_directory_exists(tmp_path):
    """Construction must fail fast for a non-existent directory."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="does not exist"):
        StaticAnalyzer(non_existent)


def test_validate_directory_is_file(tmp_path):
    """Construction must fail fast when the path is a file, not a directory."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")
    with pytest.raises(ValueError, match="not a directory"):
        StaticAnalyzer(file_path)


# ---------------------------------------------------------------------------
# Reproducibility: when MAX_FILES_PER_TOOL truncation triggers, the analyzed
# subset must be deterministic across runs. Filesystem rglob order is not
# guaranteed, so we sort before slicing. This test pins that contract; if
# anyone replaces `sorted(files)[:N]` with `files[:N]` it should fail.
# ---------------------------------------------------------------------------


def test_tool_timeout_default(tmp_path):
    """Default per-tool timeout is the documented constant."""
    analyzer = StaticAnalyzer(tmp_path)
    assert analyzer.tool_timeout == StaticAnalyzer.DEFAULT_TOOL_TIMEOUT_SECONDS


def test_tool_timeout_override(tmp_path):
    """Caller-provided tool_timeout overrides the default."""
    analyzer = StaticAnalyzer(tmp_path, tool_timeout=600)
    assert analyzer.tool_timeout == 600


def test_tool_timeout_rejects_non_positive(tmp_path):
    """tool_timeout must be > 0; zero or negative raises at construction."""
    with pytest.raises(ValueError, match="positive"):
        StaticAnalyzer(tmp_path, tool_timeout=0)
    with pytest.raises(ValueError, match="positive"):
        StaticAnalyzer(tmp_path, tool_timeout=-30)


@patch("subprocess.run")
def test_tool_timeout_passed_to_subprocess(mock_subprocess, sample_directory):
    """The configured timeout is forwarded to subprocess.run, not hardcoded."""
    mock_subprocess.return_value = Mock(returncode=0, stdout="ok", stderr="")

    analyzer = StaticAnalyzer(sample_directory, tool_timeout=900)
    analyzer.available_tools = ["ruff"]
    analyzer._tool_paths = {"ruff": "/usr/bin/ruff"}

    analyzer.run_tool("ruff")

    # subprocess.run is called several times during init/check + once per
    # tool. The last call is the actual tool execution; verify its timeout.
    last_call_kwargs = mock_subprocess.call_args.kwargs
    assert last_call_kwargs["timeout"] == 900


@patch("subprocess.run")
def test_truncation_is_deterministic(mock_subprocess, tmp_path):
    """File-list truncation must select the same subset regardless of walk order."""
    from codereview.static_analysis import MAX_FILES_PER_TOOL

    mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

    # Two analyzers, identical filesystem content, mocked rglob returning
    # the same paths in *different* orders. Truncation should still pick
    # the same 500 files.
    files = [tmp_path / f"script_{i:04d}.sh" for i in range(MAX_FILES_PER_TOOL + 50)]
    for f in files:
        f.write_text("#!/bin/sh\necho hi\n")

    analyzer = StaticAnalyzer(tmp_path)
    analyzer.available_tools = ["shellcheck"]
    analyzer._tool_paths = {"shellcheck": "/usr/bin/shellcheck"}

    forward = list(files)
    reverse = list(reversed(files))

    # Run with rglob walk in reverse order
    with patch.object(
        StaticAnalyzer, "_safe_rglob", side_effect=lambda pattern: reverse
    ):
        analyzer.run_tool("shellcheck")
    cmd_reverse = mock_subprocess.call_args[0][0]

    # Run with rglob walk in forward order
    with patch.object(
        StaticAnalyzer, "_safe_rglob", side_effect=lambda pattern: forward
    ):
        analyzer.run_tool("shellcheck")
    cmd_forward = mock_subprocess.call_args[0][0]

    # Both runs must analyze the same files (the lexicographically-first
    # MAX_FILES_PER_TOOL of them — in this case, script_0000..script_0499).
    # _safe_rglob was patched twice (once for *.sh, once for *.bash) so the
    # combined list is the doubled file count; we just need order parity.
    assert cmd_reverse[1:] == cmd_forward[1:], (
        "Truncation produced different file lists for forward vs reverse "
        "filesystem walk order — sort-before-truncate guarantee broken."
    )

    # And the chosen files must be the lexicographic prefix, not a tail
    # or arbitrary slice.
    files_in_cmd = [Path(arg).name for arg in cmd_forward[1:]]
    expected_prefix = sorted([f.name for f in files] * 2)[:MAX_FILES_PER_TOOL]
    assert files_in_cmd == expected_prefix


def test_condense_for_prompt_skips_passing_tools():
    """Tools with passed=True and 0 issues produce no block."""
    results = {
        "ruff": StaticAnalysisResult(
            tool="ruff", passed=True, issues_count=0, output="", errors=[]
        ),
    }
    assert StaticAnalyzer.condense_for_prompt(results) == ""


def test_condense_for_prompt_includes_failing_tool_output():
    """A failing tool with output gets a labeled section."""
    results = {
        "ruff": StaticAnalysisResult(
            tool="ruff",
            passed=False,
            issues_count=2,
            output="app.py:1:1 E501 line too long\napp.py:5:1 F401 unused import",
            errors=[],
        )
    }
    block = StaticAnalyzer.condense_for_prompt(results)
    assert "ruff" in block and "2 issue" in block
    assert "E501" in block and "F401" in block


def test_condense_for_prompt_truncates_long_output():
    """Per-tool line cap kicks in and adds an elision marker."""
    output = "\n".join(f"app.py:{i}:1 E001 issue" for i in range(100))
    results = {
        "ruff": StaticAnalysisResult(
            tool="ruff", passed=False, issues_count=100, output=output, errors=[]
        )
    }
    block = StaticAnalyzer.condense_for_prompt(results, max_lines_per_tool=10)
    assert "more line(s) elided" in block


def test_condense_for_prompt_caps_total_chars():
    """Total output is capped at max_chars."""
    huge_output = "x" * 50_000
    results = {
        "ruff": StaticAnalysisResult(
            tool="ruff", passed=False, issues_count=1, output=huge_output, errors=[]
        )
    }
    block = StaticAnalyzer.condense_for_prompt(
        results, max_chars=500, max_lines_per_tool=99
    )
    assert len(block) <= 500 + len("\n... (linter output truncated)")
    assert "linter output truncated" in block


def test_condense_for_prompt_filters_to_batch_paths():
    """only_paths slices linter output to lines mentioning the batch's files."""
    output = (
        "src/foo.py:1:1 E501 line too long\n"
        "src/foo.py:5:1 F401 unused import\n"
        "src/bar.py:2:1 E302 expected 2 blank lines\n"
        "src/baz.go:10:1 unused variable\n"
    )
    results = {
        "ruff": StaticAnalysisResult(
            tool="ruff",
            passed=False,
            issues_count=4,
            output=output,
            errors=[],
        )
    }
    block = StaticAnalyzer.condense_for_prompt(
        results, only_paths=["src/foo.py", "src/bar.py"]
    )
    assert "foo.py" in block and "bar.py" in block
    assert "baz.go" not in block


def test_condense_for_prompt_filter_empty_for_unrelated_batch():
    """A batch with no files matching any tool output yields empty block."""
    output = "src/foo.py:1:1 E501 line too long"
    results = {
        "ruff": StaticAnalysisResult(
            tool="ruff",
            passed=False,
            issues_count=1,
            output=output,
            errors=[],
        )
    }
    block = StaticAnalyzer.condense_for_prompt(
        results, only_paths=["src/main.go", "src/util.go"]
    )
    assert block == ""


def test_condense_for_prompt_filter_matches_across_path_forms():
    """An absolute linter path still matches a relative only_paths entry when
    they share the parent/basename suffix (tolerant of abs-vs-rel forms)."""
    output = "/abs/path/to/src/foo.py:1:1 E501 line too long"
    results = {
        "ruff": StaticAnalysisResult(
            tool="ruff",
            passed=False,
            issues_count=1,
            output=output,
            errors=[],
        )
    }
    block = StaticAnalyzer.condense_for_prompt(results, only_paths=["src/foo.py"])
    assert "foo.py" in block and "E501" in block


def test_condense_for_prompt_filter_rejects_basename_collision():
    """Regression: a same-named file in a different directory must NOT match.

    Basename-only matching pulled tests/fixtures/config.py into a batch that
    only contained src/config.py. Matching on parent/basename keeps them
    distinct.
    """
    output = (
        "src/config.py:1:1 E501 line too long\n"
        "tests/fixtures/config.py:9:1 F401 unused import\n"
    )
    results = {
        "ruff": StaticAnalysisResult(
            tool="ruff", passed=False, issues_count=2, output=output, errors=[]
        )
    }
    block = StaticAnalyzer.condense_for_prompt(results, only_paths=["src/config.py"])
    assert "src/config.py" in block
    assert "fixtures/config.py" not in block


def test_filter_lines_for_paths_drops_summary_banners():
    """Lines without any allowed path token are dropped (e.g. 'Found 3 errors')."""
    output = "src/foo.py:1: error\nFound 1 error in 1 file\nSome unrelated line\n"
    kept, dropped = StaticAnalyzer._filter_lines_for_paths(output, {"src/foo.py"})
    assert kept == ["src/foo.py:1: error"]
    assert dropped == 2


def test_npm_audit_count_uses_severity_buckets_not_total():
    """metadata.vulnerabilities sums per-severity buckets, excluding `total`.

    Including npm's `total` (itself the sum of the buckets) would double-count.
    """
    import json

    output = json.dumps(
        {
            "metadata": {
                "vulnerabilities": {
                    "info": 0,
                    "low": 2,
                    "moderate": 1,
                    "high": 3,
                    "critical": 0,
                    "total": 6,
                }
            }
        }
    )
    assert StaticAnalyzer._count_npm_audit_issues(output) == 6


def test_npm_audit_count_returns_zero_on_non_json():
    """A non-JSON body (e.g. HTML error page from a proxy) must report 0, not a
    fabricated per-line count that would inflate the issue total."""
    html = "<html><body>\n" + "<p>proxy error</p>\n" * 200 + "</body></html>\n"
    assert StaticAnalyzer._count_npm_audit_issues(html) == 0


def test_safe_rglob_caches_repeated_patterns(tmp_path):
    """Repeated _safe_rglob calls for the same pattern walk the tree once.

    Several tools target one language (e.g. four Go tools all glob "*.go"); the
    per-instance cache must collapse those to a single rglob walk per run.
    """
    (tmp_path / "a.go").write_text("package main\n")
    (tmp_path / "b.go").write_text("package main\n")
    (tmp_path / "c.py").write_text("x = 1\n")
    analyzer = StaticAnalyzer(tmp_path)

    real_rglob = Path.rglob
    calls = {"n": 0}

    def counting_rglob(self, pattern):
        calls["n"] += 1
        return real_rglob(self, pattern)

    with patch.object(Path, "rglob", counting_rglob):
        first = analyzer._safe_rglob("*.go")
        second = analyzer._safe_rglob("*.go")
        third = analyzer._safe_rglob("*.go")
        py = analyzer._safe_rglob("*.py")

    assert first == second == third
    assert len(first) == 2
    assert len(py) == 1
    # "*.go" walked once (then cached) + "*.py" walked once == 2 total.
    assert calls["n"] == 2


def test_safe_rglob_suffixes_cache_key_is_order_independent(tmp_path):
    """Suffix-set cache is keyed order-independently and shares one walk."""
    (tmp_path / "a.go").write_text("package main\n")
    (tmp_path / "c.py").write_text("x = 1\n")
    analyzer = StaticAnalyzer(tmp_path)

    real_rglob = Path.rglob
    calls = {"n": 0}

    def counting_rglob(self, pattern):
        calls["n"] += 1
        return real_rglob(self, pattern)

    with patch.object(Path, "rglob", counting_rglob):
        r1 = analyzer._safe_rglob_suffixes({".go", ".py"})
        r2 = analyzer._safe_rglob_suffixes({".py", ".go"})  # same key

    assert r1 == r2
    assert calls["n"] == 1
