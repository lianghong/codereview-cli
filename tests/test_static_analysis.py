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
    """Test validation fails for non-existent directory."""
    non_existent = tmp_path / "does_not_exist"
    analyzer = StaticAnalyzer(non_existent)

    is_valid, error = analyzer._validate_directory()
    assert is_valid is False
    assert "does not exist" in error


def test_validate_directory_is_file(tmp_path):
    """Test validation fails when path is a file, not directory."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")

    analyzer = StaticAnalyzer(file_path)

    is_valid, error = analyzer._validate_directory()
    assert is_valid is False
    assert "not a directory" in error
