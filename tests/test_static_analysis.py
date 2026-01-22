"""Tests for static analysis integration."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from codereview.static_analysis import StaticAnalyzer, StaticAnalysisResult


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
        tool="ruff",
        passed=True,
        issues_count=0,
        output="All checks passed",
        errors=[]
    )

    assert result.tool == "ruff"
    assert result.passed is True
    assert result.issues_count == 0
    assert result.output == "All checks passed"
    assert len(result.errors) == 0


def test_static_analysis_result_with_errors():
    """Test StaticAnalysisResult with errors."""
    result = StaticAnalysisResult(
        tool="mypy",
        passed=False,
        issues_count=5,
        output="",
        errors=["Tool not found"]
    )

    assert result.passed is False
    assert result.issues_count == 5
    assert len(result.errors) == 1


@patch('subprocess.run')
def test_run_tool_success(mock_subprocess, sample_directory):
    """Test running a tool successfully."""
    # Mock successful tool execution
    mock_subprocess.return_value = Mock(
        returncode=0,
        stdout="All checks passed",
        stderr=""
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["ruff"]  # Force tool to be available

    result = analyzer.run_tool("ruff")

    assert result.tool == "ruff"
    assert result.passed is True


@patch('subprocess.run')
def test_run_tool_with_issues(mock_subprocess, sample_directory):
    """Test running a tool that finds issues."""
    # Mock tool finding issues
    mock_subprocess.return_value = Mock(
        returncode=1,
        stdout="error: line too long\nwarning: unused import\n",
        stderr=""
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


@patch('subprocess.run')
def test_run_all_tools(mock_subprocess, sample_directory):
    """Test running all available tools."""
    # Mock all tools passing
    mock_subprocess.return_value = Mock(
        returncode=0,
        stdout="All checks passed",
        stderr=""
    )

    analyzer = StaticAnalyzer(sample_directory)
    analyzer.available_tools = ["ruff", "mypy"]

    results = analyzer.run_all()

    assert len(results) == 2
    assert "ruff" in results
    assert "mypy" in results
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
