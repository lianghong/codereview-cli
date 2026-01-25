from io import StringIO

import pytest
from rich.console import Console

from codereview.models import CodeReviewReport, ReviewIssue, ReviewMetrics
from codereview.providers.base import ValidationResult
from codereview.renderer import (
    MarkdownExporter,
    StaticAnalysisRenderer,
    TerminalRenderer,
    ValidationRenderer,
)
from codereview.static_analysis import StaticAnalysisResult


@pytest.fixture
def sample_report():
    """Create sample report for testing."""
    issue1 = ReviewIssue(
        category="Security",
        severity="Critical",
        file_path="app.py",
        line_start=42,
        title="SQL Injection",
        description="User input not sanitized",
        rationale="Security risk",
    )

    issue2 = ReviewIssue(
        category="Code Quality",
        severity="Low",
        file_path="utils.py",
        line_start=10,
        title="Complex function",
        description="High complexity",
        rationale="Maintainability",
    )

    return CodeReviewReport(
        summary="Found 2 issues",
        metrics=ReviewMetrics(
            files_analyzed=2, total_issues=2, critical_issues=1, low_issues=1
        ),
        issues=[issue1, issue2],
        system_design_insights="Architecture is solid",
        recommendations=["Fix SQL injection", "Refactor complex function"],
    )


def test_renderer_initialization():
    """Test renderer can be initialized."""
    renderer = TerminalRenderer()
    assert renderer is not None


def test_render_summary(sample_report):
    """Test rendering summary section."""
    renderer = TerminalRenderer()
    output = renderer._format_summary(sample_report)

    assert "Found 2 issues" in output
    assert "files" in output.lower()


def test_severity_color_mapping():
    """Test severity to color mapping."""
    renderer = TerminalRenderer()

    assert renderer._get_severity_color("Critical") == "red"
    assert renderer._get_severity_color("High") == "bright_red"
    assert renderer._get_severity_color("Medium") == "yellow"
    assert renderer._get_severity_color("Low") == "blue"
    assert renderer._get_severity_color("Info") == "white"


def test_group_issues_by_severity(sample_report):
    """Test grouping issues by severity."""
    renderer = TerminalRenderer()
    grouped = renderer._group_by_severity(sample_report.issues)

    assert "Critical" in grouped
    assert "Low" in grouped
    assert len(grouped["Critical"]) == 1
    assert len(grouped["Low"]) == 1


class TestStaticAnalysisRenderer:
    """Tests for StaticAnalysisRenderer."""

    def test_initialization(self):
        """Test renderer can be initialized."""
        renderer = StaticAnalysisRenderer()
        assert renderer is not None
        assert renderer.console is not None

    def test_initialization_with_custom_console(self):
        """Test renderer accepts custom console."""
        custom_console = Console(file=StringIO())
        renderer = StaticAnalysisRenderer(console=custom_console)
        assert renderer.console is custom_console

    def test_render_all_passed(self):
        """Test rendering when all tools pass."""
        console = Console(file=StringIO(), force_terminal=True)
        renderer = StaticAnalysisRenderer(console=console)

        results = {
            "ruff": StaticAnalysisResult(
                tool="ruff", passed=True, issues_count=0, output="", errors=[]
            ),
            "mypy": StaticAnalysisResult(
                tool="mypy", passed=True, issues_count=0, output="", errors=[]
            ),
        }

        # Should not raise
        renderer.render(results)

    def test_render_with_failures(self):
        """Test rendering when some tools fail."""
        console = Console(file=StringIO(), force_terminal=True)
        renderer = StaticAnalysisRenderer(console=console)

        results = {
            "ruff": StaticAnalysisResult(
                tool="ruff",
                passed=False,
                issues_count=5,
                output="error: line too long\nerror: unused import",
                errors=[],
            ),
            "mypy": StaticAnalysisResult(
                tool="mypy", passed=True, issues_count=0, output="", errors=[]
            ),
        }

        # Should not raise
        renderer.render(results)

    def test_render_shows_all_outputs(self):
        """Test that rendering shows output for all tools, not just failed ones."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        renderer = StaticAnalysisRenderer(console=console)

        results = {
            "ruff": StaticAnalysisResult(
                tool="ruff",
                passed=True,
                issues_count=0,
                output="All checks passed!",
                errors=[],
            ),
            "mypy": StaticAnalysisResult(
                tool="mypy",
                passed=False,
                issues_count=3,
                output="Found 3 errors in 2 files",
                errors=[],
            ),
        }

        renderer.render(results)
        rendered = output.getvalue()

        # Both tool outputs should be shown
        assert "Ruff" in rendered or "ruff" in rendered
        assert "Mypy" in rendered or "mypy" in rendered


class TestValidationRenderer:
    """Tests for ValidationRenderer."""

    def test_initialization(self):
        """Test renderer can be initialized."""
        renderer = ValidationRenderer()
        assert renderer is not None
        assert renderer.console is not None

    def test_initialization_with_custom_console(self):
        """Test renderer accepts custom console."""
        custom_console = Console(file=StringIO())
        renderer = ValidationRenderer(console=custom_console)
        assert renderer.console is custom_console

    def test_render_valid_result(self):
        """Test rendering valid validation result."""
        console = Console(file=StringIO(), force_terminal=True)
        renderer = ValidationRenderer(console=console)

        validation = ValidationResult(valid=True, provider="Test Provider")
        validation.add_check("Check 1", True, "All good")
        validation.add_check("Check 2", True, "Passed")

        # Should not raise
        renderer.render(validation)

    def test_render_invalid_result_with_suggestions(self):
        """Test rendering invalid result with suggestions."""
        console = Console(file=StringIO(), force_terminal=True)
        renderer = ValidationRenderer(console=console)

        validation = ValidationResult(valid=False, provider="Test Provider")
        validation.add_check("Check 1", False, "Failed check")
        validation.add_warning("This is a warning")
        validation.add_suggestion("Try this fix")

        # Should not raise
        renderer.render(validation)


class TestMarkdownExporter:
    """Tests for MarkdownExporter."""

    def test_export_without_skipped_files(self, sample_report, tmp_path):
        """Test export without skipped files."""
        exporter = MarkdownExporter()
        output_file = tmp_path / "report.md"

        exporter.export(sample_report, output_file)

        content = output_file.read_text()
        assert "# Code Review Report" in content
        assert "Skipped Files" not in content

    def test_export_with_skipped_files(self, sample_report, tmp_path):
        """Test export with skipped files includes the section."""
        exporter = MarkdownExporter()
        output_file = tmp_path / "report.md"

        skipped_files = [
            ("large_file.py", "File too large: 600.0KB > 500KB"),
            ("binary.bin", "Encoding error: not UTF-8"),
        ]

        exporter.export(sample_report, output_file, skipped_files=skipped_files)

        content = output_file.read_text()
        assert "## Skipped Files" in content
        assert "**2 file(s)**" in content
        assert "`large_file.py`" in content
        assert "File too large" in content
        assert "`binary.bin`" in content
        assert "Encoding error" in content

    def test_export_with_many_skipped_files(self, sample_report, tmp_path):
        """Test export truncates skipped files list when > 20."""
        exporter = MarkdownExporter()
        output_file = tmp_path / "report.md"

        # Create 25 skipped files
        skipped_files = [(f"file_{i}.py", f"Reason {i}") for i in range(25)]

        exporter.export(sample_report, output_file, skipped_files=skipped_files)

        content = output_file.read_text()
        assert "## Skipped Files" in content
        assert "**25 file(s)**" in content
        # First 20 should be shown
        assert "`file_0.py`" in content
        assert "`file_19.py`" in content
        # File 20 should not be shown (0-indexed, so file_20 is the 21st)
        assert "`file_20.py`" not in content
        # Should show "and X more"
        assert "... and 5 more file(s)" in content

    def test_skipped_files_method(self):
        """Test _skipped_files generates correct Markdown."""
        exporter = MarkdownExporter()

        skipped_files = [
            ("src/large.py", "File too large: 750KB > 500KB"),
            ("data/config.json", "Not a supported file type"),
        ]

        result = exporter._skipped_files(skipped_files)

        assert "## Skipped Files" in result
        assert "**2 file(s)**" in result
        assert "- `src/large.py`: File too large" in result
        assert "- `data/config.json`: Not a supported file type" in result
