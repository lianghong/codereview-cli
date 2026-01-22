import pytest
from io import StringIO
from codereview.renderer import TerminalRenderer
from codereview.models import CodeReviewReport, ReviewIssue


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
        metrics={"files": 2, "issues": 2, "critical": 1, "low": 1},
        issues=[issue1, issue2],
        system_design_insights="Architecture is solid",
        recommendations=["Fix SQL injection", "Refactor complex function"]
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
