import pytest
from pathlib import Path
from codereview.renderer import MarkdownExporter
from codereview.models import CodeReviewReport, ReviewIssue


@pytest.fixture
def sample_report():
    """Create sample report."""
    issue = ReviewIssue(
        category="Security",
        severity="Critical",
        file_path="app.py",
        line_start=42,
        line_end=45,
        title="SQL Injection",
        description="User input not sanitized",
        suggested_code="cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        rationale="Prevents SQL injection",
        references=["https://owasp.org/sql-injection"]
    )

    return CodeReviewReport(
        summary="Found 1 critical security issue",
        metrics={"files": 1, "issues": 1, "critical": 1},
        issues=[issue],
        system_design_insights="Single file reviewed",
        recommendations=["Fix SQL injection immediately"]
    )


def test_markdown_exporter_initialization():
    """Test exporter can be initialized."""
    exporter = MarkdownExporter()
    assert exporter is not None


def test_export_to_file(sample_report, tmp_path):
    """Test exporting report to Markdown file."""
    output_file = tmp_path / "report.md"

    exporter = MarkdownExporter()
    exporter.export(sample_report, output_file)

    assert output_file.exists()
    content = output_file.read_text()

    assert "# Code Review Report" in content
    assert "SQL Injection" in content
    assert "Critical" in content


def test_markdown_contains_all_sections(sample_report, tmp_path):
    """Test Markdown contains all expected sections."""
    output_file = tmp_path / "report.md"

    exporter = MarkdownExporter()
    exporter.export(sample_report, output_file)

    content = output_file.read_text()

    assert "## Executive Summary" in content
    assert "## Metrics" in content
    assert "## Issues by Severity" in content
    assert "### 🔴 Critical" in content
    assert "## System Design Insights" in content
    assert "## Top Recommendations" in content


def test_markdown_includes_code_blocks(sample_report, tmp_path):
    """Test Markdown includes code in proper blocks."""
    output_file = tmp_path / "report.md"

    exporter = MarkdownExporter()
    exporter.export(sample_report, output_file)

    content = output_file.read_text()

    assert "```python" in content or "```" in content
    assert "cursor.execute" in content
