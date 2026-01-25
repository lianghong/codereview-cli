# tests/test_models.py
import pytest
from pydantic import ValidationError

from codereview.models import CodeReviewReport, ReviewIssue, ReviewMetrics


def test_review_issue_creation():
    """Test ReviewIssue model with all fields."""
    issue = ReviewIssue(
        category="Security",
        severity="Critical",
        file_path="app/main.py",
        line_start=42,
        line_end=45,
        title="SQL Injection Risk",
        description="User input not sanitized",
        suggested_code="Use parameterized queries",
        rationale="Prevents SQL injection attacks",
        references=["https://owasp.org/sql-injection"],
    )

    assert issue.category == "Security"
    assert issue.severity == "Critical"
    assert issue.line_start == 42
    assert len(issue.references) == 1


def test_review_issue_minimal():
    """Test ReviewIssue with minimal required fields."""
    issue = ReviewIssue(
        category="Code Quality",
        severity="Low",
        file_path="app/utils.py",
        line_start=10,
        title="Complex function",
        description="Function has high cyclomatic complexity",
        rationale="Hard to maintain",
    )

    assert issue.line_end is None
    assert issue.suggested_code is None
    assert issue.references == []


def test_code_review_report():
    """Test CodeReviewReport model."""
    issue = ReviewIssue(
        category="Security",
        severity="High",
        file_path="app.py",
        line_start=1,
        title="Issue",
        description="Description",
        rationale="Rationale",
    )

    report = CodeReviewReport(
        summary="Found 1 security issue",
        metrics=ReviewMetrics(files_analyzed=5, total_lines=100, total_issues=1),
        issues=[issue],
        system_design_insights="Architecture looks good",
        recommendations=["Fix security issue"],
    )

    assert report.summary == "Found 1 security issue"
    assert len(report.issues) == 1
    assert report.metrics.files_analyzed == 5


def test_category_normalization():
    """Test that unknown categories are normalized to valid values."""
    # Unknown category defaults to "Code Quality"
    issue = ReviewIssue(
        category="InvalidCategory",
        severity="High",
        file_path="app.py",
        line_start=1,
        title="Test",
        description="Test",
        rationale="Test",
    )
    assert issue.category == "Code Quality"

    # Known variations are mapped correctly
    issue2 = ReviewIssue(
        category="error handling",
        severity="High",
        file_path="app.py",
        line_start=1,
        title="Test",
        description="Test",
        rationale="Test",
    )
    assert issue2.category == "Code Quality"

    issue3 = ReviewIssue(
        category="architecture",
        severity="High",
        file_path="app.py",
        line_start=1,
        title="Test",
        description="Test",
        rationale="Test",
    )
    assert issue3.category == "System Design"


def test_severity_normalization():
    """Test that invalid/unknown severity is normalized to Medium."""
    # Unknown severity should normalize to Medium
    issue1 = ReviewIssue(
        category="Security",
        severity="SuperCritical",
        file_path="app.py",
        line_start=1,
        title="Test",
        description="Test",
        rationale="Test",
    )
    assert issue1.severity == "Medium"

    # Common variations should normalize correctly
    issue2 = ReviewIssue(severity="major")
    assert issue2.severity == "High"

    issue3 = ReviewIssue(severity="warning")
    assert issue3.severity == "Medium"

    issue4 = ReviewIssue(severity="minor")
    assert issue4.severity == "Low"

    # Missing severity should default to Medium
    issue5 = ReviewIssue()
    assert issue5.severity == "Medium"


def test_invalid_line_start():
    """Test that line_start < 1 raises ValidationError."""
    with pytest.raises(ValidationError):
        ReviewIssue(
            category="Security",
            severity="High",
            file_path="app.py",
            line_start=0,
            title="Test",
            description="Test",
            rationale="Test",
        )


def test_line_end_before_line_start():
    """Test that line_end < line_start raises ValidationError."""
    with pytest.raises(ValidationError):
        ReviewIssue(
            category="Security",
            severity="High",
            file_path="app.py",
            line_start=50,
            line_end=40,
            title="Test",
            description="Test",
            rationale="Test",
        )
