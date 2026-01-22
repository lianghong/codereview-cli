# tests/test_models.py
import pytest
from codereview.models import ReviewIssue, CodeReviewReport


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
        references=["https://owasp.org/sql-injection"]
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
        rationale="Hard to maintain"
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
        rationale="Rationale"
    )

    report = CodeReviewReport(
        summary="Found 1 security issue",
        metrics={"files": 5, "lines": 100, "issues": 1},
        issues=[issue],
        system_design_insights="Architecture looks good",
        recommendations=["Fix security issue"]
    )

    assert report.summary == "Found 1 security issue"
    assert len(report.issues) == 1
    assert report.metrics["files"] == 5
