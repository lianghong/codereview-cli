# tests/test_models.py
import logging

import pytest
from pydantic import ValidationError

from codereview import models
from codereview.models import CodeReviewReport, ReviewIssue, ReviewMetrics


@pytest.fixture(autouse=True)
def _reset_warn_dedupe():
    """Clear the warn-once caches so each test sees a fresh log state."""
    models._warned_unknown_severities.clear()
    models._warned_unknown_categories.clear()
    yield
    models._warned_unknown_severities.clear()
    models._warned_unknown_categories.clear()


# Required-field placeholders for tests that exercise validators
# (severity normalization, category coercion, references filtering, drift
# counters) and don't otherwise care about the issue's content. Use
# concrete strings rather than the now-forbidden defaults so the placeholder
# validator doesn't reject them.
_MINIMAL_ISSUE_KWARGS = dict(
    file_path="src/example.py",
    line_start=10,
    title="Validator placeholder title",
    description="Validator placeholder description",
    rationale="Validator placeholder rationale",
)


def _minimal_issue(**overrides):
    """Build a ReviewIssue for validator tests, filling in required fields.

    Each override replaces one or more of the minimal placeholder values.
    Used by tests that focus on a single validator (severity/category/
    references) and don't otherwise care about the rest of the fields.
    """
    return ReviewIssue(**{**_MINIMAL_ISSUE_KWARGS, **overrides})


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
        title="SQL injection in query construction",
        description="User-controlled input is interpolated into SQL.",
        rationale="Allows arbitrary database access.",
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
    assert _minimal_issue(severity="SuperCritical").severity == "Medium"

    # Common variations should normalize correctly
    assert _minimal_issue(severity="major").severity == "High"
    assert _minimal_issue(severity="warning").severity == "Medium"
    assert _minimal_issue(severity="minor").severity == "Low"

    # Missing severity should default to Medium
    assert _minimal_issue().severity == "Medium"


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


def test_unknown_severity_logs_warning(caplog):
    """Unknown severity must coerce to Medium AND emit a warning."""
    with caplog.at_level(logging.WARNING, logger="codereview.models"):
        issue = _minimal_issue(severity="SuperCritical")
    assert issue.severity == "Medium"
    assert any(
        "SuperCritical" in rec.message and "severity" in rec.message
        for rec in caplog.records
    )


def test_unknown_category_logs_warning(caplog):
    """Unknown category must coerce to Code Quality AND emit a warning."""
    with caplog.at_level(logging.WARNING, logger="codereview.models"):
        issue = _minimal_issue(category="QuantumEntanglement")
    assert issue.category == "Code Quality"
    assert any(
        "QuantumEntanglement" in rec.message and "category" in rec.message
        for rec in caplog.records
    )


def test_known_severity_mapping_does_not_log(caplog):
    """Mapped variations (e.g. 'major' -> 'High') must NOT emit a warning."""
    with caplog.at_level(logging.WARNING, logger="codereview.models"):
        issue = _minimal_issue(severity="major")
    assert issue.severity == "High"
    assert not caplog.records


def test_valid_severity_does_not_log(caplog):
    """Already-valid values must NOT emit a warning."""
    with caplog.at_level(logging.WARNING, logger="codereview.models"):
        issue = _minimal_issue(severity="Critical", category="Security")
    assert issue.severity == "Critical"
    assert issue.category == "Security"
    assert not caplog.records


def test_unknown_severity_warning_is_deduped(caplog):
    """Same unknown value must only warn once even across many issues."""
    with caplog.at_level(logging.WARNING, logger="codereview.models"):
        for _ in range(5):
            _minimal_issue(severity="BogusLevel")
    matching = [r for r in caplog.records if "BogusLevel" in r.message]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# References URL filtering + drift counters
# ---------------------------------------------------------------------------


def test_references_keeps_authoritative_urls():
    """CWE/OWASP/MDN/language-doc URLs survive the filter."""
    issue = _minimal_issue(
        references=[
            "https://cwe.mitre.org/data/definitions/89.html",
            "https://owasp.org/www-project-top-ten/",
            "https://developer.mozilla.org/en-US/docs/Web/API",
            "https://docs.python.org/3/library/sqlite3.html",
            "https://go.dev/doc/effective_go",
        ]
    )
    assert len(issue.references) == 5


def test_references_drops_non_authoritative_urls():
    """Search results, blogspam, SO links, ftp:// — all dropped."""
    models.reset_drift_counters()
    issue = _minimal_issue(
        references=[
            "https://stackoverflow.com/questions/12345",
            "https://medium.com/some-blog-post",
            "https://random-blog.example.com/",
            "ftp://example.com/file",
            "not-a-url",
            "https://cwe.mitre.org/data/definitions/79.html",  # this one survives
        ]
    )
    assert issue.references == ["https://cwe.mitre.org/data/definitions/79.html"]
    assert models.get_drift_counters()["reference_dropped"] == 5


def test_references_handles_subdomain_authoritative_hosts():
    """Subdomains of authoritative hosts (e.g. peps.python.org) are accepted."""
    issue = _minimal_issue(
        references=[
            "https://peps.python.org/pep-0008/",
            "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
        ]
    )
    assert len(issue.references) == 2


def test_drift_counter_tracks_severity_coercion():
    """Coerced severities bump severity_coerced; valid ones don't."""
    models.reset_drift_counters()
    _minimal_issue(severity="bogus_level_1")
    _minimal_issue(severity="another_bogus")
    _minimal_issue(severity="Critical")
    assert models.get_drift_counters()["severity_coerced"] == 2


def test_drift_counter_tracks_category_coercion():
    """Coerced categories bump category_coerced; valid ones don't."""
    models.reset_drift_counters()
    _minimal_issue(category="QuantumEntanglement")
    _minimal_issue(category="HelloWorld")
    _minimal_issue(category="Security")
    assert models.get_drift_counters()["category_coerced"] == 2


def test_drift_counter_reset_zeroes_all():
    """reset_drift_counters() clears everything."""
    _minimal_issue(
        severity="bogus_a", category="bogus_b", references=["http://blog.example.com"]
    )
    counters = models.get_drift_counters()
    assert sum(counters.values()) > 0
    models.reset_drift_counters()
    assert all(v == 0 for v in models.get_drift_counters().values())


# ---------------------------------------------------------------------------
# Required fields + placeholder rejection
# ---------------------------------------------------------------------------


def test_required_fields_must_be_provided():
    """Omitting any required field raises ValidationError, not silent default."""
    # Missing file_path
    with pytest.raises(ValidationError):
        ReviewIssue(
            line_start=1,
            title="Some title",
            description="Some description",
            rationale="Some rationale",
        )
    # Missing line_start
    with pytest.raises(ValidationError):
        ReviewIssue(
            file_path="x.py",
            title="Some title",
            description="Some description",
            rationale="Some rationale",
        )
    # Missing title
    with pytest.raises(ValidationError):
        ReviewIssue(
            file_path="x.py",
            line_start=1,
            description="Some description",
            rationale="Some rationale",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", "Issue"),
        ("title", "issue"),  # case-insensitive
        ("title", " Issue "),  # whitespace tolerated
        ("title", "Problem found"),
        ("title", "Bug"),
        ("description", "No description provided"),
        ("description", "n/a"),
        ("description", "None"),
        ("rationale", "Review recommended"),
        ("rationale", "n/a"),
        ("file_path", "unknown"),
    ],
)
def test_placeholder_strings_are_rejected(field, value):
    """The exact placeholders the prompt forbids must raise ValidationError.

    Without this, a model that emits 'Issue' / 'No description provided' as
    a conscious value (rather than relying on the old schema defaults)
    would still produce a fabricated ReviewIssue. Reject at the schema
    boundary so the retry loop sees the failure and asks for real content.
    """
    overrides = {field: value}
    with pytest.raises(ValidationError, match="placeholder"):
        _minimal_issue(**overrides)


def test_legitimate_text_containing_forbidden_word_is_accepted():
    """'Issue' is forbidden as the whole title; 'Issue with X' is fine.

    The validator matches on the trimmed lowercase value being EXACTLY one
    of the forbidden strings — substring matches must not trip it.
    """
    issue = _minimal_issue(title="Issue with input validation in handler")
    assert "input validation" in issue.title
