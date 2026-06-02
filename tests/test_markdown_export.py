import pytest

from codereview.models import CodeReviewReport, ReviewIssue, ReviewMetrics
from codereview.renderer import MarkdownExporter


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
        references=["https://owasp.org/sql-injection"],
    )

    return CodeReviewReport(
        summary="Found 1 critical security issue",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=1, critical_issues=1),
        issues=[issue],
        system_design_insights="Single file reviewed",
        recommendations=["Fix SQL injection immediately"],
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


# ---------------------------------------------------------------------------
# Raw-dict / partial metrics resilience
#
# metrics_to_dict() deliberately supports a raw-dict fallback (it returns
# report.metrics unchanged when it isn't a Pydantic model). The token-metrics
# rendering and cost math must survive a raw dict whose token values are
# stringified or None without raising during export — the same guard the
# regular-metrics loop and _format_summary already apply.
# ---------------------------------------------------------------------------


def _report_with_raw_metrics(raw_metrics: dict) -> CodeReviewReport:
    """Build a report whose .metrics is a plain dict (not ReviewMetrics).

    model_construct skips validation so we can plant the raw-dict shape that
    metrics_to_dict()'s fallback branch is written to tolerate.
    """
    return CodeReviewReport.model_construct(
        summary="raw-dict metrics",
        metrics=raw_metrics,
        issues=[],
    )


@pytest.mark.parametrize(
    "raw_metrics",
    [
        # Stringified token counts (e.g. from a hand-built/legacy raw dict).
        {"input_tokens": "100", "output_tokens": "50", "total_tokens": "150"},
        # None token fields alongside pricing.
        {
            "input_tokens": None,
            "output_tokens": None,
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
        },
        # Mixed: one real int, one stringified — cost math must be skipped.
        {
            "input_tokens": 100,
            "output_tokens": "50",
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
        },
    ],
)
def test_markdown_export_survives_raw_dict_token_metrics(raw_metrics, tmp_path):
    """A malformed/stringified token metric must not abort the whole export."""
    report = _report_with_raw_metrics(raw_metrics)
    output_file = tmp_path / "report.md"

    # Should not raise (no ValueError from f"{value:,}", no TypeError from
    # dividing a str/None token count).
    MarkdownExporter().export(report, output_file)

    content = output_file.read_text()
    assert "# Code Review Report" in content
    assert "### Token Usage & Cost" in content


def test_markdown_export_raw_dict_int_tokens_still_compute_cost(tmp_path):
    """Well-formed int tokens in a raw dict still produce a cost line."""
    report = _report_with_raw_metrics(
        {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
        }
    )
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(report, output_file)

    content = output_file.read_text()
    # 1M input @ $3 + 1M output @ $15 = $18.0000
    assert "**Estimated Cost:** $18.0000 USD" in content


# ---------------------------------------------------------------------------
# Audit-trail section
# ---------------------------------------------------------------------------


def test_audit_trail_section_omitted_when_no_audit(sample_report, tmp_path):
    """Without an audit dict, the report has no Audit Trail section."""
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(sample_report, output_file)
    assert "## Audit Trail" not in output_file.read_text()


def test_audit_trail_includes_dedupe_counts(sample_report, tmp_path):
    """Dedupe counts are surfaced when non-zero."""
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(
        sample_report,
        output_file,
        audit={
            "issues_deduplicated": 4,
            "design_insights_deduplicated": 2,
            "linter_tools_injected": 0,
            "drift": {},
            "languages_in_batches": [],
        },
    )
    content = output_file.read_text()
    assert "## Audit Trail" in content
    assert "4 duplicate(s) collapsed" in content
    assert "2 paraphrase(s) collapsed" in content


def test_audit_trail_surfaces_drift(sample_report, tmp_path):
    """Schema-drift counters are listed when any are non-zero."""
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(
        sample_report,
        output_file,
        audit={
            "drift": {
                "severity_coerced": 3,
                "category_coerced": 1,
                "reference_dropped": 7,
            },
        },
    )
    content = output_file.read_text()
    assert "## Audit Trail" in content
    assert "3 severity" in content
    assert "1 category" in content
    assert "7 reference URL(s)" in content


def test_audit_trail_omits_zero_drift(sample_report, tmp_path):
    """An all-zero drift dict produces no drift line."""
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(
        sample_report,
        output_file,
        audit={
            "drift": {
                "severity_coerced": 0,
                "category_coerced": 0,
                "reference_dropped": 0,
            },
        },
    )
    content = output_file.read_text()
    # Header still emitted ...
    assert "## Audit Trail" in content
    # ... but no per-counter line.
    assert "Schema drift" not in content


def test_audit_trail_lists_languages(sample_report, tmp_path):
    """Per-batch language slicing line is rendered when languages provided."""
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(
        sample_report,
        output_file,
        audit={"languages_in_batches": ["go", "python"]},
    )
    content = output_file.read_text()
    assert "## Audit Trail" in content
    assert "Per-batch language slicing" in content
    assert "go, python" in content


def test_audit_trail_handles_empty_audit_gracefully(sample_report, tmp_path):
    """An empty audit dict still produces a stable section shape."""
    output_file = tmp_path / "report.md"
    # Empty dict is falsy → audit section is skipped (matches the if-guard).
    MarkdownExporter().export(sample_report, output_file, audit={})
    assert "## Audit Trail" not in output_file.read_text()


def test_audit_trail_unknown_keys_emits_placeholder(sample_report, tmp_path):
    """An audit dict with truthy-but-unrecognized keys still emits the section."""
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(
        sample_report,
        output_file,
        audit={"unrecognized_key": "ignored"},
    )
    content = output_file.read_text()
    assert "## Audit Trail" in content
    assert "No audit signals reported" in content


def test_audit_trail_zero_linter_tools_emits_negative_line(sample_report, tmp_path):
    """linter_tools_injected=0 says 'none' rather than being silent.

    Telling the reader "linters were not run, so the LLM did not see any
    pre-flagged findings" is itself a useful signal — it differentiates a
    run that simply hadn't gone through static analysis from one that did
    but found nothing relevant.
    """
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(
        sample_report,
        output_file,
        audit={"linter_tools_injected": 0},
    )
    content = output_file.read_text()
    assert "## Audit Trail" in content
    assert "Linter findings injected:" in content
    assert "none" in content
