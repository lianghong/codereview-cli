import pytest

from codereview.models import CodeReviewReport, ReviewIssue, ReviewMetrics
from codereview.renderer import MarkdownExporter, balance_code_fences


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


# ---------------------------------------------------------------------------
# Code-fence balancing in model-generated prose
#
# A model that opens a ``` fence in a free-text field without closing it would
# otherwise swallow every following section of the report into one code block —
# the exported artifact silently loses its issues, recommendations, and metrics.
# balance_code_fences closes the block instead.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("no fences here", "no fences here"),
        ("```python\nx = 1\n```", "```python\nx = 1\n```"),  # balanced, untouched
        ("```python\nx = 1", "```python\nx = 1\n```"),  # unclosed -> closed
        ("````\n```\n````", "````\n```\n````"),  # nested, balanced
        ("````md\n```py\nx\n```", "````md\n```py\nx\n```\n````"),  # outer unclosed
        ("use ``` inline mid-sentence", "use ``` inline mid-sentence"),  # not a fence
    ],
)
def test_balance_code_fences(text, expected):
    """Only fence *lines* count; unbalanced blocks get a closer appended."""
    assert balance_code_fences(text) == expected


def _fence_runs(markdown: str) -> int:
    """Count fence-opening/closing lines in a rendered document."""
    return sum(1 for line in markdown.splitlines() if line.strip().startswith("```"))


def _inside_code_block(markdown: str, marker: str | None) -> bool:
    """Is *marker* inside a fenced code block per CommonMark's closing rule?

    Fence parity is the wrong model when fence widths vary: a block opened with
    N backticks closes only on a fence line of **at least N** backticks *and*
    with no info string, so a wider inner fence neither closes the block nor
    opens one of its own. A document whose fence lines balance numerically can
    still leave later text inside a block. ``marker=None`` asks whether the
    document *ends* inside one.
    """
    open_width = 0
    for line in markdown.splitlines():
        if marker is not None and marker in line:
            return open_width > 0
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        width = len(stripped) - len(stripped.lstrip("`"))
        info = stripped[width:].strip()
        if open_width == 0:
            open_width = width
        elif width >= open_width and not info:
            open_width = 0
    return open_width > 0


def test_unclosed_fence_in_description_does_not_swallow_report(tmp_path):
    """An unclosed fence in issue.description must not eat later sections."""
    report = CodeReviewReport(
        summary="One issue found",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=1),
        issues=[
            ReviewIssue(
                category="Security",
                severity="High",
                file_path="a.py",
                line_start=1,
                line_end=1,
                title="Unsafe eval",
                description="Bad code:\n```python\neval(x)",  # never closed
                rationale="eval executes arbitrary input",
            )
        ],
        system_design_insights="MARKER_DESIGN",
        recommendations=["MARKER_RECOMMENDATION"],
    )
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(report, output_file)
    content = output_file.read_text()

    assert _fence_runs(content) % 2 == 0, "odd number of fences: document is broken"
    # The sections after the offending field must still be real Markdown,
    # not trapped inside an open code block.
    design_index = content.index("MARKER_DESIGN")
    assert content.count("```", 0, design_index) % 2 == 0
    assert "MARKER_RECOMMENDATION" in content


@pytest.mark.parametrize(
    "field,value",
    [
        ("summary", "Overview:\n```text\nunclosed"),
        ("system_design_insights", "Layers:\n```\nunclosed"),
    ],
)
def test_unclosed_fence_in_prose_fields_is_balanced(tmp_path, field, value):
    """summary and system_design_insights get the same treatment."""
    kwargs = {
        "summary": "ok",
        "metrics": ReviewMetrics(files_analyzed=1),
        "issues": [],
        "system_design_insights": "ok",
        field: value,
    }
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(CodeReviewReport(**kwargs), output_file)
    assert _fence_runs(output_file.read_text()) % 2 == 0


def test_unclosed_fence_in_list_fields_is_balanced(tmp_path):
    """recommendations / improvement_suggestions are model-generated too."""
    report = CodeReviewReport(
        summary="ok",
        metrics=ReviewMetrics(files_analyzed=1),
        issues=[],
        system_design_insights="ok",
        recommendations=["Do this:\n```sh\nmake fix"],
        improvement_suggestions=["And this:\n```sh\nmake lint"],
    )
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(report, output_file)
    assert _fence_runs(output_file.read_text()) % 2 == 0


# ---------------------------------------------------------------------------
# suggested_code wrapping: the fence must be wider than anything inside it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        ("x = 1", "```"),  # no fences inside
        ("```py\nx\n```", "````"),  # 3 inside -> 4 outside
        ("````md\n```py\nx\n```\n````", "`````"),  # 4 inside -> 5 outside
        ("`````\nx\n`````", "``````"),  # 5 inside -> 6 outside
        ("use ``` inline", "```"),  # inline span can't close a block
        ("  ```py\n  x\n  ```", "````"),  # indented fence lines still count
    ],
)
def test_enclosing_fence_is_wider_than_every_inner_fence(code, expected):
    """CommonMark closes on the first fence at least as long as the opener."""
    from codereview.renderer import enclosing_fence

    assert enclosing_fence(code) == expected


def test_four_backtick_snippet_does_not_swallow_the_rest_of_the_report(tmp_path):
    """A suggested_code snippet containing ```` must not truncate the export.

    The wrapper used to be a hardcoded four backticks, chosen for the common
    "snippet contains ```" case. A snippet holding a *four*-backtick fence (real
    whenever the reviewed file is Markdown, or code that emits fenced blocks)
    then closed the wrapper early, and the wrapper's own closing fence opened a
    new block that ran to the end of the document — silently eating References,
    every later issue, System Design Insights, Recommendations and Metrics. The
    export still "succeeded", which is what makes this worse than a hard error.
    """
    snippet = "Docs example:\n````markdown\n```python\nprint(1)\n```\n````"
    report = CodeReviewReport(
        summary="One issue found",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=1),
        issues=[
            ReviewIssue(
                category="Documentation",
                severity="Low",
                file_path="README.md",
                line_start=1,
                line_end=1,
                title="Nested fence example is wrong",
                description="See the fix",
                rationale="The example does not render",
                suggested_code=snippet,
            )
        ],
        system_design_insights="MARKER_DESIGN",
        recommendations=["MARKER_RECOMMENDATION"],
        improvement_suggestions=["MARKER_SUGGESTION"],
    )
    output_file = tmp_path / "report.md"
    MarkdownExporter().export(report, output_file)
    content = output_file.read_text()

    # Counting fence *lines* for parity is not enough here: this snippet's own
    # fences happen to balance, so a naive even/odd check passes on a document
    # CommonMark still renders wrong. The width rule is what decides it, so the
    # assertion has to apply the width rule.
    for marker in ("MARKER_DESIGN", "MARKER_RECOMMENDATION", "MARKER_SUGGESTION"):
        assert not _inside_code_block(content, marker), (
            f"{marker} is trapped inside a code block; the snippet's own fence "
            "closed the wrapper early"
        )
    assert not _inside_code_block(content, None), "document ends mid-code-block"

    # And the snippet itself must survive verbatim — an early close corrupts it
    # into prose even when nothing later is swallowed.
    assert snippet in content
