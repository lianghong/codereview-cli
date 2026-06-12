"""Pydantic models for code review data structures.

This module defines the core data models used throughout the code review system:

- ReviewMetrics: Metrics and statistics from code analysis
- ReviewIssue: Individual code review findings with severity, category, and fixes
- CodeReviewReport: Aggregated report containing all review results

The models include validators to normalize LLM output variations (e.g., mapping
"error" to "Critical" severity, "architecture" to "System Design" category).
"""

import logging
import threading
from typing import Any, Literal

from pydantic import (  # type: ignore[attr-defined]
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

# Track values we've already warned about so we don't spam the log when the
# same unknown severity/category shows up on every issue in a batch.
# Validators run inside provider threads (ThreadPoolExecutor in cli.py), so
# the check-then-add must be guarded; otherwise two threads can both pass the
# membership test and emit the same warning twice. The lock is uncontended in
# the common path (already-warned values short-circuit before locking).
_warned_unknown_severities: set[str] = set()
_warned_unknown_categories: set[str] = set()
_warned_lock = threading.Lock()

# Process-wide counters surfaced in the final CLI report so prompt drift is
# visible to users, not just hidden in logs. Reset by reset_drift_counters()
# at the start of each run.
_drift_counters: dict[str, int] = {
    "severity_coerced": 0,
    "category_coerced": 0,
    "reference_dropped": 0,
}
_drift_lock = threading.Lock()


def _bump_drift(key: str) -> None:
    """Thread-safe increment of a drift counter. See _drift_counters."""
    with _drift_lock:
        _drift_counters[key] = _drift_counters.get(key, 0) + 1


def get_drift_counters() -> dict[str, int]:
    """Snapshot the drift counters (thread-safe copy)."""
    with _drift_lock:
        return dict(_drift_counters)


def reset_drift_counters() -> None:
    """Zero the drift counters at the start of a new analysis run."""
    with _drift_lock:
        for k in _drift_counters:
            _drift_counters[k] = 0


# Hosts considered authoritative for review-issue references. Anything else
# gets silently dropped to keep the report free of search-result blogspam.
# Match is on lowercased hostname with leading "www." stripped, suffix-aware
# (so docs.python.org matches "python.org").
_AUTHORITATIVE_REFERENCE_HOSTS = (
    "cwe.mitre.org",
    "cve.mitre.org",
    "nvd.nist.gov",
    "csrc.nist.gov",
    "owasp.org",
    "cheatsheetseries.owasp.org",
    "developer.mozilla.org",
    "docs.python.org",
    "peps.python.org",
    "go.dev",
    "pkg.go.dev",
    "kernel.org",
    "rust-lang.org",
    "doc.rust-lang.org",
    "ecma-international.org",
    "tc39.es",
    "typescriptlang.org",
    "openjdk.org",
    "docs.oracle.com",
    "isocpp.org",
    "en.cppreference.com",
    "pubs.opengroup.org",
)


def _is_authoritative_url(url: str) -> bool:
    """Return True if url's host matches an entry in _AUTHORITATIVE_REFERENCE_HOSTS."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url.strip())
    except ValueError, AttributeError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return any(
        host == h or host.endswith("." + h) for h in _AUTHORITATIVE_REFERENCE_HOSTS
    )


def _warn_once(bucket: set[str], value: str, kind: str, default: str) -> None:
    with _warned_lock:
        if value in bucket:
            return
        bucket.add(value)
    logger.warning(
        "Unknown %s %r from model output; coerced to %r. "
        "This may indicate a model prompt/schema drift.",
        kind,
        value,
        default,
    )


# Valid categories for issues
VALID_CATEGORIES = (
    "Code Style",
    "Code Quality",
    "Security",
    "Performance",
    "Best Practices",
    "System Design",
    "Testing",
    "Documentation",
)

# Valid severity levels
VALID_SEVERITIES = ("Critical", "High", "Medium", "Low", "Info")

# Map common LLM severity variations to valid severities
SEVERITY_MAPPING = {
    # Critical variations
    "critical": "Critical",
    "severe": "Critical",
    "blocker": "Critical",
    "fatal": "Critical",
    "error": "Critical",
    # High variations
    "high": "High",
    "major": "High",
    "important": "High",
    "significant": "High",
    # Medium variations
    "medium": "Medium",
    "moderate": "Medium",
    "normal": "Medium",
    "warning": "Medium",
    "warn": "Medium",
    # Low variations
    "low": "Low",
    "minor": "Low",
    "trivial": "Low",
    "cosmetic": "Low",
    # Info variations
    "info": "Info",
    "information": "Info",
    "informational": "Info",
    "note": "Info",
    "suggestion": "Info",
    "hint": "Info",
}

# Map common LLM category variations to valid categories
CATEGORY_MAPPING = {
    # Code Style variations
    "style": "Code Style",
    "formatting": "Code Style",
    "code style": "Code Style",
    "codestyle": "Code Style",
    # Code Quality variations
    "quality": "Code Quality",
    "code quality": "Code Quality",
    "codequality": "Code Quality",
    "maintainability": "Code Quality",
    "readability": "Code Quality",
    "complexity": "Code Quality",
    # Security variations
    "security": "Security",
    "vulnerability": "Security",
    "safety": "Security",
    # Performance variations
    "performance": "Performance",
    "optimization": "Performance",
    "efficiency": "Performance",
    # Best Practices variations
    "best practices": "Best Practices",
    "bestpractices": "Best Practices",
    "best practice": "Best Practices",
    "practices": "Best Practices",
    "convention": "Best Practices",
    "conventions": "Best Practices",
    # System Design variations
    "system design": "System Design",
    "systemdesign": "System Design",
    "design": "System Design",
    "architecture": "System Design",
    "structure": "System Design",
    # Testing variations
    "testing": "Testing",
    "test": "Testing",
    "tests": "Testing",
    "testability": "Testing",
    # Documentation variations
    "documentation": "Documentation",
    "docs": "Documentation",
    "comments": "Documentation",
    # Error handling often maps to Code Quality
    "error handling": "Code Quality",
    "errorhandling": "Code Quality",
    "error": "Code Quality",
    # Type hints often maps to Code Quality
    "type hints": "Code Quality",
    "typing": "Code Quality",
    "types": "Code Quality",
}


class ReviewMetrics(BaseModel):
    """Metrics from code review analysis.

    All fields are optional to allow flexibility in what metrics the LLM reports.
    Uses model_config to set additionalProperties=false for Responses API compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    # Basic analysis metrics (populated by LLM)
    files_analyzed: int | None = Field(
        default=None, description="Number of files analyzed"
    )
    total_lines: int | None = Field(
        default=None, description="Total lines of code reviewed"
    )
    total_issues: int | None = Field(
        default=None, description="Total number of issues found"
    )

    # Issue counts by severity (populated by LLM or CLI aggregation)
    critical_issues: int | None = Field(
        default=None, description="Number of critical issues"
    )
    high_issues: int | None = Field(
        default=None, description="Number of high severity issues"
    )
    medium_issues: int | None = Field(
        default=None, description="Number of medium severity issues"
    )
    low_issues: int | None = Field(
        default=None, description="Number of low severity issues"
    )
    info_issues: int | None = Field(
        default=None, description="Number of info level issues"
    )

    # Issue category counts
    security_issues: int | None = Field(
        default=None, description="Number of security issues"
    )
    performance_issues: int | None = Field(
        default=None, description="Number of performance issues"
    )

    # Quality score
    code_quality_score: int | None = Field(
        default=None, ge=0, le=100, description="Overall code quality score (0-100)"
    )

    # Token usage metrics (populated by CLI after analysis)
    input_tokens: int | None = Field(
        default=None, description="Total input tokens used"
    )
    output_tokens: int | None = Field(
        default=None, description="Total output tokens used"
    )
    total_tokens: int | None = Field(
        default=None, description="Total tokens (input + output)"
    )

    # Cost and model info (populated by CLI)
    model_name: str | None = Field(
        default=None, description="Model name used for analysis"
    )
    input_price_per_million: float | None = Field(
        default=None, description="Input token price per million"
    )
    output_price_per_million: float | None = Field(
        default=None, description="Output token price per million"
    )

    # Static analysis flags (populated by CLI)
    static_analysis_run: bool | None = Field(
        default=None, description="Whether static analysis was run"
    )
    static_tools_passed: int | None = Field(
        default=None, description="Number of static analysis tools that passed"
    )
    static_tools_failed: int | None = Field(
        default=None, description="Number of static analysis tools that failed"
    )
    static_issues_found: int | None = Field(
        default=None, description="Number of issues found by static analysis"
    )


class ReviewIssue(BaseModel):
    """Represents a single code review issue."""

    category: Literal[
        "Code Style",
        "Code Quality",
        "Security",
        "Performance",
        "Best Practices",
        "System Design",
        "Testing",
        "Documentation",
    ] = Field(default="Code Quality", description="Issue category")

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: Any) -> str:
        """Normalize category to valid value, handling LLM variations."""
        if v is None:
            return "Code Quality"
        if not isinstance(v, str):
            _warn_once(
                _warned_unknown_categories,
                type(v).__name__,
                "category type",
                "Code Quality",
            )
            _bump_drift("category_coerced")
            return "Code Quality"
        if v in VALID_CATEGORIES:
            return v
        normalized = CATEGORY_MAPPING.get(v.lower().strip())
        if normalized:
            return normalized
        _warn_once(_warned_unknown_categories, v, "category", "Code Quality")
        _bump_drift("category_coerced")
        return "Code Quality"

    severity: Literal["Critical", "High", "Medium", "Low", "Info"] = Field(
        default="Medium",  # Default for models that don't return severity
        description="Issue severity level",
    )

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> str:
        """Normalize severity to valid value, handling LLM variations."""
        if v is None:
            return "Medium"  # Default for missing severity
        if not isinstance(v, str):
            _warn_once(
                _warned_unknown_severities,
                type(v).__name__,
                "severity type",
                "Medium",
            )
            _bump_drift("severity_coerced")
            return "Medium"
        if v in VALID_SEVERITIES:
            return v
        normalized = SEVERITY_MAPPING.get(v.lower().strip())
        if normalized:
            return normalized
        _warn_once(_warned_unknown_severities, v, "severity", "Medium")
        _bump_drift("severity_coerced")
        return "Medium"

    # The five fields below are REQUIRED. Removing the defaults makes
    # them required in the JSON schema sent to the LLM via
    # with_structured_output, so a malformed model response that omits
    # them raises ValidationError (which the retry loop already handles).
    # Without `...` here, Pydantic fills in placeholders ("Issue",
    # "No description provided", line 1, "unknown") and the CLI counts
    # them as real findings — see the placeholder model_validator below
    # for the second line of defense against models that emit those
    # exact strings as conscious values.
    file_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Relative path to the file as it appears in the batch header "
            "(e.g. 'app/api/views.py'). Must be a real path from the batch — "
            "do NOT invent paths or use 'unknown'."
        ),
    )
    line_start: int = Field(
        ...,
        description=(
            "Starting line number where the issue actually appears. Must point "
            "to a real line in the provided code. NEVER use 1 as a placeholder "
            "when the issue is not actually on line 1."
        ),
        ge=1,
    )
    line_end: int | None = Field(
        default=None,
        description="Ending line number, if the issue spans multiple lines. Omit for single-line issues.",
        ge=1,
    )

    title: str = Field(
        ...,
        min_length=1,
        description=(
            "Specific, concrete problem in 5-12 words. "
            "GOOD: 'SQL injection via f-string in user lookup query'. "
            "BAD: 'Issue', 'Problem found', 'Code quality concern', 'Bug'."
        ),
    )
    description: str = Field(
        ...,
        min_length=1,
        description=(
            "What is wrong AND the concrete consequence. Reference the actual "
            "code (variable/function names) shown in the batch. Do NOT emit "
            "'No description provided' — omit the issue if you cannot describe it."
        ),
    )
    suggested_code: str | None = Field(
        default=None,
        description=(
            "A concrete code fix in the SAME LANGUAGE as the file. Omit (null) "
            "if a code-level fix is not applicable; never output a description "
            "of the fix in this field — put that in `description`."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description=(
            "Why this matters — the security, reliability, correctness, or "
            "maintainability impact. Tie to a concrete consequence, not a "
            "generic 'best practice'. Do NOT emit 'Review recommended'."
        ),
    )
    references: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of authoritative URLs (CWE, CVE, OWASP, official "
            "language docs). Omit search engine links, blog posts, and "
            "Stack Overflow."
        ),
    )

    @field_validator("references", mode="before")
    @classmethod
    def filter_references(cls, v: Any) -> list[str]:
        """Drop reference URLs that aren't from authoritative sources.

        Models given a free-text URL field reliably mix in search-result
        blogspam, Stack Overflow links, and made-up URLs. We keep only
        well-known authoritative hosts (CWE, OWASP, language docs, MDN,
        NIST). Dropped URLs are counted in the drift counter so the CLI
        can surface "N references dropped" if it spikes.
        """
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        kept: list[str] = []
        for item in v:
            if not isinstance(item, str):
                _bump_drift("reference_dropped")
                continue
            if _is_authoritative_url(item):
                kept.append(item)
            else:
                _bump_drift("reference_dropped")
                logger.debug("Dropped non-authoritative reference URL: %r", item)
        return kept

    @model_validator(mode="after")
    def validate_line_range(self) -> "ReviewIssue":
        """Ensure line_end is not before line_start."""
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self

    @model_validator(mode="after")
    def reject_placeholder_values(self) -> "ReviewIssue":
        """Reject the placeholder strings the prompt explicitly forbids.

        Even with required fields, a model can emit ``"Issue"`` /
        ``"No description provided"`` / ``"Review recommended"`` /
        ``"unknown"`` as conscious values. Those are the same defaults
        the schema used to fill in, just generated by the model rather
        than by Pydantic. Reject them at the schema boundary so the
        retry loop gets a chance to ask for real content; otherwise
        these synthetic findings pollute metrics and the markdown
        report exactly the way silent defaults used to.

        Comparison is case-insensitive on the trimmed value to catch
        capitalization variants ("ISSUE", " issue ") without
        false-flagging legitimate text that happens to contain the
        word "issue" inside a longer string.
        """
        forbidden_pairs = (
            ("file_path", self.file_path, {"unknown", ""}),
            ("title", self.title, {"issue", "problem found", "bug"}),
            (
                "description",
                self.description,
                {"no description provided", "n/a", "none"},
            ),
            ("rationale", self.rationale, {"review recommended", "n/a", "none"}),
        )
        for field_name, value, forbidden in forbidden_pairs:
            if value.strip().lower() in forbidden:
                raise ValueError(
                    f"{field_name}={value!r} is a placeholder forbidden by the "
                    "review-output contract; the prompt asks the model to omit "
                    "the issue entirely when it has no real content for this field"
                )
        return self


class CodeReviewReport(BaseModel):
    """Aggregated code review report."""

    summary: str = Field(
        default="Code review completed",
        description="Balanced executive summary including: overall quality assessment, "
        "key strengths, main concerns, and priority focus areas",
    )
    metrics: ReviewMetrics = Field(
        default_factory=ReviewMetrics, description="Analysis metrics"
    )
    issues: list[ReviewIssue] = Field(
        default_factory=list, description="All identified issues"
    )
    system_design_insights: str = Field(
        default="No architectural concerns identified",
        description="Architectural observations covering both strengths and concerns",
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Top priority actions"
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Constructive suggestions for code improvement and enhancement "
        "(e.g., better patterns, cleaner abstractions, improved testability)",
    )
