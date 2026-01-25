# codereview/models.py
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    # Legacy severity count fields (for backward compatibility with CLI aggregation)
    critical: int | None = Field(default=None, description="Alias for critical_issues")
    high: int | None = Field(default=None, description="Alias for high_issues")
    medium: int | None = Field(default=None, description="Alias for medium_issues")
    low: int | None = Field(default=None, description="Alias for low_issues")
    info: int | None = Field(default=None, description="Alias for info_issues")

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
        _ = cls  # Suppress unused parameter warning
        if v is None:
            return "Code Quality"
        if not isinstance(v, str):
            return "Code Quality"
        if v in VALID_CATEGORIES:
            return v
        # Try case-insensitive lookup
        normalized = CATEGORY_MAPPING.get(v.lower().strip())
        if normalized:
            return normalized
        # Default fallback for unknown categories
        return "Code Quality"

    severity: Literal["Critical", "High", "Medium", "Low", "Info"] = Field(
        default="Medium",  # Default for models that don't return severity
        description="Issue severity level",
    )

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> str:
        """Normalize severity to valid value, handling LLM variations."""
        _ = cls  # Suppress unused parameter warning
        if v is None:
            return "Medium"  # Default for missing severity
        if not isinstance(v, str):
            return "Medium"
        if v in VALID_SEVERITIES:
            return v
        # Try case-insensitive lookup
        normalized = SEVERITY_MAPPING.get(v.lower().strip())
        if normalized:
            return normalized
        # Default fallback for unknown severities
        return "Medium"

    file_path: str = Field(default="unknown", description="Relative path to file")
    line_start: int = Field(default=1, description="Starting line number", ge=1)
    line_end: int | None = Field(default=None, description="Ending line number", ge=1)

    title: str = Field(default="Issue", description="Brief issue summary")
    description: str = Field(
        default="No description provided", description="Detailed explanation"
    )
    suggested_code: str | None = Field(default=None, description="Suggested fix")
    rationale: str = Field(default="Review recommended", description="Why this matters")
    references: list[str] = Field(default_factory=list, description="Reference links")

    @model_validator(mode="after")
    def validate_line_range(self) -> "ReviewIssue":
        """Ensure line_end is not before line_start."""
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
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
