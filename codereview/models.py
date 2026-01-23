# codereview/models.py
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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
    ] = Field(description="Issue category")

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: str) -> str:
        """Normalize category to valid value, handling LLM variations."""
        if v in VALID_CATEGORIES:
            return v
        # Try case-insensitive lookup
        normalized = CATEGORY_MAPPING.get(v.lower().strip())
        if normalized:
            return normalized
        # Default fallback for unknown categories
        return "Code Quality"

    severity: Literal["Critical", "High", "Medium", "Low", "Info"] = Field(
        description="Issue severity level"
    )

    file_path: str = Field(description="Relative path to file")
    line_start: int = Field(description="Starting line number", ge=1)
    line_end: int | None = Field(default=None, description="Ending line number", ge=1)

    title: str = Field(description="Brief issue summary")
    description: str = Field(description="Detailed explanation")
    suggested_code: str | None = Field(default=None, description="Suggested fix")
    rationale: str = Field(description="Why this matters")
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
        description="Balanced executive summary including: overall quality assessment, "
        "key strengths, main concerns, and priority focus areas"
    )
    metrics: dict[str, Any] = Field(description="Analysis metrics")
    issues: list[ReviewIssue] = Field(description="All identified issues")
    system_design_insights: str = Field(
        description="Architectural observations covering both strengths and concerns"
    )
    recommendations: list[str] = Field(description="Top priority actions")
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Constructive suggestions for code improvement and enhancement "
        "(e.g., better patterns, cleaner abstractions, improved testability)",
    )
