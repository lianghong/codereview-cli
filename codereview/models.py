# codereview/models.py
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any


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
        "Documentation"
    ] = Field(description="Issue category")

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

    @model_validator(mode='after')
    def validate_line_range(self) -> 'ReviewIssue':
        """Ensure line_end is not before line_start."""
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self


class CodeReviewReport(BaseModel):
    """Aggregated code review report."""

    summary: str = Field(description="Executive summary")
    metrics: dict[str, Any] = Field(description="Analysis metrics")
    issues: list[ReviewIssue] = Field(description="All identified issues")
    system_design_insights: str = Field(description="Architectural observations")
    recommendations: list[str] = Field(description="Top priority actions")
