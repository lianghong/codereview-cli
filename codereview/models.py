# codereview/models.py
from pydantic import BaseModel, Field
from typing import Literal


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


class CodeReviewReport(BaseModel):
    """Aggregated code review report."""

    summary: str = Field(description="Executive summary")
    metrics: dict = Field(description="Analysis metrics")
    issues: list[ReviewIssue] = Field(description="All identified issues")
    system_design_insights: str = Field(description="Architectural observations")
    recommendations: list[str] = Field(description="Top priority actions")
