"""Code review tool using LLM-based analysis."""

import warnings

# Suppress Pydantic V1 compatibility warning from LangChain (Python 3.14+)
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from codereview.analyzer import CodeAnalyzer  # noqa: E402
from codereview.models import CodeReviewReport, ReviewIssue  # noqa: E402
from codereview.scanner import FileScanner  # noqa: E402

__all__ = ["CodeAnalyzer", "CodeReviewReport", "ReviewIssue", "FileScanner"]
