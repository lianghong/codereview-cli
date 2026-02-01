"""LangChain-based CLI tool for AI-powered code reviews.

This package provides automated code review capabilities using multiple LLM
backends (AWS Bedrock, Azure OpenAI, NVIDIA NIM). It supports Python, Go,
Shell, C++, Java, JavaScript, and TypeScript codebases with structured output
including issue categories, severity levels, line numbers, and suggested fixes.

Main classes:
    CodeAnalyzer: Orchestrates code analysis using provider abstraction
    CodeReviewReport: Aggregated report with issues, metrics, and recommendations
    ReviewIssue: Individual code review finding
    FileScanner: Discovers and filters code files for analysis
"""

import warnings

# Suppress Pydantic V1 compatibility warning from LangChain (Python 3.14+)
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from codereview.analyzer import CodeAnalyzer  # noqa: E402
from codereview.models import CodeReviewReport, ReviewIssue  # noqa: E402
from codereview.scanner import FileScanner  # noqa: E402

__all__ = ["CodeAnalyzer", "CodeReviewReport", "ReviewIssue", "FileScanner"]
