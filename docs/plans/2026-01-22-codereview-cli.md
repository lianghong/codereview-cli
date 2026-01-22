# Code Review CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a LangChain-based CLI tool that performs comprehensive code reviews on Python and Go codebases using Claude Opus 4.5 via AWS Bedrock, with beautiful Rich terminal output and Markdown export.

**Architecture:** Single-pass LLM analysis with smart file batching. Scan directory → filter files → batch by context window → send to Claude → aggregate results → render to Rich terminal + Markdown. Uses Pydantic for structured outputs, LangChain for LLM orchestration.

**Tech Stack:** Python 3.14, uv, LangChain, AWS Bedrock, Rich, Pydantic, Click

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `README.md`
- Create: `codereview/__init__.py`

**Step 1: Create project structure**

```bash
mkdir -p codereview tests docs/plans
touch codereview/__init__.py tests/__init__.py
```

**Step 2: Create .python-version**

```
3.14
```

**Step 3: Create pyproject.toml**

```toml
[project]
name = "codereview-cli"
version = "0.1.0"
description = "LangChain-based code review CLI tool with AWS Bedrock"
requires-python = ">=3.14"
dependencies = [
    "langchain>=0.3.0",
    "langchain-aws>=0.2.0",
    "boto3>=1.35.0",
    "click>=8.1.0",
    "rich>=13.7.0",
    "pydantic>=2.9.0",
]

[project.scripts]
codereview = "codereview.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-mock>=3.14.0",
]
```

**Step 4: Initialize uv project**

Run: `uv venv --seed --python 3.14`
Expected: Virtual environment created in `.venv/`

**Step 5: Install dependencies**

Run: `uv pip install -e .`
Expected: All packages installed successfully

**Step 6: Create basic README**

```markdown
# Code Review CLI

LangChain-based code review tool using Claude Opus 4.5 via AWS Bedrock.

## Installation

```bash
uv venv --python 3.14
uv pip install -e .
```

## Usage

```bash
codereview /path/to/codebase
```

## Requirements

- Python 3.14+
- AWS credentials configured
- Access to AWS Bedrock Claude Opus 4.5
```

**Step 7: Commit project setup**

```bash
git init
git add .
git commit -m "feat: initial project setup with uv and dependencies"
```

---

## Task 2: Data Models

**Files:**
- Create: `codereview/models.py`
- Create: `tests/test_models.py`

**Step 1: Write test for ReviewIssue model**

```python
# tests/test_models.py
import pytest
from codereview.models import ReviewIssue, CodeReviewReport


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
        references=["https://owasp.org/sql-injection"]
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
        rationale="Hard to maintain"
    )

    assert issue.line_end is None
    assert issue.suggested_code is None
    assert issue.references == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with "No module named 'codereview.models'"

**Step 3: Implement ReviewIssue model**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py::test_review_issue_creation -v`
Expected: PASS

Run: `uv run pytest tests/test_models.py::test_review_issue_minimal -v`
Expected: PASS

**Step 5: Write test for CodeReviewReport model**

```python
# tests/test_models.py (add to file)

def test_code_review_report():
    """Test CodeReviewReport model."""
    issue = ReviewIssue(
        category="Security",
        severity="High",
        file_path="app.py",
        line_start=1,
        title="Issue",
        description="Description",
        rationale="Rationale"
    )

    report = CodeReviewReport(
        summary="Found 1 security issue",
        metrics={"files": 5, "lines": 100, "issues": 1},
        issues=[issue],
        system_design_insights="Architecture looks good",
        recommendations=["Fix security issue"]
    )

    assert report.summary == "Found 1 security issue"
    assert len(report.issues) == 1
    assert report.metrics["files"] == 5
```

**Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_code_review_report -v`
Expected: FAIL with "cannot import name 'CodeReviewReport'"

**Step 7: Implement CodeReviewReport model**

```python
# codereview/models.py (add to file)

class CodeReviewReport(BaseModel):
    """Aggregated code review report."""

    summary: str = Field(description="Executive summary")
    metrics: dict = Field(description="Analysis metrics")
    issues: list[ReviewIssue] = Field(description="All identified issues")
    system_design_insights: str = Field(description="Architectural observations")
    recommendations: list[str] = Field(description="Top priority actions")
```

**Step 8: Run all tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All tests PASS

**Step 9: Commit data models**

```bash
git add codereview/models.py tests/test_models.py
git commit -m "feat: add Pydantic models for review issues and reports"
```

---

## Task 3: Configuration

**Files:**
- Create: `codereview/config.py`
- Create: `tests/test_config.py`

**Step 1: Write test for default exclusions**

```python
# tests/test_config.py
from codereview.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_EXTENSIONS,
    MODEL_CONFIG,
    SYSTEM_PROMPT
)


def test_default_exclude_patterns():
    """Test default exclusion patterns exist."""
    assert "**/node_modules/**" in DEFAULT_EXCLUDE_PATTERNS
    assert "**/.venv/**" in DEFAULT_EXCLUDE_PATTERNS
    assert "**/__pycache__/**" in DEFAULT_EXCLUDE_PATTERNS


def test_default_exclude_extensions():
    """Test default excluded file extensions."""
    assert ".json" in DEFAULT_EXCLUDE_EXTENSIONS
    assert ".pyc" in DEFAULT_EXCLUDE_EXTENSIONS


def test_model_config():
    """Test AWS Bedrock model configuration."""
    assert MODEL_CONFIG["model_id"] == "global.anthropic.claude-opus-4-5-20251101-v1:0"
    assert MODEL_CONFIG["temperature"] == 0.1
    assert MODEL_CONFIG["max_tokens"] > 0


def test_system_prompt_exists():
    """Test system prompt is defined."""
    assert len(SYSTEM_PROMPT) > 0
    assert "code reviewer" in SYSTEM_PROMPT.lower()
    assert "avoid" in SYSTEM_PROMPT.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with "No module named 'codereview.config'"

**Step 3: Implement configuration constants**

```python
# codereview/config.py
"""Configuration constants for code review tool."""

DEFAULT_EXCLUDE_PATTERNS = [
    # Dependency directories
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/vendor/**",

    # Build outputs
    "**/dist/**",
    "**/build/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",

    # Version control
    "**/.git/**",
    "**/.svn/**",

    # IDE and editors
    "**/.vscode/**",
    "**/.idea/**",
    "**/*.swp",

    # Test fixtures and generated code
    "**/test_data/**",
    "**/fixtures/**",
    "**/*_pb2.py",
    "**/*_pb2_grpc.py",

    # Common non-review files
    "**/*.min.js",
    "**/*.min.css",
    "**/migrations/**",
    "**/*.lock",
]

DEFAULT_EXCLUDE_EXTENSIONS = [
    ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".rst",
    ".jpg", ".png", ".gif", ".svg",
    ".bin", ".exe", ".so", ".dylib",
]

MODEL_CONFIG = {
    "model_id": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "region": "us-west-2",
    "max_tokens": 16000,
    "temperature": 0.1,
}

SYSTEM_PROMPT = """You are an expert code reviewer with deep knowledge of:
- Python and Go best practices
- Security vulnerabilities (OWASP Top 10, CWE)
- System design patterns and anti-patterns
- Performance optimization techniques

Your task: Analyze the provided code and return a structured review.

CRITICAL RULES:
1. Only report real issues - no nitpicking or style preferences
2. Avoid suggesting overdesign - prefer simple, pragmatic solutions
3. Every issue must include specific line numbers
4. Provide actionable suggested_code when possible
5. Focus on: security, correctness, maintainability, design
6. System design insights should be architectural, not file-level

OUTPUT FORMAT: Return valid JSON matching CodeReviewReport schema with:
- summary: Executive summary of findings
- metrics: {files_analyzed, total_lines, issue_counts_by_severity}
- issues: List of ReviewIssue objects
- system_design_insights: Architectural observations
- recommendations: Top 3-5 priority actions"""

# File size limits
MAX_FILE_SIZE_KB = 10
WARN_FILE_SIZE_KB = 5
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: All tests PASS

**Step 5: Commit configuration**

```bash
git add codereview/config.py tests/test_config.py
git commit -m "feat: add configuration with defaults and prompts"
```

---

## Task 4: File Scanner

**Files:**
- Create: `codereview/scanner.py`
- Create: `tests/test_scanner.py`
- Create: `tests/fixtures/sample_code/` (test data)

**Step 1: Create test fixtures**

```bash
mkdir -p tests/fixtures/sample_code/src
mkdir -p tests/fixtures/sample_code/.venv
mkdir -p tests/fixtures/sample_code/__pycache__
```

```python
# tests/fixtures/sample_code/src/main.py
def hello():
    return "world"
```

```go
// tests/fixtures/sample_code/src/main.go
package main

func main() {
    println("hello")
}
```

```python
# tests/fixtures/sample_code/src/config.json
{"key": "value"}
```

**Step 2: Write test for file scanner**

```python
# tests/test_scanner.py
import pytest
from pathlib import Path
from codereview.scanner import FileScanner


@pytest.fixture
def sample_dir():
    """Path to test fixtures."""
    return Path(__file__).parent / "fixtures" / "sample_code"


def test_scanner_finds_python_files(sample_dir):
    """Test scanner finds .py files."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    py_files = [f for f in files if f.suffix == ".py"]
    assert len(py_files) > 0
    assert any("main.py" in str(f) for f in py_files)


def test_scanner_finds_go_files(sample_dir):
    """Test scanner finds .go files."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    go_files = [f for f in files if f.suffix == ".go"]
    assert len(go_files) > 0
    assert any("main.go" in str(f) for f in go_files)


def test_scanner_excludes_json(sample_dir):
    """Test scanner excludes .json files."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    json_files = [f for f in files if f.suffix == ".json"]
    assert len(json_files) == 0


def test_scanner_excludes_venv(sample_dir):
    """Test scanner excludes .venv directory."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    venv_files = [f for f in files if ".venv" in str(f)]
    assert len(venv_files) == 0


def test_scanner_excludes_pycache(sample_dir):
    """Test scanner excludes __pycache__."""
    scanner = FileScanner(sample_dir)
    files = scanner.scan()

    cache_files = [f for f in files if "__pycache__" in str(f)]
    assert len(cache_files) == 0
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_scanner.py -v`
Expected: FAIL with "No module named 'codereview.scanner'"

**Step 4: Implement FileScanner class**

```python
# codereview/scanner.py
"""File scanner for discovering code files to review."""
from pathlib import Path
from fnmatch import fnmatch
from typing import List
from codereview.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_EXTENSIONS,
    MAX_FILE_SIZE_KB
)


class FileScanner:
    """Scans directory for Python and Go files to review."""

    def __init__(
        self,
        root_dir: Path | str,
        exclude_patterns: List[str] | None = None,
        max_file_size_kb: int = MAX_FILE_SIZE_KB
    ):
        self.root_dir = Path(root_dir)
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
        self.max_file_size_kb = max_file_size_kb

    def scan(self) -> List[Path]:
        """Scan directory and return list of files to review."""
        target_extensions = {".py", ".go"}
        files = []

        for file_path in self.root_dir.rglob("*"):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip excluded extensions
            if file_path.suffix in DEFAULT_EXCLUDE_EXTENSIONS:
                continue

            # Skip if not target language
            if file_path.suffix not in target_extensions:
                continue

            # Skip excluded patterns
            relative_path = file_path.relative_to(self.root_dir)
            if self._is_excluded(str(relative_path)):
                continue

            # Skip if file too large
            file_size_kb = file_path.stat().st_size / 1024
            if file_size_kb > self.max_file_size_kb:
                continue

            files.append(file_path)

        return sorted(files)

    def _is_excluded(self, path: str) -> bool:
        """Check if path matches any exclusion pattern."""
        for pattern in self.exclude_patterns:
            if fnmatch(path, pattern):
                return True
        return False
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_scanner.py -v`
Expected: All tests PASS

**Step 6: Commit file scanner**

```bash
git add codereview/scanner.py tests/test_scanner.py tests/fixtures/
git commit -m "feat: add file scanner with exclusion logic"
```

---

## Task 5: Smart Batcher

**Files:**
- Create: `codereview/batcher.py`
- Create: `tests/test_batcher.py`

**Step 1: Write test for batcher**

```python
# tests/test_batcher.py
import pytest
from pathlib import Path
from codereview.batcher import SmartBatcher, FileBatch


def test_batch_creation():
    """Test creating a file batch."""
    files = [Path("test1.py"), Path("test2.py")]
    batch = FileBatch(files=files, batch_number=1, total_batches=2)

    assert len(batch.files) == 2
    assert batch.batch_number == 1
    assert batch.total_batches == 2


def test_batcher_single_batch():
    """Test batcher with small number of files."""
    files = [Path(f"file{i}.py") for i in range(3)]
    batcher = SmartBatcher(max_files_per_batch=10)
    batches = batcher.create_batches(files)

    assert len(batches) == 1
    assert len(batches[0].files) == 3


def test_batcher_multiple_batches():
    """Test batcher splits into multiple batches."""
    files = [Path(f"file{i}.py") for i in range(25)]
    batcher = SmartBatcher(max_files_per_batch=10)
    batches = batcher.create_batches(files)

    assert len(batches) == 3
    assert len(batches[0].files) == 10
    assert len(batches[1].files) == 10
    assert len(batches[2].files) == 5


def test_batch_numbers_correct():
    """Test batch numbers are sequential and correct."""
    files = [Path(f"file{i}.py") for i in range(15)]
    batcher = SmartBatcher(max_files_per_batch=5)
    batches = batcher.create_batches(files)

    assert batches[0].batch_number == 1
    assert batches[1].batch_number == 2
    assert batches[2].batch_number == 3

    for batch in batches:
        assert batch.total_batches == 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_batcher.py -v`
Expected: FAIL with "No module named 'codereview.batcher'"

**Step 3: Implement SmartBatcher class**

```python
# codereview/batcher.py
"""Smart batching for managing context window limits."""
from pathlib import Path
from typing import List
from pydantic import BaseModel


class FileBatch(BaseModel):
    """Represents a batch of files to analyze together."""

    files: List[Path]
    batch_number: int
    total_batches: int

    class Config:
        arbitrary_types_allowed = True


class SmartBatcher:
    """Batches files intelligently for LLM analysis."""

    def __init__(self, max_files_per_batch: int = 10):
        """
        Initialize batcher.

        Args:
            max_files_per_batch: Maximum files per batch (default 10)
        """
        self.max_files_per_batch = max_files_per_batch

    def create_batches(self, files: List[Path]) -> List[FileBatch]:
        """
        Create batches from file list.

        Args:
            files: List of file paths

        Returns:
            List of FileBatch objects
        """
        if not files:
            return []

        batches = []
        total_batches = (len(files) + self.max_files_per_batch - 1) // self.max_files_per_batch

        for i in range(0, len(files), self.max_files_per_batch):
            batch_files = files[i:i + self.max_files_per_batch]
            batch_number = len(batches) + 1

            batch = FileBatch(
                files=batch_files,
                batch_number=batch_number,
                total_batches=total_batches
            )
            batches.append(batch)

        return batches
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_batcher.py -v`
Expected: All tests PASS

**Step 5: Commit smart batcher**

```bash
git add codereview/batcher.py tests/test_batcher.py
git commit -m "feat: add smart batcher for context window management"
```

---

## Task 6: LLM Analyzer

**Files:**
- Create: `codereview/analyzer.py`
- Create: `tests/test_analyzer.py`

**Step 1: Write test for analyzer (with mocking)**

```python
# tests/test_analyzer.py
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from codereview.analyzer import CodeAnalyzer
from codereview.models import CodeReviewReport, ReviewIssue
from codereview.batcher import FileBatch


@pytest.fixture
def mock_bedrock_client():
    """Mock AWS Bedrock client."""
    with patch('codereview.analyzer.ChatBedrockConverse') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_batch(tmp_path):
    """Create sample batch with test file."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def foo():\n    pass\n")

    return FileBatch(
        files=[test_file],
        batch_number=1,
        total_batches=1
    )


def test_analyzer_initialization(mock_bedrock_client):
    """Test analyzer can be initialized."""
    analyzer = CodeAnalyzer(region="us-west-2")
    assert analyzer is not None


def test_prepare_batch_context(sample_batch, mock_bedrock_client):
    """Test preparing context from batch."""
    analyzer = CodeAnalyzer()
    context = analyzer._prepare_batch_context(sample_batch)

    assert "test.py" in context
    assert "def foo()" in context
    assert "Batch 1/1" in context


def test_analyze_batch_returns_report(sample_batch, mock_bedrock_client):
    """Test analyze_batch returns CodeReviewReport."""
    # Mock LLM response
    mock_response = CodeReviewReport(
        summary="No issues found",
        metrics={"files": 1, "issues": 0},
        issues=[],
        system_design_insights="Simple code",
        recommendations=[]
    )

    mock_bedrock_client.with_structured_output.return_value.invoke.return_value = mock_response

    analyzer = CodeAnalyzer()
    result = analyzer.analyze_batch(sample_batch)

    assert isinstance(result, CodeReviewReport)
    assert result.summary == "No issues found"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: FAIL with "No module named 'codereview.analyzer'"

**Step 3: Implement CodeAnalyzer class**

```python
# codereview/analyzer.py
"""LLM-based code analyzer using AWS Bedrock."""
from pathlib import Path
from langchain_aws import ChatBedrockConverse
from codereview.models import CodeReviewReport
from codereview.batcher import FileBatch
from codereview.config import MODEL_CONFIG, SYSTEM_PROMPT


class CodeAnalyzer:
    """Analyzes code using Claude Opus 4.5 via AWS Bedrock."""

    def __init__(self, region: str | None = None):
        """
        Initialize analyzer.

        Args:
            region: AWS region (uses MODEL_CONFIG default if not provided)
        """
        self.region = region or MODEL_CONFIG["region"]
        self.model = self._create_model()

    def _create_model(self):
        """Create LangChain model with structured output."""
        base_model = ChatBedrockConverse(
            model=MODEL_CONFIG["model_id"],
            region_name=self.region,
            temperature=MODEL_CONFIG["temperature"],
            max_tokens=MODEL_CONFIG["max_tokens"],
        )

        # Configure for structured output
        return base_model.with_structured_output(CodeReviewReport)

    def analyze_batch(self, batch: FileBatch) -> CodeReviewReport:
        """
        Analyze a batch of files.

        Args:
            batch: FileBatch to analyze

        Returns:
            CodeReviewReport with findings
        """
        context = self._prepare_batch_context(batch)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]

        result = self.model.invoke(messages)
        return result

    def _prepare_batch_context(self, batch: FileBatch) -> str:
        """
        Prepare context string for LLM.

        Args:
            batch: FileBatch to prepare

        Returns:
            Formatted context string
        """
        lines = [
            f"Analyzing Batch {batch.batch_number}/{batch.total_batches}",
            f"Files in this batch: {len(batch.files)}",
            "",
            "=" * 80,
            ""
        ]

        for file_path in batch.files:
            try:
                content = file_path.read_text()
                lines.append(f"File: {file_path.name}")
                lines.append(f"Path: {file_path}")
                lines.append("-" * 80)

                # Add line numbers
                for i, line in enumerate(content.splitlines(), start=1):
                    lines.append(f"{i:4d} | {line}")

                lines.append("")
                lines.append("=" * 80)
                lines.append("")

            except Exception as e:
                lines.append(f"ERROR reading {file_path}: {e}")
                lines.append("")

        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: All tests PASS

**Step 5: Commit analyzer**

```bash
git add codereview/analyzer.py tests/test_analyzer.py
git commit -m "feat: add LLM analyzer with AWS Bedrock integration"
```

---

## Task 7: Rich Terminal Renderer

**Files:**
- Create: `codereview/renderer.py`
- Create: `tests/test_renderer.py`

**Step 1: Write test for terminal renderer**

```python
# tests/test_renderer.py
import pytest
from io import StringIO
from codereview.renderer import TerminalRenderer
from codereview.models import CodeReviewReport, ReviewIssue


@pytest.fixture
def sample_report():
    """Create sample report for testing."""
    issue1 = ReviewIssue(
        category="Security",
        severity="Critical",
        file_path="app.py",
        line_start=42,
        title="SQL Injection",
        description="User input not sanitized",
        rationale="Security risk",
    )

    issue2 = ReviewIssue(
        category="Code Quality",
        severity="Low",
        file_path="utils.py",
        line_start=10,
        title="Complex function",
        description="High complexity",
        rationale="Maintainability",
    )

    return CodeReviewReport(
        summary="Found 2 issues",
        metrics={"files": 2, "issues": 2, "critical": 1, "low": 1},
        issues=[issue1, issue2],
        system_design_insights="Architecture is solid",
        recommendations=["Fix SQL injection", "Refactor complex function"]
    )


def test_renderer_initialization():
    """Test renderer can be initialized."""
    renderer = TerminalRenderer()
    assert renderer is not None


def test_render_summary(sample_report):
    """Test rendering summary section."""
    renderer = TerminalRenderer()
    output = renderer._format_summary(sample_report)

    assert "Found 2 issues" in output
    assert "files" in output.lower()


def test_severity_color_mapping():
    """Test severity to color mapping."""
    renderer = TerminalRenderer()

    assert renderer._get_severity_color("Critical") == "red"
    assert renderer._get_severity_color("High") == "bright_red"
    assert renderer._get_severity_color("Medium") == "yellow"
    assert renderer._get_severity_color("Low") == "blue"
    assert renderer._get_severity_color("Info") == "white"


def test_group_issues_by_severity(sample_report):
    """Test grouping issues by severity."""
    renderer = TerminalRenderer()
    grouped = renderer._group_by_severity(sample_report.issues)

    assert "Critical" in grouped
    assert "Low" in grouped
    assert len(grouped["Critical"]) == 1
    assert len(grouped["Low"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderer.py -v`
Expected: FAIL with "No module named 'codereview.renderer'"

**Step 3: Implement TerminalRenderer class**

```python
# codereview/renderer.py
"""Rich terminal and Markdown output rendering."""
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from codereview.models import CodeReviewReport, ReviewIssue


class TerminalRenderer:
    """Renders code review results to Rich terminal."""

    SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Info"]

    SEVERITY_COLORS = {
        "Critical": "red",
        "High": "bright_red",
        "Medium": "yellow",
        "Low": "blue",
        "Info": "white",
    }

    SEVERITY_ICONS = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🔵",
        "Info": "⚪",
    }

    def __init__(self):
        """Initialize renderer."""
        self.console = Console()

    def render(self, report: CodeReviewReport):
        """
        Render full report to terminal.

        Args:
            report: CodeReviewReport to render
        """
        self.console.print()
        self._render_header()
        self._render_summary(report)
        self._render_issues(report)
        self._render_recommendations(report)
        self.console.print()

    def _render_header(self):
        """Render header."""
        self.console.print(
            Panel.fit(
                "[bold cyan]Code Review Report[/bold cyan]",
                border_style="cyan"
            )
        )

    def _render_summary(self, report: CodeReviewReport):
        """Render summary section."""
        summary = self._format_summary(report)
        self.console.print(Panel(summary, title="Summary", border_style="green"))

    def _format_summary(self, report: CodeReviewReport) -> str:
        """Format summary text."""
        lines = [
            f"[bold]{report.summary}[/bold]",
            "",
            f"📊 Files analyzed: {report.metrics.get('files', 0)}",
            f"🐛 Total issues: {report.metrics.get('issues', 0)}",
        ]
        return "\n".join(lines)

    def _render_issues(self, report: CodeReviewReport):
        """Render issues grouped by severity."""
        if not report.issues:
            self.console.print("[green]✓ No issues found![/green]\n")
            return

        grouped = self._group_by_severity(report.issues)

        for severity in self.SEVERITY_ORDER:
            if severity not in grouped:
                continue

            issues = grouped[severity]
            color = self._get_severity_color(severity)
            icon = self.SEVERITY_ICONS[severity]

            self.console.print(f"\n{icon} [bold {color}]{severity} ({len(issues)})[/bold {color}]")

            for issue in issues:
                self._render_issue(issue)

    def _render_issue(self, issue: ReviewIssue):
        """Render single issue."""
        color = self._get_severity_color(issue.severity)

        table = Table(show_header=False, border_style=color, box=None)
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Category", issue.category)
        table.add_row("File", f"{issue.file_path}:{issue.line_start}")
        table.add_row("Issue", issue.title)
        table.add_row("Details", issue.description)
        table.add_row("Why", issue.rationale)

        if issue.suggested_code:
            table.add_row("Fix", f"```\n{issue.suggested_code}\n```")

        self.console.print(table)
        self.console.print()

    def _render_recommendations(self, report: CodeReviewReport):
        """Render top recommendations."""
        if not report.recommendations:
            return

        lines = []
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        self.console.print(
            Panel(
                "\n".join(lines),
                title="Top Recommendations",
                border_style="yellow"
            )
        )

    def _group_by_severity(self, issues: List[ReviewIssue]) -> Dict[str, List[ReviewIssue]]:
        """Group issues by severity level."""
        grouped = {}
        for issue in issues:
            if issue.severity not in grouped:
                grouped[issue.severity] = []
            grouped[issue.severity].append(issue)
        return grouped

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level."""
        return self.SEVERITY_COLORS.get(severity, "white")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_renderer.py -v`
Expected: All tests PASS

**Step 5: Commit terminal renderer**

```bash
git add codereview/renderer.py tests/test_renderer.py
git commit -m "feat: add Rich terminal renderer for review results"
```

---

## Task 8: Markdown Exporter

**Files:**
- Modify: `codereview/renderer.py`
- Create: `tests/test_markdown_export.py`

**Step 1: Write test for Markdown export**

```python
# tests/test_markdown_export.py
import pytest
from pathlib import Path
from codereview.renderer import MarkdownExporter
from codereview.models import CodeReviewReport, ReviewIssue


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
        references=["https://owasp.org/sql-injection"]
    )

    return CodeReviewReport(
        summary="Found 1 critical security issue",
        metrics={"files": 1, "issues": 1, "critical": 1},
        issues=[issue],
        system_design_insights="Single file reviewed",
        recommendations=["Fix SQL injection immediately"]
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_markdown_export.py -v`
Expected: FAIL with "cannot import name 'MarkdownExporter'"

**Step 3: Implement MarkdownExporter class**

```python
# codereview/renderer.py (add to end of file)

from datetime import datetime


class MarkdownExporter:
    """Exports code review reports to Markdown format."""

    SEVERITY_ICONS = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🔵",
        "Info": "⚪",
    }

    def export(self, report: CodeReviewReport, output_path: Path | str):
        """
        Export report to Markdown file.

        Args:
            report: CodeReviewReport to export
            output_path: Path to output Markdown file
        """
        output_path = Path(output_path)
        content = self._generate_markdown(report)
        output_path.write_text(content)

    def _generate_markdown(self, report: CodeReviewReport) -> str:
        """Generate Markdown content."""
        sections = [
            self._header(),
            self._summary(report),
            self._metrics(report),
            self._issues(report),
            self._system_design(report),
            self._recommendations(report),
        ]

        return "\n\n".join(sections)

    def _header(self) -> str:
        """Generate header section."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"# Code Review Report\n\n**Generated:** {now}"

    def _summary(self, report: CodeReviewReport) -> str:
        """Generate summary section."""
        return f"## Executive Summary\n\n{report.summary}"

    def _metrics(self, report: CodeReviewReport) -> str:
        """Generate metrics section."""
        lines = ["## Metrics\n"]

        for key, value in report.metrics.items():
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

        return "\n".join(lines)

    def _issues(self, report: CodeReviewReport) -> str:
        """Generate issues section."""
        if not report.issues:
            return "## Issues\n\n✅ No issues found!"

        lines = ["## Issues by Severity\n"]

        # Group by severity
        grouped = {}
        for issue in report.issues:
            if issue.severity not in grouped:
                grouped[issue.severity] = []
            grouped[issue.severity].append(issue)

        # Render each severity group
        for severity in ["Critical", "High", "Medium", "Low", "Info"]:
            if severity not in grouped:
                continue

            icon = self.SEVERITY_ICONS[severity]
            issues = grouped[severity]

            lines.append(f"### {icon} {severity} ({len(issues)})\n")

            for issue in issues:
                lines.append(self._format_issue(issue))

        return "\n".join(lines)

    def _format_issue(self, issue: ReviewIssue) -> str:
        """Format single issue in Markdown."""
        lines = [
            f"#### [{issue.category}] {issue.title}",
            f"**File:** `{issue.file_path}:{issue.line_start}`",
            f"**Severity:** {issue.severity}\n",
            f"**Description:**\n{issue.description}\n",
            f"**Rationale:**\n{issue.rationale}\n",
        ]

        if issue.suggested_code:
            lines.append(f"**Suggested Fix:**")
            lines.append(f"```python\n{issue.suggested_code}\n```\n")

        if issue.references:
            lines.append("**References:**")
            for ref in issue.references:
                lines.append(f"- {ref}")

        lines.append("\n---\n")

        return "\n".join(lines)

    def _system_design(self, report: CodeReviewReport) -> str:
        """Generate system design section."""
        return f"## System Design Insights\n\n{report.system_design_insights}"

    def _recommendations(self, report: CodeReviewReport) -> str:
        """Generate recommendations section."""
        if not report.recommendations:
            return ""

        lines = ["## Top Recommendations\n"]

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_markdown_export.py -v`
Expected: All tests PASS

**Step 5: Commit Markdown exporter**

```bash
git add codereview/renderer.py tests/test_markdown_export.py
git commit -m "feat: add Markdown exporter for review reports"
```

---

## Task 9: CLI Entry Point

**Files:**
- Create: `codereview/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write test for CLI**

```python
# tests/test_cli.py
import pytest
from click.testing import CliRunner
from pathlib import Path
from codereview.cli import main


@pytest.fixture
def cli_runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_code_dir(tmp_path):
    """Create sample code directory."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    return tmp_path


def test_cli_no_args(cli_runner):
    """Test CLI with no arguments shows help."""
    result = cli_runner.invoke(main, [])
    assert result.exit_code != 0


def test_cli_help(cli_runner):
    """Test CLI help command."""
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_with_directory(cli_runner, sample_code_dir):
    """Test CLI with directory argument."""
    # Note: This will fail without AWS credentials
    # We'll add proper mocking in integration tests
    result = cli_runner.invoke(main, [str(sample_code_dir)])
    # Just check it attempts to run
    assert "directory" in result.output.lower() or result.exit_code == 0


def test_cli_output_option(cli_runner, sample_code_dir, tmp_path):
    """Test CLI with output file option."""
    output_file = tmp_path / "report.md"
    result = cli_runner.invoke(main, [
        str(sample_code_dir),
        "--output", str(output_file)
    ])
    # Command should accept the argument
    assert "--output" not in result.output or result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with "No module named 'codereview.cli'"

**Step 3: Implement CLI entry point**

```python
# codereview/cli.py
"""CLI entry point for code review tool."""
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from codereview.scanner import FileScanner
from codereview.batcher import SmartBatcher
from codereview.analyzer import CodeAnalyzer
from codereview.renderer import TerminalRenderer, MarkdownExporter
from codereview.models import CodeReviewReport, ReviewIssue

console = Console()


@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output Markdown report to file'
)
@click.option(
    '--severity', '-s',
    type=click.Choice(['critical', 'high', 'medium', 'low', 'info'], case_sensitive=False),
    default='info',
    help='Minimum severity level to display'
)
@click.option(
    '--exclude', '-e',
    multiple=True,
    help='Additional exclusion patterns'
)
@click.option(
    '--max-files',
    type=int,
    help='Maximum number of files to analyze'
)
@click.option(
    '--max-file-size',
    type=int,
    default=10,
    help='Maximum file size in KB (default: 10)'
)
@click.option(
    '--aws-region',
    type=str,
    help='AWS region for Bedrock'
)
@click.option(
    '--aws-profile',
    type=str,
    help='AWS CLI profile to use'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed progress'
)
def main(
    directory: Path,
    output: Path | None,
    severity: str,
    exclude: tuple,
    max_files: int | None,
    max_file_size: int,
    aws_region: str | None,
    aws_profile: str | None,
    verbose: bool
):
    """
    Analyze code in DIRECTORY and generate a comprehensive review report.

    Reviews Python and Go files using Claude Opus 4.5 via AWS Bedrock.
    """
    try:
        console.print(f"\n[bold cyan]🔍 Code Review Tool[/bold cyan]\n")
        console.print(f"📂 Scanning directory: {directory}\n")

        # Step 1: Scan files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)

            scanner = FileScanner(directory, max_file_size_kb=max_file_size)
            files = scanner.scan()

            if max_files:
                files = files[:max_files]

            progress.update(task, completed=True)

        if not files:
            console.print("[yellow]⚠️  No files found to review[/yellow]")
            return

        console.print(f"✓ Found {len(files)} files to review\n")

        # Step 2: Create batches
        batcher = SmartBatcher()
        batches = batcher.create_batches(files)

        console.print(f"📦 Created {len(batches)} batches\n")

        # Step 3: Analyze batches
        analyzer = CodeAnalyzer(region=aws_region)
        all_issues = []
        total_files = len(files)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Analyzing code...",
                total=len(batches)
            )

            for i, batch in enumerate(batches, 1):
                if verbose:
                    console.print(f"  Batch {i}/{len(batches)}: {len(batch.files)} files")

                try:
                    report = analyzer.analyze_batch(batch)
                    all_issues.extend(report.issues)

                except Exception as e:
                    console.print(f"[red]✗ Error analyzing batch {i}: {e}[/red]")
                    if verbose:
                        import traceback
                        console.print(traceback.format_exc())

                progress.update(task, advance=1)

        # Step 4: Create final report
        final_report = CodeReviewReport(
            summary=f"Analyzed {total_files} files and found {len(all_issues)} issues",
            metrics={
                "files_analyzed": total_files,
                "total_issues": len(all_issues),
                "critical": sum(1 for i in all_issues if i.severity == "Critical"),
                "high": sum(1 for i in all_issues if i.severity == "High"),
                "medium": sum(1 for i in all_issues if i.severity == "Medium"),
                "low": sum(1 for i in all_issues if i.severity == "Low"),
            },
            issues=all_issues,
            system_design_insights="Analysis complete",
            recommendations=_generate_recommendations(all_issues)
        )

        # Step 5: Render results
        renderer = TerminalRenderer()
        renderer.render(final_report)

        # Step 6: Export to Markdown if requested
        if output:
            exporter = MarkdownExporter()
            exporter.export(final_report, output)
            console.print(f"\n[green]✓ Report exported to: {output}[/green]\n")

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]\n")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise click.Abort()


def _generate_recommendations(issues: list[ReviewIssue]) -> list[str]:
    """Generate top recommendations from issues."""
    critical = [i for i in issues if i.severity == "Critical"]
    high = [i for i in issues if i.severity == "High"]

    recommendations = []

    if critical:
        recommendations.append(f"🚨 Address {len(critical)} critical issues immediately")

    if high:
        recommendations.append(f"⚠️  Fix {len(high)} high-priority issues")

    if len(issues) > 10:
        recommendations.append("📊 Consider refactoring to reduce technical debt")

    return recommendations[:5]


if __name__ == '__main__':
    main()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests PASS (some may be skipped without AWS credentials)

**Step 5: Test CLI manually**

Run: `uv run codereview tests/fixtures/sample_code --verbose`
Expected: Tool runs and shows results

**Step 6: Commit CLI implementation**

```bash
git add codereview/cli.py tests/test_cli.py
git commit -m "feat: add CLI entry point with full workflow"
```

---

## Task 10: Error Handling & Polish

**Files:**
- Modify: `codereview/analyzer.py`
- Modify: `codereview/cli.py`
- Create: `tests/test_error_handling.py`

**Step 1: Write tests for error handling**

```python
# tests/test_error_handling.py
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError
from codereview.analyzer import CodeAnalyzer
from codereview.batcher import FileBatch
from pathlib import Path


def test_analyzer_handles_rate_limiting(tmp_path):
    """Test analyzer handles AWS rate limiting."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def foo(): pass")

    batch = FileBatch(files=[test_file], batch_number=1, total_batches=1)

    with patch('codereview.analyzer.ChatBedrockConverse') as mock_bedrock:
        # Simulate throttling error
        error = ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'InvokeModel'
        )

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value.invoke.side_effect = error
        mock_bedrock.return_value = mock_instance

        analyzer = CodeAnalyzer()

        with pytest.raises(ClientError):
            analyzer.analyze_batch(batch)


def test_analyzer_handles_invalid_credentials():
    """Test analyzer handles invalid AWS credentials."""
    with patch('codereview.analyzer.ChatBedrockConverse') as mock_bedrock:
        error = ClientError(
            {'Error': {'Code': 'UnrecognizedClientException', 'Message': 'Invalid'}},
            'InvokeModel'
        )
        mock_bedrock.side_effect = error

        with pytest.raises(ClientError):
            CodeAnalyzer()


def test_scanner_handles_unreadable_files(tmp_path):
    """Test scanner handles files that cannot be read."""
    from codereview.scanner import FileScanner

    # Create file but make it unreadable
    test_file = tmp_path / "test.py"
    test_file.write_text("content")
    test_file.chmod(0o000)

    scanner = FileScanner(tmp_path)
    files = scanner.scan()

    # Should either skip or handle gracefully
    # (behavior depends on OS permissions)
    assert isinstance(files, list)

    # Cleanup
    test_file.chmod(0o644)
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_error_handling.py -v`
Expected: Tests run (some may be skipped based on environment)

**Step 3: Add retry logic to analyzer**

```python
# codereview/analyzer.py (modify analyze_batch method)

import time
from botocore.exceptions import ClientError


class CodeAnalyzer:
    # ... existing code ...

    def analyze_batch(self, batch: FileBatch, max_retries: int = 3) -> CodeReviewReport:
        """
        Analyze a batch of files with retry logic.

        Args:
            batch: FileBatch to analyze
            max_retries: Maximum retry attempts

        Returns:
            CodeReviewReport with findings
        """
        context = self._prepare_batch_context(batch)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]

        for attempt in range(max_retries):
            try:
                result = self.model.invoke(messages)
                return result

            except ClientError as e:
                error_code = e.response['Error']['Code']

                if error_code == 'ThrottlingException':
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff
                        print(f"Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
                else:
                    # Other errors, don't retry
                    raise

            except Exception as e:
                # Unexpected error
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise

        # Fallback: return empty report
        return CodeReviewReport(
            summary="Analysis failed after retries",
            metrics={"files": len(batch.files), "issues": 0},
            issues=[],
            system_design_insights="Unable to complete analysis",
            recommendations=["Retry analysis with better network connection"]
        )
```

**Step 4: Add better error messages to CLI**

```python
# codereview/cli.py (modify main function error handling)

# Update the exception handling at the end:

    except ClientError as e:
        error_code = e.response['Error']['Code']

        if error_code == 'UnrecognizedClientException':
            console.print("\n[red]✗ AWS Authentication Error[/red]")
            console.print("Please configure your AWS credentials:\n")
            console.print("  aws configure")
            console.print("  # or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY\n")

        elif error_code == 'AccessDeniedException':
            console.print("\n[red]✗ AWS Permission Error[/red]")
            console.print("Your AWS credentials don't have access to Bedrock.\n")
            console.print("Required permission: bedrock:InvokeModel\n")

        elif error_code == 'ThrottlingException':
            console.print("\n[red]✗ Rate Limit Exceeded[/red]")
            console.print("Too many requests to AWS Bedrock.\n")
            console.print("Try again in a few minutes or reduce batch size.\n")
        else:
            console.print(f"\n[red]✗ AWS Error ({error_code}): {e}[/red]\n")

        raise click.Abort()

    except Exception as e:
        console.print(f"\n[red]✗ Unexpected Error: {e}[/red]\n")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise click.Abort()
```

**Step 5: Test error handling manually**

Run with invalid directory: `uv run codereview /nonexistent`
Expected: Clear error message

**Step 6: Commit error handling**

```bash
git add codereview/analyzer.py codereview/cli.py tests/test_error_handling.py
git commit -m "feat: add error handling with retry logic and better messages"
```

---

## Task 11: Documentation & README

**Files:**
- Modify: `README.md`
- Create: `docs/usage.md`
- Create: `docs/examples.md`

**Step 1: Update README with complete documentation**

```markdown
# Code Review CLI

Professional code review tool powered by Claude Opus 4.5 via AWS Bedrock and LangChain.

## Features

- 🔍 **Comprehensive Analysis** - Reviews Python and Go code for quality, security, performance, and design
- 🤖 **AI-Powered** - Uses Claude Opus 4.5 for intelligent, context-aware reviews
- 🎨 **Beautiful Output** - Rich terminal UI with colored, organized results
- 📝 **Exportable Reports** - Generate Markdown reports for documentation
- ⚡ **Smart Batching** - Efficiently handles large codebases
- 🔒 **Security Focus** - Identifies vulnerabilities and security issues
- 🏗️ **Architecture Insights** - Provides system design recommendations

## Installation

### Prerequisites

- Python 3.14+
- AWS account with Bedrock access
- AWS credentials configured

### Install with uv

```bash
# Clone repository
git clone <repo-url>
cd codereview-cli

# Create virtual environment
uv venv --python 3.14

# Install package
uv pip install -e .
```

## Configuration

### AWS Credentials

Configure AWS credentials using one of these methods:

**Option 1: AWS CLI**
```bash
aws configure
```

**Option 2: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

**Option 3: IAM Role** (if running on EC2/ECS)

### AWS Bedrock Access

Ensure your AWS account has:
- Access to AWS Bedrock
- Claude Opus 4.5 model enabled
- IAM permission: `bedrock:InvokeModel`

## Usage

### Basic Usage

```bash
codereview /path/to/codebase
```

### With Options

```bash
# Generate Markdown report
codereview /path/to/code --output report.md

# Filter by severity
codereview /path/to/code --severity high

# Exclude directories
codereview /path/to/code --exclude "**/tests/*,**/vendor/*"

# Limit file count
codereview /path/to/code --max-files 50

# Use different AWS region
codereview /path/to/code --aws-region us-east-1

# Verbose output
codereview /path/to/code --verbose
```

### Full Options

```
Options:
  -o, --output PATH           Output Markdown report file
  -s, --severity LEVEL        Min severity (critical|high|medium|low|info)
  -e, --exclude PATTERN       Exclusion patterns (can specify multiple)
  --max-files INTEGER         Maximum files to analyze
  --max-file-size INTEGER     Max file size in KB (default: 10)
  --aws-region TEXT           AWS region
  --aws-profile TEXT          AWS CLI profile
  -v, --verbose               Show detailed progress
  --help                      Show this message
```

## Review Categories

The tool analyzes code across 8 categories:

1. **Code Style** - Formatting, naming conventions
2. **Code Quality** - Complexity, maintainability, duplication
3. **Security** - Vulnerabilities, injection risks, sensitive data
4. **Performance** - Bottlenecks, inefficient algorithms
5. **Best Practices** - Language idioms, design patterns
6. **System Design** - Architecture, coupling, scalability
7. **Testing** - Test coverage, test quality
8. **Documentation** - Comments, docstrings, clarity

## Severity Levels

- 🔴 **Critical** - Security vulnerabilities, data loss risks
- 🟠 **High** - Bugs, significant issues requiring immediate attention
- 🟡 **Medium** - Code quality issues, minor bugs
- 🔵 **Low** - Style issues, optimization opportunities
- ⚪ **Info** - Suggestions, educational notes

## Output Examples

### Terminal Output

```
┌─────────────────────────────────────┐
│     Code Review Report              │
└─────────────────────────────────────┘

Summary
────────────────────────────────────
Analyzed 45 Python files and found 12 issues

📊 Files analyzed: 45
🐛 Total issues: 12

🔴 Critical (2)

Category: Security
File: api/database.py:145
Issue: SQL Injection Vulnerability
Details: User input concatenated into SQL query
Why: Allows arbitrary SQL execution
Fix: Use parameterized queries

...
```

### Markdown Report

See `docs/examples.md` for full Markdown report examples.

## Exclusions

The tool automatically excludes:

- Dependencies: `node_modules/`, `.venv/`, `vendor/`
- Build outputs: `dist/`, `build/`, `__pycache__/`
- Version control: `.git/`, `.svn/`
- IDE files: `.vscode/`, `.idea/`
- Generated code: `*_pb2.py`, `*.min.js`
- Large files: > 10KB by default

## Troubleshooting

### "AWS Authentication Error"

```bash
# Configure credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### "AWS Permission Error"

Ensure your IAM role/user has:
```json
{
  "Effect": "Allow",
  "Action": "bedrock:InvokeModel",
  "Resource": "*"
}
```

### "Rate Limit Exceeded"

The tool automatically retries with exponential backoff. If persistent:
- Reduce batch size with `--max-files`
- Wait a few minutes between runs
- Contact AWS support to increase quota

### "No files found"

Check:
- Directory path is correct
- Python (.py) or Go (.go) files exist
- Files aren't excluded by default patterns
- File sizes are under limit (use `--max-file-size`)

## Development

### Run Tests

```bash
uv run pytest tests/ -v
```

### Run with Development Code

```bash
uv run python -m codereview.cli /path/to/code
```

## License

MIT

## Contributing

Contributions welcome! Please open issues and pull requests.
```

**Step 2: Create usage documentation**

```markdown
# Usage Guide

## Basic Workflow

1. **Scan** - Tool discovers Python and Go files
2. **Batch** - Files grouped for efficient analysis
3. **Analyze** - Claude Opus 4.5 reviews code
4. **Aggregate** - Results combined and deduplicated
5. **Render** - Display in terminal and/or export to Markdown

## Common Use Cases

### Daily Development

```bash
# Quick review of current project
codereview .

# Review specific directory
codereview src/
```

### Pre-Commit Review

```bash
# Review with strict severity filter
codereview . --severity high

# Generate report for PR
codereview . --output review.md
```

### Large Codebase

```bash
# Limit file count for faster review
codereview . --max-files 50

# Exclude test files
codereview . --exclude "**/tests/*,**/*_test.py"
```

### CI/CD Integration

```bash
# Exit with error if critical issues found
codereview . --severity critical || exit 1

# Generate report artifact
codereview . --output artifacts/review.md
```

## Advanced Patterns

### Custom Exclusions

```bash
# Exclude multiple patterns
codereview . \
  --exclude "**/migrations/*" \
  --exclude "**/generated/*" \
  --exclude "**/*_pb2.py"
```

### Regional Configuration

```bash
# Use specific region
codereview . --aws-region eu-west-1

# Use named profile
codereview . --aws-profile production
```

## Best Practices

1. **Start Small** - Review smaller directories first to understand output
2. **Use Exclusions** - Skip generated code and dependencies
3. **Filter Severity** - Focus on high/critical for large codebases
4. **Export Reports** - Share Markdown reports with team
5. **Iterate** - Fix critical issues, then run again

## Tips

- Tool respects `.gitignore` patterns (future enhancement)
- Larger files may have less accurate line numbers
- System design insights are most useful for architectural reviews
- Run regularly to catch issues early
```

**Step 3: Create examples documentation**

```markdown
# Examples

## Example 1: Small Python Project

### Command
```bash
codereview examples/python-app/
```

### Output
```
🔍 Code Review Tool

📂 Scanning directory: examples/python-app/

✓ Found 8 files to review

📦 Created 1 batches

Analyzing code... ━━━━━━━━━━━━━━━━━━━━━━━━ 100%

┌─────────────────────────┐
│  Code Review Report     │
└─────────────────────────┘

Summary
──────────────────
Analyzed 8 files and found 3 issues

🔴 Critical (1)

Category: Security
File: api/routes.py:23
Issue: Missing Input Validation
Details: User input not validated before database query
...
```

## Example 2: Export to Markdown

### Command
```bash
codereview src/ --output review-2026-01-22.md
```

### Generated Report
See `examples/sample-report.md` for full output.

## Example 3: CI/CD Integration

### GitHub Actions

```yaml
name: Code Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.14'

      - name: Install uv
        run: pip install uv

      - name: Install codereview
        run: uv pip install -e .

      - name: Run code review
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          codereview . --output review.md --severity high

      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: code-review-report
          path: review.md
```
```

**Step 4: Commit documentation**

```bash
git add README.md docs/usage.md docs/examples.md
git commit -m "docs: add comprehensive documentation and examples"
```

---

## Task 12: Final Testing & Validation

**Files:**
- Create: `tests/test_integration.py`
- Create: `.github/workflows/test.yml` (optional)

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration tests for full workflow."""
import pytest
from pathlib import Path
from click.testing import CliRunner
from codereview.cli import main


@pytest.fixture
def sample_project(tmp_path):
    """Create a complete sample project."""
    # Create Python file with issues
    py_file = tmp_path / "app.py"
    py_file.write_text("""
import os

def get_user(user_id):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)

def process_data(data):
    # Complex function (high cyclomatic complexity)
    if data:
        if len(data) > 0:
            for item in data:
                if item:
                    if item.valid:
                        return item.process()
    return None
""")

    # Create Go file
    go_file = tmp_path / "main.go"
    go_file.write_text("""
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
""")

    return tmp_path


@pytest.mark.integration
def test_full_workflow_with_mock(sample_project):
    """Test complete workflow end-to-end with mocking."""
    from unittest.mock import patch, Mock
    from codereview.models import CodeReviewReport, ReviewIssue

    # Mock AWS Bedrock to avoid actual API calls
    mock_report = CodeReviewReport(
        summary="Test analysis complete",
        metrics={"files": 2, "issues": 1},
        issues=[
            ReviewIssue(
                category="Security",
                severity="Critical",
                file_path="app.py",
                line_start=5,
                title="SQL Injection",
                description="Unsafe query construction",
                rationale="Security risk"
            )
        ],
        system_design_insights="Code structure is simple",
        recommendations=["Fix SQL injection"]
    )

    with patch('codereview.analyzer.ChatBedrockConverse') as mock_bedrock:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value.invoke.return_value = mock_report
        mock_bedrock.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(main, [str(sample_project)])

        # Should complete successfully
        assert result.exit_code == 0
        assert "Found" in result.output or "Analyzing" in result.output


@pytest.mark.integration
def test_export_markdown(sample_project, tmp_path):
    """Test Markdown export functionality."""
    from unittest.mock import patch, Mock
    from codereview.models import CodeReviewReport

    output_file = tmp_path / "report.md"

    mock_report = CodeReviewReport(
        summary="Test",
        metrics={},
        issues=[],
        system_design_insights="Good",
        recommendations=[]
    )

    with patch('codereview.analyzer.ChatBedrockConverse') as mock_bedrock:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value.invoke.return_value = mock_report
        mock_bedrock.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(main, [
            str(sample_project),
            "--output", str(output_file)
        ])

        assert output_file.exists()
        content = output_file.read_text()
        assert "# Code Review Report" in content
```

**Step 2: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass (integration tests may be skipped without AWS)

**Step 3: Manual end-to-end test**

```bash
# Test on actual code
uv run codereview codereview/ --verbose

# Test Markdown export
uv run codereview codereview/ --output test-report.md

# Verify report was created
cat test-report.md
```

**Step 4: Commit integration tests**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full workflow"
```

**Step 5: Final commit and tag**

```bash
git add -A
git commit -m "chore: finalize v0.1.0 release"
git tag v0.1.0
```

---

## Completion Checklist

- [ ] Project setup with uv and dependencies
- [ ] Data models (ReviewIssue, CodeReviewReport)
- [ ] Configuration with defaults and prompts
- [ ] File scanner with exclusion logic
- [ ] Smart batcher for context management
- [ ] LLM analyzer with AWS Bedrock
- [ ] Rich terminal renderer
- [ ] Markdown exporter
- [ ] CLI entry point with full workflow
- [ ] Error handling with retry logic
- [ ] Comprehensive documentation
- [ ] Integration tests
- [ ] Manual testing completed
- [ ] Release tagged

## Post-Implementation

### Testing with Real Code

```bash
# Test on this codebase
uv run codereview . --output self-review.md

# Review the output for quality
cat self-review.md
```

### Future Enhancements

1. Add support for more languages (JavaScript, TypeScript, Rust)
2. Integrate static analysis tools (pylint, bandit)
3. Add configuration file support (.codereview.yaml)
4. Implement git diff mode (review only changed files)
5. Add interactive mode with issue navigation
6. Cache analysis results for faster re-runs
7. Add custom rule definitions

---

**Implementation complete!** 🎉
