# Code Review CLI Tool - Design Document

**Date:** 2026-01-22
**Status:** Approved
**Author:** Design Session

## Overview

A LangChain-based command-line tool that performs comprehensive, professional code reviews on Python and Go codebases using Claude Opus 4.5 via AWS Bedrock. The tool provides intelligent analysis with beautiful terminal output and exportable Markdown reports.

## Goals

- Provide professional, high-quality code reviews comparable to senior engineer reviews
- Analyze Python and Go codebases comprehensively
- Output results in a beautiful, easy-to-understand format
- Support both immediate terminal feedback and shareable reports
- Focus on actionable, real issues (avoid nitpicking and overdesign)

## Technical Requirements

- **Python Version:** 3.14
- **Package Manager:** uv
- **Virtual Environment:** `.venv`
- **Model Provider:** AWS Bedrock
- **Model:** Claude Opus 4.5
- **Model ID:** `global.anthropic.claude-opus-4-5-20251101-v1:0`

## Architecture

### High-Level Pipeline

```
┌─────────────────┐
│ File Discovery  │
│   & Filtering   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Smart Batching  │
│ & Context Mgmt  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Analysis   │
│ (Single Pass)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Result       │
│  Aggregation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Rich Terminal   │
│  + Markdown     │
└─────────────────┘
```

### Core Components

```
codereview/
├── cli.py              # Entry point, argument parsing
├── scanner.py          # File discovery and filtering
├── batcher.py          # Smart context window management
├── analyzer.py         # LLM interaction via LangChain
├── models.py           # Pydantic models for structured output
├── renderer.py         # Rich terminal + Markdown rendering
└── config.py           # Configuration and prompts
```

## Data Models

### ReviewIssue

```python
class ReviewIssue(BaseModel):
    category: Literal[
        "Code Style",
        "Code Quality",
        "Security",
        "Performance",
        "Best Practices",
        "System Design",
        "Testing",
        "Documentation"
    ]
    severity: Literal["Critical", "High", "Medium", "Low", "Info"]
    file_path: str
    line_start: int
    line_end: int | None = None
    title: str                    # Brief issue summary
    description: str              # Detailed explanation
    suggested_code: str | None    # Code fix if applicable
    rationale: str                # Why this matters
    references: list[str] = []    # Links to docs/standards
```

### CodeReviewReport

```python
class CodeReviewReport(BaseModel):
    summary: str                          # Executive summary
    metrics: dict                         # Lines analyzed, files reviewed, counts
    issues: list[ReviewIssue]
    system_design_insights: str           # Architectural observations
    recommendations: list[str]            # Top priority actions
```

## LLM Integration Strategy

### Single-Pass Analysis Approach

**Why Single-Pass:**
- Claude Opus 4.5 excels at multi-faceted analysis
- Simpler architecture with fewer moving parts
- Better system design insights from holistic view
- Faster overall (one LLM call per batch vs. multiple)
- More coherent analysis across categories

### Prompt Architecture

```python
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

OUTPUT FORMAT: Return valid JSON matching CodeReviewReport schema."""
```

### Context Management

Each batch sent to Claude includes:
- File contents with line numbers
- File paths and relationships (package/module structure)
- Batch context: "This is batch 2/5 analyzing the authentication module"
- Previous findings summary (if relevant) for consistency

### AWS Bedrock Configuration

```python
MODEL_CONFIG = {
    "model_id": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "region": "us-west-2",  # Configurable
    "max_tokens": 16000,    # For comprehensive responses
    "temperature": 0.1,     # Low for consistent reviews
}
```

**Authentication Priority:**
1. AWS CLI credentials (`~/.aws/credentials`)
2. Environment variables (`AWS_ACCESS_KEY_ID`, etc.)
3. IAM role (if running on EC2/ECS)

## CLI Interface

### Command Structure

```bash
# Basic usage
codereview /path/to/codebase

# With options
codereview /path/to/codebase \
  --output report.md \
  --severity high \
  --exclude "**/tests/*,**/migrations/*" \
  --max-files 100 \
  --aws-region us-east-1
```

### Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `directory` | positional | Path to codebase | (required) |
| `--output, -o` | string | Markdown report path | None |
| `--severity, -s` | choice | Min severity filter | info |
| `--exclude, -e` | string | Glob patterns to exclude | (see defaults) |
| `--max-files` | int | Limit number of files | unlimited |
| `--max-file-size` | int | Max file size in KB | 10 |
| `--category, -c` | string | Filter by category | all |
| `--aws-region` | string | AWS region | us-west-2 |
| `--aws-profile` | string | AWS CLI profile | default |
| `--verbose, -v` | flag | Detailed progress | False |
| `--debug` | flag | Full debug logs | False |

### Built-in Exclusion Rules

```python
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
```

## Output Formats

### Rich Terminal Output

**1. Progress Display:**
- Animated spinner during analysis
- Progress bar showing batches processed
- Current file being analyzed

**2. Results Table:**
- Grouped by severity with color coding:
  - Critical = Red 🔴
  - High = Orange 🟠
  - Medium = Yellow 🟡
  - Low = Blue 🔵
  - Info = Gray ⚪
- Columns: Category | Severity | File:Line | Title

**3. Summary Panel:**
- Total issues by severity
- Files analyzed count
- Top 3 recommendations
- Execution time

### Markdown Report

```markdown
# Code Review Report
**Generated:** 2026-01-22 19:45:32
**Directory:** /path/to/codebase
**Files Analyzed:** 45 Python, 12 Go

## Executive Summary
[AI-generated overview of key findings and health assessment]

## Metrics
- **Total Issues:** 23 (3 Critical, 5 High, 10 Medium, 5 Low)
- **Lines of Code:** ~4,500
- **Review Duration:** 45s

## Issues by Severity

### 🔴 Critical (3)

#### [Security] SQL Injection Vulnerability
**File:** `api/database.py:145-148`
**Category:** Security | **Severity:** Critical

**Description:**
User input directly concatenated into SQL query...

**Suggested Fix:**
```python
# Before
query = f"SELECT * FROM users WHERE id = {user_id}"

# After
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

**Rationale:**
Allows arbitrary SQL execution...

**References:**
- [OWASP SQL Injection](https://owasp.org/...)
- [CWE-89](https://cwe.mitre.org/...)

---

## System Design Insights
[Architectural observations, patterns, improvement areas]

## Top Recommendations
1. Immediately patch SQL injection vulnerabilities
2. Implement input validation layer
3. Consider breaking monolithic handler into services
```

## Error Handling

### AWS Connection Errors
- Graceful failure with clear message
- Suggest credential check
- Retry with exponential backoff (3 attempts)

### Rate Limiting
- Detect throttling errors
- Automatic retry with backoff
- Progress message: "Rate limited, waiting 30s..."

### Token Limit Exceeded
- Split batch into smaller chunks automatically
- Warn about very large files
- Continue with remaining batches

### Malformed LLM Responses
- Retry with clarified prompt (up to 2 times)
- Fall back to partial results
- Log raw response for debugging

### File Access Errors
- Skip unreadable files with warning
- Continue processing remaining files
- Report skipped files in summary

## Quality Assurance

### Validation
- Pydantic models ensure structured output
- Line numbers verified against actual file length
- Required fields checked

### Deduplication
- Hash similar issues across batches
- Merge duplicate findings
- Combine file references for same issue

### Prioritization
- Critical/High severity issues highlighted first
- Security issues always elevated
- System design insights separated from tactical issues

## Implementation Phases

### Phase 1: Core Infrastructure
- Project setup with uv
- CLI argument parsing
- File scanner with exclusion logic
- Basic AWS Bedrock connection

### Phase 2: LLM Integration
- LangChain setup with structured output
- Prompt engineering and testing
- Batch management
- Response parsing

### Phase 3: Output Rendering
- Rich terminal UI implementation
- Markdown report generation
- Issue formatting and presentation

### Phase 4: Polish & Testing
- Error handling refinement
- Performance optimization
- End-to-end testing
- Documentation

## Success Criteria

- ✅ Successfully analyzes Python and Go codebases
- ✅ Provides actionable, professional review feedback
- ✅ Beautiful, easy-to-read terminal output
- ✅ Exportable Markdown reports
- ✅ Handles errors gracefully
- ✅ Reasonable performance (< 1 minute for typical projects)
- ✅ Clear, helpful error messages

## Future Enhancements (Out of Scope)

- Support for additional languages
- Integration with static analysis tools
- CI/CD pipeline integration
- Configuration files for custom rules
- Interactive mode with issue navigation
- Git diff mode (review only changed files)
