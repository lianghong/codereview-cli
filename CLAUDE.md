# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangChain-based CLI tool for AI-powered code reviews using Claude Opus 4.5 via AWS Bedrock. Reviews Python and Go codebases with structured output (categories, severity levels, line numbers, suggested fixes).

**Tech Stack:** Python 3.14, LangChain, AWS Bedrock, Pydantic V2, Click, Rich

## Development Commands

### Environment Setup
```bash
# Create virtual environment with Python 3.14
uv venv --python 3.14

# Install in editable mode
uv pip install -e .
```

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_analyzer.py -v

# Run single test function
uv run pytest tests/test_models.py::test_review_issue_creation -v

# Run with coverage
uv run pytest tests/ --cov=codereview --cov-report=html
```

### Running the Tool
```bash
# Basic usage (uses Claude Opus 4.5 by default)
uv run codereview /path/to/code

# With model selection
uv run codereview /path/to/code --model-id global.anthropic.claude-sonnet-4-5-20250929-v1:0

# With static analysis (ruff, mypy, black, isort)
uv run codereview /path/to/code --static-analysis

# With options
uv run codereview ./src --output report.md --severity high --verbose

# Direct Python invocation (for debugging)
uv run python -m codereview.cli /path/to/code
```

**Static Analysis Integration:**
```bash
# Install optional tools for static analysis
pip install ruff mypy black isort

# Run combined AI + static analysis review
uv run codereview ./src --static-analysis --output comprehensive-review.md
```

## Architecture

### Pipeline Flow
```
FileScanner → SmartBatcher → CodeAnalyzer → Aggregation → TerminalRenderer/MarkdownExporter
```

1. **FileScanner** (`scanner.py`): Discovers .py/.go files, applies exclusion patterns from `config.py`
2. **SmartBatcher** (`batcher.py`): Groups files into batches (default 10 files/batch) for token efficiency
3. **CodeAnalyzer** (`analyzer.py`): Sends batches to Claude via LangChain with structured output
4. **Aggregation** (`cli.py`): Merges results from all batches into single report
5. **Renderers** (`renderer.py`): Outputs to Rich terminal UI or Markdown file

### Key Architectural Patterns

**LangChain Structured Output Integration:**
- Uses `.with_structured_output(CodeReviewReport)` on ChatBedrockConverse
- Returns Pydantic models directly from LLM (no manual JSON parsing)
- System prompt in `config.py` specifies JSON schema expectations
- See `analyzer.py:34` and `models.py` for the Pydantic → LangChain flow

**Pydantic Data Models:**
- `ReviewIssue`: Single finding (category, severity, file_path, line_start, description, etc.)
- `CodeReviewReport`: Aggregated results (summary, metrics, issues list, recommendations)
- Models have validators (e.g., `line_end >= line_start` check in `models.py`)

**Retry Logic with Exponential Backoff:**
- `CodeAnalyzer.analyze_batch()` has built-in retry for AWS rate limiting
- Handles `ThrottlingException` and `TooManyRequestsException`
- Backoff: 2^attempt × 2 seconds (2s, 4s, 8s)
- See `analyzer.py:36-80` for implementation

### Configuration Constants

All in `codereview/config.py`:
- `DEFAULT_EXCLUDE_PATTERNS`: File patterns to skip (.venv, __pycache__, etc.)
- `DEFAULT_EXCLUDE_EXTENSIONS`: Extensions to skip (.json, .md, binaries, etc.)
- `MODEL_CONFIG`: AWS Bedrock settings (model_id, region, temperature, max_tokens)
- `SUPPORTED_MODELS`: Dict of available models with pricing information
  - Claude Opus 4.5: $15/M input, $75/M output (default)
  - Claude Sonnet 4.5: $3/M input, $15/M output
  - Claude Haiku 4.5: $0.25/M input, $1.25/M output
- `DEFAULT_MODEL_ID`: Default model (Claude Opus 4.5)
- `SYSTEM_PROMPT`: Instructions for Claude including "avoid overdesign" rule
- `MAX_FILE_SIZE_KB`: File size limit (default 10KB)

**Important:** All model IDs use region-agnostic `global.anthropic.*` format

## Testing Patterns

### Mocking AWS Bedrock
All tests mock AWS calls to avoid real API usage:

```python
# Standard pattern in tests
from unittest.mock import Mock, patch

with patch('codereview.analyzer.ChatBedrockConverse') as mock_bedrock:
    mock_instance = Mock()
    mock_instance.with_structured_output.return_value.invoke.return_value = mock_report
    mock_bedrock.return_value = mock_instance

    # Test code here
```

See `tests/test_analyzer.py` and `tests/test_integration.py` for examples.

### Test Fixtures
- `tests/fixtures/sample_code/`: Test files for scanner (.py, .go, .json)
- Fixtures verify exclusion logic (venv, pycache, json files excluded)

### Pydantic Validation Tests
Test both positive cases and `ValidationError` exceptions:
- Invalid categories/severities raise ValidationError
- Line number constraints (ge=1) enforced
- Custom validators (line_end >= line_start) tested

See `tests/test_models.py` for validation patterns.

## AWS Configuration Requirements

**Runtime Prerequisites:**
1. AWS credentials configured (via `aws configure`, env vars, or IAM role)
2. Bedrock access enabled in AWS region
3. Claude Opus 4.5 model access approved
4. IAM permissions: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`

**For Development/Testing:**
- All AWS calls are mocked in tests (no real credentials needed)
- Use `--aws-region` flag to test different regions
- Set `AWS_PROFILE` env var for specific profiles

## Code Modifications

### Adding New Review Categories
1. Update `models.py`: Add to `ReviewIssue.category` Literal
2. Update `config.py`: Add category description to SYSTEM_PROMPT
3. Update `README.md`: Add to "Review Categories" section

### Changing Default Exclusions
Edit `DEFAULT_EXCLUDE_PATTERNS` or `DEFAULT_EXCLUDE_EXTENSIONS` in `config.py`

### Modifying LLM Behavior
Edit `SYSTEM_PROMPT` in `config.py`. Keep JSON output format specification intact for structured output to work.

### Adding Support for New Languages
1. Add extension to `scanner.py:58` target_extensions
2. Update SYSTEM_PROMPT in `config.py` with language-specific guidance
3. Test with fixture files in `tests/fixtures/`

## Common Issues

**Pydantic V1 Compatibility Warning (Python 3.14):**
- LangChain uses Pydantic V1 compat layer
- Shows warning but code works fine
- Documented in `docs/python-compatibility-notes.txt`

**Test Failures Related to Pydantic:**
- Ensure using `pytest.raises(ValidationError)` not `ValueError`
- Import: `from pydantic import ValidationError`

**LangChain Structured Output Not Working:**
- Verify Pydantic model has proper `Field()` descriptions
- Check SYSTEM_PROMPT mentions JSON output format
- Ensure model returned from `.with_structured_output()` is used

**AWS ClientError in Tests:**
- Make sure to mock `ChatBedrockConverse` BEFORE importing analyzer
- Use `patch('codereview.analyzer.ChatBedrockConverse')` not `patch('langchain_aws.ChatBedrockConverse')`
