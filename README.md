# Code Review CLI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> AI-powered code review tool with multiple LLM providers (AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google Generative AI)

A LangChain-based CLI tool that provides comprehensive, intelligent code reviews for Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript projects using Claude, GPT, Gemini, Devstral, and other leading models through AWS Bedrock, Azure OpenAI, NVIDIA NIM, and Google Generative AI.

## Features

- **Multi-Provider Support**: AWS Bedrock (Claude, Mistral, Minimax, Kimi, Qwen, DeepSeek, GLM), Azure OpenAI (GPT, Kimi K2.5, Grok 4), NVIDIA NIM (Devstral, MiniMax M2.1, Qwen3.5, DeepSeek, GLM 4.7), and Google GenAI (Gemini 3 Pro, Gemini 3 Flash)
- **AI-Powered Analysis**: Leverages Claude Opus 4.6, GPT-5.2 Codex, Grok 4 Fast Reasoning, Gemini 3 Pro, Devstral 2, and other leading models for deep code understanding
- **Multi-Language Support**: Reviews Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript codebases
- **Smart Batching**: Automatically groups files for efficient token usage
- **Structured Output**: Get categorized issues with severity levels and actionable suggestions
- **Static Analysis Integration**: Combine AI review with ruff, mypy, black, eslint, and other tools
- **Architectural Review**: Detects boundary violations, coupling issues, and layering leaks
- **Operational Readiness**: Checks for missing error handling, timeouts, and observability gaps
- **Testing Quality**: Identifies test anti-patterns and coverage gaps
- **Terminal UI**: Rich, colorful terminal output with progress indicators
- **Markdown/JSON Export**: Generate shareable reports in Markdown or JSON format for CI/CD
- **Error Handling**: Robust retry logic with exponential backoff for API rate limits
- **Flexible Configuration**: Customize file size limits, exclusion patterns, and provider settings

## Installation

### Prerequisites

- Python 3.14+
- **One of the following:**
  - AWS account with Bedrock access (for Claude, Mistral, Minimax, Kimi, Qwen models)
  - Azure OpenAI resource with model deployment (for GPT, Kimi K2.5, Grok 4 models)
  - NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) (for Devstral, MiniMax M2.1, Qwen3.5, DeepSeek, free tier available)
  - Google API key from [AI Studio](https://aistudio.google.com/apikey) (for Gemini 3 Pro, Gemini 3 Flash)

### Install with uv (recommended)

```bash
# Clone the repository
git clone https://github.com/lianghong/codereview-cli.git
cd codereview-cli

# Create virtual environment
uv venv --python 3.14

# Install the package
uv pip install -e .
```

### Install with pip

```bash
pip install -e .
```

## AWS Configuration

### 1. Configure AWS Credentials

Choose one of the following methods:

**Option A: AWS CLI**
```bash
aws configure
```

**Option B: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

**Option C: AWS Profile**
```bash
codereview /path/to/code --aws-profile your-profile
```

### 2. Enable Bedrock Access

1. Go to AWS Console > Bedrock
2. Navigate to "Model access" in your region
3. Request access to "Anthropic Claude Opus 4.6"
4. Wait for approval (usually instant for supported regions)

### 3. Verify IAM Permissions

Ensure your IAM user/role has the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-*"
    }
  ]
}
```

## Azure OpenAI Configuration (Alternative to AWS)

Azure OpenAI provides access to GPT-5.2 Codex, Kimi K2.5, and Grok 4 Fast Reasoning via Microsoft Azure AI Foundry.

### 1. Set Environment Variables

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### 2. Deploy Models in Azure AI Foundry

1. Create an Azure OpenAI resource in Azure Portal
2. Deploy models from Azure AI Foundry catalog:
   - **GPT-5.2 Codex** (deployment name: `gpt-5.2-codex`)
   - **Kimi K2.5** (deployment name: `Kimi-K2.5`) - Moonshot AI's multimodal MoE model
   - **Grok 4 Fast Reasoning** (deployment name: `grok-4-fast-reasoning`) - xAI's cost-efficient reasoning model
3. Note your deployment name, endpoint, and API key

### 3. Use Azure Models

```bash
# GPT-5.2 Codex - Code-specialized
codereview /path/to/code --model gpt

# Kimi K2.5 - Multimodal MoE, 256K context
codereview /path/to/code --model kimi-azure

# Grok 4 Fast Reasoning - 2M context, cost-efficient
codereview /path/to/code --model grok
```

### 4. Test Connection

```bash
codereview --list-models  # Should show Azure models
```

**Note:** Azure OpenAI models require you to deploy them in your Azure resource first. The deployment names in your configuration must match your actual Azure deployments. Kimi K2.5 and Grok 4 Fast Reasoning are available as "Direct from Azure" models in the Azure AI Foundry catalog.

## NVIDIA NIM Configuration (Alternative Provider)

NVIDIA NIM provides access to models like Devstral 2, MiniMax M2.1, Qwen3.5, DeepSeek V3.2, Qwen3 Coder, and more with a free tier for development.

### 1. Get API Key

1. Visit [NVIDIA Build](https://build.nvidia.com/explore/discover)
2. Sign in or create an account
3. Generate an API key (format: `nvapi-xxxxx...`)

### 2. Set Environment Variable

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

### 3. Use NVIDIA Models

```bash
# Devstral 2 - Code-specialized model
codereview /path/to/code --model devstral

# DeepSeek V3.2 - Large reasoning model with thinking mode
codereview /path/to/code --model deepseek-nvidia

# Qwen3 Coder - Ultra-large coding model with thinking mode
codereview /path/to/code --model qwen-nvidia

# Qwen3.5 - Next-gen Qwen reasoning model with thinking mode (262K context)
codereview /path/to/code --model qwen3.5

# GLM 4.7 - Reasoning model with interleaved thinking (73.8% SWE-bench)
codereview /path/to/code --model glm47

# MiniMax M2.1 - Enhanced reasoning model with thinking mode
codereview /path/to/code --model minimax-m2.1

# Kimi K2.5 - Latest Kimi model with 256K context
codereview /path/to/code --model kimi-k2.5
```

**Note:** NVIDIA NIM models are currently in free tier. No charges apply during the preview period. Models with thinking mode enabled (MiniMax M2.1, Qwen3.5, DeepSeek, Qwen3 Coder, GLM 4.7) provide deeper reasoning for complex code analysis.

## Google Generative AI Configuration (Alternative Provider)

Google Generative AI provides access to Gemini 3 models with 1M token context windows.

### 1. Get API Key

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Create an API key

### 2. Set Environment Variable

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 3. Use Gemini Models

```bash
# Gemini 3 Pro - Flagship reasoning model (1M context)
codereview /path/to/code --model gemini-3-pro

# Gemini 3 Flash - Fast and cost-efficient (1M context)
codereview /path/to/code --model gemini-3-flash
```

## Usage

### Basic Usage

```bash
# Uses Claude Opus 4.6 by default
codereview /path/to/your/codebase
```

### Choose Your Model

```bash
# List all available models
codereview --list-models

# AWS Bedrock Models (Claude family)
codereview /path/to/code --model opus      # Claude Opus 4.6 (highest quality)
codereview /path/to/code --model sonnet    # Claude Sonnet 4.6 (balanced)
codereview /path/to/code --model haiku     # Claude Haiku 4.5 (fastest)

# AWS Bedrock (other providers)
codereview /path/to/code --model kimi-k2.5-bedrock  # Kimi K2.5 (262K context)
codereview /path/to/code --model qwen-bedrock       # Qwen3 Coder 480B
codereview /path/to/code --model qwen-next-bedrock  # Qwen3 Coder Next (80B MoE)
codereview /path/to/code --model deepseek-r1-bedrock  # DeepSeek-R1 (reasoning)
codereview /path/to/code --model deepseek-v3.2-bedrock # DeepSeek V3.2 (agentic)
codereview /path/to/code --model glm47-bedrock      # GLM 4.7 (thinking mode)
codereview /path/to/code --model glm47-flash-bedrock # GLM 4.7 Flash (cost-efficient)

# Azure OpenAI Models
codereview /path/to/code --model gpt-5.2-codex  # GPT-5.2 Codex
codereview /path/to/code --model kimi-azure      # Kimi K2.5 (256K context)
codereview /path/to/code --model grok            # Grok 4 Fast Reasoning (2M context)

# NVIDIA NIM Models (free tier)
codereview /path/to/code --model devstral           # Devstral 2 123B
codereview /path/to/code --model minimax-m2.1         # MiniMax M2.1 (thinking mode)
codereview /path/to/code --model deepseek-v3.2-nvidia # DeepSeek V3.2 (thinking mode)
codereview /path/to/code --model qwen-nvidia        # Qwen3 Coder 480B (thinking mode)
codereview /path/to/code --model qwen3.5            # Qwen3.5 397B (thinking mode, 262K context)
codereview /path/to/code --model glm47              # GLM 4.7 (thinking mode)
codereview /path/to/code --model kimi-k2.5          # Kimi K2.5 (256K context)

# Google Generative AI Models
codereview /path/to/code --model gemini-3-pro       # Gemini 3 Pro (1M context)
codereview /path/to/code --model gemini-3-flash     # Gemini 3 Flash (fast, cheap)

# Short aliases work too
codereview /path/to/code -m haiku
codereview /path/to/code -m devstral
```

**Model Comparison:**

| Model | Provider | Use Case | Input $/M | Output $/M |
|-------|----------|----------|-----------|------------|
| Opus 4.6 | AWS Bedrock | Highest quality, critical reviews | $5.00 | $25.00 |
| Sonnet 4.6 | AWS Bedrock | Balanced performance and cost | $3.00 | $15.00 |
| Haiku 4.5 | AWS Bedrock | Fast, economical, large codebases | $1.00 | $5.00 |
| GPT-5.2 Codex | Azure OpenAI | Code-specialized, Microsoft ecosystem | $1.75 | $14.00 |
| Kimi K2.5 (Azure) | Azure OpenAI | Multimodal MoE, 256K context | $0.60 | $3.00 |
| Grok 4 Fast (Azure) | Azure OpenAI | 2M context, cost-efficient reasoning | $0.20 | $0.50 |
| Devstral 2 | NVIDIA NIM | Code-specialized, free tier | Free* | Free* |
| MiniMax M2.1 | NVIDIA NIM | 200K context, 128K output, thinking mode | Free* | Free* |
| DeepSeek V3.2 | NVIDIA NIM | Large reasoning model, thinking mode | Free* | Free* |
| Qwen3 Coder (NIM) | NVIDIA NIM | Ultra-large coding, thinking mode | Free* | Free* |
| Qwen3.5 397B | NVIDIA NIM | Next-gen Qwen, thinking mode, 262K context | Free* | Free* |
| GLM 4.7 | NVIDIA NIM | 73.8% SWE-bench, thinking mode | Free* | Free* |
| Kimi K2.5 | NVIDIA NIM | 256K context, instant/thinking modes | Free* | Free* |
| Gemini 3 Pro | Google GenAI | Flagship reasoning, 1M context | $2.00 | $12.00 |
| Gemini 3 Flash | Google GenAI | Fast and cheap, 1M context | $0.50 | $3.00 |
| DeepSeek-R1 | AWS Bedrock | Reasoning model, 128K context | $1.35 | $5.40 |
| DeepSeek V3.2 (Bedrock) | AWS Bedrock | Agentic workflows, tool calling | $0.62 | $1.85 |
| MiniMax M2.1 (Bedrock) | AWS Bedrock | Multilingual coding, 128K output | $0.30 | $1.20 |
| Kimi K2.5 (Bedrock) | AWS Bedrock | Multimodal MoE, 262K context | $0.60 | $3.00 |
| Qwen3 Coder (Bedrock) | AWS Bedrock | Ultra-large model, deep analysis | $0.22 | $1.40 |
| Qwen3 Coder Next (Bedrock) | AWS Bedrock | Ultra-sparse MoE, 70%+ SWE-bench | $0.50 | $1.20 |
| GLM 4.7 (Bedrock) | AWS Bedrock | 73.8% SWE-bench, thinking mode | TBD* | TBD* |
| GLM 4.7 Flash (Bedrock) | AWS Bedrock | Lightweight MoE, cost-efficient | TBD* | TBD* |

*NVIDIA NIM models are currently in free preview tier. Models with thinking mode use interleaved reasoning for deeper code analysis. GLM Bedrock pricing TBD - update when AWS publishes official pricing.

### Export Reports

```bash
# Export to Markdown (default)
codereview /path/to/code --output review-report.md

# Export to JSON for CI/CD pipelines
codereview /path/to/code --output review-report.json --format json
```

### Filter by Severity

```bash
# Show only critical and high severity issues
codereview /path/to/code --severity high
```

### Limit Files

```bash
# Analyze only first 50 files
codereview /path/to/code --max-files 50
```

### Custom File Size Limit

```bash
# Only analyze files under 20KB
codereview /path/to/code --max-file-size 20
```

### Exclude Patterns

```bash
# Exclude test files and specific directories
codereview /path/to/code --exclude "**/tests/**" --exclude "**/deprecated/**"
```

### Run Static Analysis

Combine AI review with static analysis tools (runs in parallel for speed):

```bash
# Run with all available static analysis tools
codereview /path/to/code --static-analysis

# Combine with specific model
codereview /path/to/code --model sonnet --static-analysis --output comprehensive-review.md
```

**Supported Static Analysis Tools:**
- **Python:** ruff (linter), mypy (type checker), black (formatter), isort (import sorter), vulture (dead code finder)
- **Go:** golangci-lint (meta-linter), go vet (static analyzer), gofmt (formatter)
- **Shell:** shellcheck (static analyzer for shell scripts)
- **C++:** clang-tidy (linter), cppcheck (static analysis), clang-format (formatter)
- **Java:** checkstyle (style checker)
- **JavaScript/TypeScript:** eslint (linter), prettier (formatter), tsc (TypeScript type checker)

**Output includes:**
- Tool pass/fail status
- Issue counts per tool
- Detailed output for failed checks
- Integrated into Markdown reports

**Note:** Only installed tools are run. Tools run in parallel using ThreadPoolExecutor for faster execution.

### Verbose Mode

```bash
# Show detailed progress and error traces
codereview /path/to/code --verbose
```

### All Options Combined

```bash
codereview /path/to/code \
  --model sonnet \
  --output report.md \
  --severity medium \
  --max-files 100 \
  --max-file-size 15 \
  --exclude "**/vendor/**" \
  --static-analysis \
  --verbose
```

## Review Categories

The tool identifies issues across 8 categories:

1. **Code Style**: Formatting, naming conventions, code organization
2. **Code Quality**: Complexity, duplication, maintainability
3. **Security**: Vulnerabilities, injection risks, data exposure
4. **Performance**: Inefficiencies, resource usage, optimization opportunities
5. **Best Practices**: Language idioms, design patterns, modern approaches
6. **System Design**: Architecture, modularity, scalability
7. **Testing**: Test coverage, test quality, missing tests
8. **Documentation**: Missing docs, unclear comments, API documentation

## Severity Levels

- **Critical**: Security vulnerabilities, data corruption risks, production blockers
- **High**: Major bugs, performance issues, important best practice violations
- **Medium**: Code quality issues, moderate technical debt, maintenance concerns
- **Low**: Minor improvements, style inconsistencies, nice-to-haves
- **Info**: Suggestions, alternative approaches, educational insights

## Output Format

### Terminal Output

The tool displays:
- File scanning progress
- Batch analysis progress
- Categorized issues with severity badges
- System design insights
- Priority recommendations
- Overall metrics summary

### Markdown Export

Generated reports include:
- Executive summary
- Metrics overview (files analyzed, total issues by severity)
- Detailed issue list with:
  - File paths and line numbers
  - Category and severity
  - Description and rationale
  - Suggested fixes (when applicable)
  - Reference links
- System design insights
- Top recommendations

### JSON Export

For CI/CD integration, use `--format json`:

```bash
codereview ./src --output report.json --format json
```

JSON output includes the full `CodeReviewReport` structure for programmatic consumption:
- Parse issues by severity for quality gates
- Integrate with dashboards and monitoring
- Automate notifications based on findings

## Troubleshooting

### Provider Credentials Not Found

```
Error: AWS credentials not found
Error: Azure OpenAI credentials not found
Error: Google API key not configured
```

**Solutions**:
- **AWS**: Configure credentials using `aws configure` or set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables
- **Azure**: Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` environment variables
- **Google**: Set `GOOGLE_API_KEY` environment variable (get from https://aistudio.google.com/apikey)

### Access Denied

```
Error: AccessDeniedException (AWS)
Error: 401 Unauthorized (Azure)
```

**Solutions**:
- **AWS**: Verify Bedrock access in AWS Console, check IAM permissions include `bedrock:InvokeModel`
- **Azure**: Verify API key is correct and resource is active in Azure Portal

### Model Not Available

```
Error: ResourceNotFoundException (AWS)
Error: DeploymentNotFound (Azure)
```

**Solutions**:
- **AWS**: Model may not be available in your region. Request access in AWS Bedrock Console
- **Azure**: Ensure you have deployed the model in your Azure OpenAI resource. Check deployment name matches configuration

### Rate Limiting

```
Error: ThrottlingException (AWS)
Error: 429 Too Many Requests (Azure)
```

**Solution**: The tool automatically retries with exponential backoff. If issues persist:
- Reduce batch size with `--batch-size 5` (fewer files per API call)
- Reduce total files with `--max-files`
- Use smaller file size limit (`--max-file-size`)
- Wait a few minutes before retrying
- Consider using a different model with higher rate limits

### No Files Found

```
Warning: No files found to review
```

**Reasons**:
- Directory is empty
- All files are excluded by default patterns
- File size limits are too restrictive

**Solution**: Check exclusion patterns and adjust `--max-file-size` if needed.

### Configuration File Not Found

```
Error: models.yaml not found
```

**Solution**: Ensure `codereview/config/models.yaml` exists. If using a custom configuration location, verify the path is correct.

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/lianghong/codereview-cli.git
cd codereview-cli

# Create virtual environment
uv venv --python 3.14

# Install in development mode with dependencies
uv pip install -e .
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_analyzer.py -v

# Run with coverage
uv run pytest tests/ --cov=codereview --cov-report=html
```

### Project Structure

```
codereview-cli/
├── codereview/
│   ├── __init__.py
│   ├── analyzer.py           # LLM-based code analysis
│   ├── batcher.py            # Smart file batching
│   ├── cli.py                # CLI entry point
│   ├── models.py             # Pydantic data models for review output
│   ├── renderer.py           # Terminal and Markdown rendering
│   ├── scanner.py            # File system scanning
│   ├── static_analysis.py    # Static analysis tool integration
│   ├── config/
│   │   ├── __init__.py       # Configuration exports
│   │   ├── models.yaml       # Provider and model configuration
│   │   ├── models.py         # Pydantic models for configuration
│   │   ├── prompts.py        # Code review rules and system prompt
│   │   └── loader.py         # YAML configuration loader
│   └── providers/
│       ├── __init__.py
│       ├── base.py           # ModelProvider abstract base class
│       ├── factory.py        # Provider factory with auto-detection
│       ├── bedrock.py        # AWS Bedrock provider implementation
│       ├── azure_openai.py   # Azure OpenAI provider implementation
│       ├── nvidia.py         # NVIDIA NIM provider implementation
│       └── google_genai.py   # Google GenAI provider implementation
├── tests/
│   ├── test_*.py             # Unit tests (301 tests)
│   └── fixtures/             # Test fixtures
├── docs/
│   ├── usage.md              # Detailed usage guide
│   ├── examples.md           # Example commands and workflows
│   ├── static-analysis.md    # Static analysis tool reference
│   └── MIGRATION.md          # Migration guide
├── pyproject.toml            # Project configuration
├── LICENSE                   # MIT License
├── CLAUDE.md                 # Claude Code instructions
└── README.md                 # This file
```

### Code Quality

The codebase follows strict quality standards:

**Code Standards:**
- Python 3.14+ modern syntax
- Type hints throughout
- Pydantic V2 for data validation
- Rich for terminal UI
- Click for CLI interface
- Comprehensive test coverage (301 tests)

**Static Analysis Tools:**
```bash
# Install development tools
uv pip install ruff mypy isort vulture types-PyYAML

# Run all checks
uv run ruff check codereview/ tests/
uv run ruff format --check codereview/ tests/
uv run mypy codereview/ --ignore-missing-imports
uv run isort --check-only codereview/ tests/
uv run vulture codereview/ --min-confidence 80

# Auto-fix formatting
uv run ruff format codereview/ tests/
uv run isort codereview/ tests/
uv run ruff check --fix codereview/ tests/
```

**Quality Requirements:**
- All code must pass: ruff (linting + formatting), mypy (type checking), isort (import sorting), vulture (dead code)
- All tests must pass (281/281)
- Type hints required for public APIs
- No unused imports or variables
- Provider implementations must include `get_pricing()` method

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. **Ensure code quality checks pass:**
   - Run `uv run pytest tests/ -v` (all tests must pass)
   - Run static analysis tools (ruff, mypy, isort, vulture)
   - See "Code Quality" section above for commands
5. Follow existing code style and architecture patterns
6. Update documentation if adding new features
7. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

For issues, questions, or contributions:
- [Open an issue](https://github.com/lianghong/codereview-cli/issues) on GitHub
- Check the [Usage Guide](docs/usage.md)
- Review [Examples](docs/examples.md)

## Version History

### v0.2.7 (Current)
- **System Prompt Rewrite**: Restructured SYSTEM_PROMPT (~513 to ~310 lines, 40% reduction) for better instruction adherence — critical constraints at top, CWE references, prompt injection defense, self-verification, concrete examples
- **Category-Based Recommendations**: Enhanced report recommendations with category-aware suggestions (Security, Performance, Testing, Code Quality, System Design)
- **NVIDIA GLM-5 Fix**: Graceful fallback when `include_raw=True` is not supported by a provider
- **Non-Interactive Mode**: CI/CD-safe README finder with `sys.stdin.isatty()` guard
- **Env Var Expansion Fix**: Regex now supports digits in variable names (`API_V2_KEY`)
- **Empty Batch Guard**: Early return when no files are readable in a batch
- **Logging Best Practices**: Converted f-string logging to %-style format
- **Test Suite**: Expanded to 301 tests

### v0.2.6
- **Google Generative AI Provider**: Added Google GenAI as 4th provider with Gemini 3 Pro Preview and Gemini 3 Flash Preview models (1M token context)
- **Code Quality Fixes**: Fixed redundant retry logic, added severity guard in renderer, optimized ESLint file detection, removed dead code, improved logging format
- **Test Suite**: Expanded to 281 tests with Google GenAI provider tests

### v0.2.5
- **Claude Opus 4.6**: Added Claude Opus 4.6 as new default model (128K max output)
- **MiniMax M2.1 Model**: Added MiniMax M2.1 via NVIDIA NIM (200K context, 128K output, thinking mode)
- **Export Error Handling**: Report export (JSON/Markdown) now handles file I/O errors gracefully instead of crashing with unhandled exceptions
- **Callbacks Cleanup**: Fixed potential Rich `Live` display errors by using consistent `cleanup()` method
- **Retry Backoff Cap**: Exponential backoff capped at 60 seconds across all providers (Bedrock, Azure OpenAI)
- **Renderer Optimization**: Eliminated redundant string splitting in static analysis output rendering

### v0.2.4
- **JSON Output Format**: New `--format json` option for CI/CD integration and programmatic consumption
- **Enhanced Code Review Prompts**:
  - Architecture fit analysis (boundary violations, coupling, layering leaks)
  - Operational readiness checks (error handling, timeouts, observability)
  - Testing quality guidelines (anti-patterns, coverage gaps)
  - Project conventions detection (reduces false positives for consistent codebases)
  - PII exposure detection in logs and responses
- **Improved Configuration**: ValidationError handling for provider configs prevents crashes on invalid URLs
- **Documentation**: Improved ReviewMetrics docstring explaining field aliasing pattern

### v0.2.3
- **Auto-confirm README Context**: README prompt auto-confirms after 3 seconds with "Y" default
- **Improved UI**: Removed left/right borders from Improvement Suggestions panel for easier copy
- **Exponential Backoff Cap**: Limited retry wait time to 60 seconds maximum
- **ASCII-safe Fallback**: Terminal fallback uses ASCII characters for better compatibility
- **Performance Optimization**: Cached repeated string operations in static analysis parsing
- **Code Quality**: Improved exception handling, docstrings, and type hints throughout codebase
- **Test Coverage**: Expanded test suite to 261 tests

### v0.2.2
- **Kimi K2.5 Model**: Added Moonshot AI's Kimi K2.5 via NVIDIA NIM (256K context window)
- **DeepSeek-R1 Model**: Added DeepSeek-R1 reasoning model via AWS Bedrock (128K context)
- **Model Version IDs**: Updated model IDs to include version numbers (deepseek-r1-bedrock, deepseek-v3.2-nvidia)
- **Non-Tool-Use Support**: Added `supports_tool_use` config for models without function calling (uses prompt-based JSON parsing fallback)
- **Static Analysis Output**: Full error output for failed tools, improved truncation for passed tools
- **Code Quality**: Fixed type annotations, moved imports to module level, improved code organization

### v0.2.1
- **GLM 4.7 Model**: Added Zhipu AI's GLM 4.7 via NVIDIA NIM (73.8% SWE-bench score)
- **Thinking Mode Support**: Enabled interleaved thinking for DeepSeek V3.2, Qwen3 Coder, and GLM 4.7
- **Batch Size Control**: Added `--batch-size` option to control files per batch (helps with timeout issues)
- **Gateway Timeout Handling**: Improved retry logic for NVIDIA 504/502/503 errors with exponential backoff
- **Configurable Polling Timeout**: NVIDIA provider now supports 15-minute polling timeout for large models

### v0.2.0
- **Multi-Provider Support**: Added Azure OpenAI and NVIDIA NIM alongside AWS Bedrock
- **NVIDIA NIM Integration**: Free tier access to Devstral 2, DeepSeek V3.2, Qwen3 Coder, Kimi K2
- **Provider Architecture**: Abstract provider system with factory pattern
- **YAML Configuration**: Centralized model configuration in `models.yaml`
- **Enhanced Model Selection**: Simplified `--model` option with aliases
- **Extended Language Support**: Added Shell Script, C++, Java, JavaScript, and TypeScript
- **Static Analysis Integration**: Parallel execution of language-specific linters and formatters
- **Security Scanning**: Added bandit (Python) and gosec (Go) for security vulnerability detection
- **Improved CLI**: Added `--list-models`, `--dry-run` flags and better error messages

### v0.1.0
- Initial release
- Support for Python and Go
- Claude integration via AWS Bedrock
- Smart batching and token management
- Terminal and Markdown output
- Comprehensive error handling
- Full test coverage

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Anthropic Claude](https://www.anthropic.com/), [OpenAI GPT](https://openai.com/), [Google Gemini](https://ai.google.dev/), [xAI Grok](https://x.ai/), and [Mistral AI](https://mistral.ai/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service), [NVIDIA NIM](https://build.nvidia.com/), and [Google AI Studio](https://aistudio.google.com/) for model hosting
- Rich library for beautiful terminal output
- Static analysis tools: ruff, mypy, eslint, golangci-lint, shellcheck, bandit, gosec, and more
