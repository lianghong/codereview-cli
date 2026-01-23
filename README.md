# Code Review CLI

> AI-powered code review tool with multiple LLM providers (AWS Bedrock, Azure OpenAI)

A LangChain-based CLI tool that provides comprehensive, intelligent code reviews for Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript projects using Claude, GPT, and other leading models through AWS Bedrock and Azure OpenAI.

## Features

- **Multi-Provider Support**: AWS Bedrock (Claude, Mistral, Minimax, Kimi, Qwen) and Azure OpenAI (GPT models)
- **AI-Powered Analysis**: Leverages Claude Opus 4.5, GPT-5.2 Codex, and other leading models for deep code understanding
- **Multi-Language Support**: Reviews Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript codebases
- **Smart Batching**: Automatically groups files for efficient token usage
- **Structured Output**: Get categorized issues with severity levels and actionable suggestions
- **Static Analysis Integration**: Combine AI review with ruff, mypy, black, eslint, and other tools
- **Terminal UI**: Rich, colorful terminal output with progress indicators
- **Markdown Export**: Generate shareable review reports in Markdown format
- **Error Handling**: Robust retry logic with exponential backoff for API rate limits
- **Flexible Configuration**: Customize file size limits, exclusion patterns, and provider settings

## Installation

### Prerequisites

- Python 3.14+
- **One of the following:**
  - AWS account with Bedrock access (for Claude, Mistral, Minimax, Kimi, Qwen models)
  - Azure OpenAI resource with GPT model deployment (for GPT models)

### Install with uv (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd langchain_projects

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
3. Request access to "Anthropic Claude Opus 4.5"
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
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-4-5*"
    }
  ]
}
```

## Azure OpenAI Configuration (Alternative to AWS)

### 1. Set Environment Variables

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### 2. Create Azure OpenAI Deployment

1. Create an Azure OpenAI resource in Azure Portal
2. Deploy a GPT model (e.g., GPT-4, GPT-3.5-Turbo, or GPT-5.2 Codex if available)
3. Note your deployment name, endpoint, and API key

### 3. Update Configuration

Edit `codereview/config/models.yaml` to add your deployment:

```yaml
providers:
  azure_openai:
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    api_version: "2024-12-01-preview"
    models:
      - id: gpt-5.2-codex
        deployment_name: gpt-5.2-codex  # Your Azure deployment name
        name: GPT-5.2 Codex
        aliases: [gpt52, codex]
        pricing:
          input_per_million: 1.75
          output_per_million: 14.00
      - id: gpt-4o
        deployment_name: gpt-4o  # Your Azure deployment name
        name: GPT-4o
        aliases: [gpt4o]
        pricing:
          input_per_million: 2.50
          output_per_million: 10.00
```

### 4. Test Connection

```bash
codereview --list-models  # Should show Azure models
```

**Note:** Azure OpenAI models require you to deploy them in your Azure resource first. The deployment names in your configuration must match your actual Azure deployments.

## Usage

### Basic Usage

```bash
# Uses Claude Opus 4.5 by default
codereview /path/to/your/codebase
```

### Choose Your Model

```bash
# List all available models
codereview --list-models

# AWS Bedrock Models (Claude family)
codereview /path/to/code --model opus      # Claude Opus 4.5 (highest quality)
codereview /path/to/code --model sonnet    # Claude Sonnet 4.5 (balanced)
codereview /path/to/code --model haiku     # Claude Haiku 4.5 (fastest)

# AWS Bedrock (other providers)
codereview /path/to/code --model minimax   # Minimax M2
codereview /path/to/code --model mistral   # Mistral Large 3
codereview /path/to/code --model kimi      # Kimi K2 Thinking
codereview /path/to/code --model qwen      # Qwen3 Coder 480B

# Azure OpenAI Models
codereview /path/to/code --model gpt-5.2-codex  # GPT-5.2 Codex
codereview /path/to/code --model gpt4o          # GPT-4o

# Short aliases work too
codereview /path/to/code -m haiku
codereview /path/to/code -m qwen
```

**Model Comparison:**

| Model | Provider | Use Case | Input $/M | Output $/M |
|-------|----------|----------|-----------|------------|
| Opus 4.5 | AWS Bedrock | Highest quality, critical reviews | $5.00 | $25.00 |
| Sonnet 4.5 | AWS Bedrock | Balanced performance and cost | $3.00 | $15.00 |
| Haiku 4.5 | AWS Bedrock | Fast, economical, large codebases | $1.00 | $5.00 |
| GPT-5.2 Codex | Azure OpenAI | Code-specialized, Microsoft ecosystem | $1.75 | $14.00 |
| GPT-4o | Azure OpenAI | Multimodal, general purpose | $2.50 | $10.00 |
| Minimax M2 | AWS Bedrock | Cost-effective, good for testing | $0.30 | $1.20 |
| Mistral Large 3 | AWS Bedrock | Open-source focused, multilingual | $2.00 | $6.00 |
| Kimi K2 | AWS Bedrock | Large context window (up to 256K) | $0.50 | $2.00 |
| Qwen3 Coder 480B | AWS Bedrock | Ultra-large model, deep analysis | $0.22 | $1.40 |

### Export to Markdown

```bash
codereview /path/to/code --output review-report.md
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

## Troubleshooting

### Provider Credentials Not Found

```
Error: AWS credentials not found
Error: Azure OpenAI credentials not found
```

**Solutions**:
- **AWS**: Configure credentials using `aws configure` or set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables
- **Azure**: Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` environment variables

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
- Reduce batch size by analyzing fewer files (`--max-files`)
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
git clone <repository-url>
cd langchain_projects

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
langchain_projects/
├── codereview/
│   ├── __init__.py
│   ├── analyzer.py           # LLM-based code analysis
│   ├── batcher.py            # Smart file batching
│   ├── cli.py                # CLI entry point
│   ├── config.py             # Configuration constants
│   ├── models.py             # Pydantic data models for review output
│   ├── renderer.py           # Terminal and Markdown rendering
│   ├── scanner.py            # File system scanning
│   ├── static_analysis.py    # Static analysis tool integration
│   ├── config/
│   │   ├── models.yaml       # Provider and model configuration
│   │   ├── config_models.py  # Pydantic models for configuration
│   │   └── loader.py         # YAML configuration loader
│   └── providers/
│       ├── base.py           # ModelProvider abstract base class
│       ├── factory.py        # Provider factory with auto-detection
│       ├── bedrock.py        # AWS Bedrock provider implementation
│       └── azure_openai.py   # Azure OpenAI provider implementation
├── tests/
│   ├── test_*.py             # Unit tests
│   └── fixtures/             # Test fixtures
├── docs/
│   ├── usage.md              # Detailed usage guide
│   └── examples.md           # Example commands and workflows
├── pyproject.toml            # Project configuration
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
- Comprehensive test coverage (162 tests)

**Static Analysis Tools:**
```bash
# Install development tools
uv pip install ruff mypy black isort vulture types-PyYAML

# Run all checks
uv run ruff check codereview/ tests/
uv run mypy codereview/ --ignore-missing-imports
uv run black --check codereview/ tests/
uv run isort --check-only codereview/ tests/
uv run vulture codereview/ --min-confidence 80

# Auto-fix formatting
uv run black codereview/ tests/
uv run isort codereview/ tests/
uv run ruff check --fix codereview/ tests/
```

**Quality Requirements:**
- All code must pass: ruff (linting), mypy (type checking), black (formatting), isort (import sorting), vulture (dead code)
- All tests must pass (162/162)
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
   - Run static analysis tools (ruff, mypy, black, isort, vulture)
   - See "Code Quality" section above for commands
5. Follow existing code style and architecture patterns
6. Update documentation if adding new features
7. Submit a pull request

## License

[Add your license information here]

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the [Usage Guide](docs/usage.md)
- Review [Examples](docs/examples.md)

## Version History

### v0.2.0 (Current)
- **Multi-Provider Support**: Added Azure OpenAI alongside AWS Bedrock
- **Provider Architecture**: Abstract provider system with factory pattern
- **YAML Configuration**: Centralized model configuration in `models.yaml`
- **Enhanced Model Selection**: Simplified `--model` option with aliases
- **Extended Language Support**: Added Shell Script, C++, Java, JavaScript, and TypeScript
- **Static Analysis Integration**: Parallel execution of language-specific linters and formatters
- **Improved CLI**: Added `--list-models` flag and better error messages

### v0.1.0
- Initial release
- Support for Python and Go
- Claude Opus 4.5 integration via AWS Bedrock
- Smart batching and token management
- Terminal and Markdown output
- Comprehensive error handling
- Full test coverage

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Anthropic Claude](https://www.anthropic.com/) and [OpenAI GPT](https://openai.com/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/) and [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) for model hosting
- Rich library for beautiful terminal output
- Static analysis tools: ruff, mypy, eslint, golangci-lint, shellcheck, and more
