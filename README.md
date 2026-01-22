# Code Review CLI

> AI-powered code review tool using Claude Opus 4.5 via AWS Bedrock

A LangChain-based CLI tool that provides comprehensive, intelligent code reviews for Python and Go projects using Claude Opus 4.5 through AWS Bedrock.

## Features

- **AI-Powered Analysis**: Leverages Claude Opus 4.5 for deep code understanding and insightful reviews
- **Multi-Language Support**: Reviews Python and Go codebases
- **Smart Batching**: Automatically groups files for efficient token usage
- **Structured Output**: Get categorized issues with severity levels and actionable suggestions
- **Terminal UI**: Rich, colorful terminal output with progress indicators
- **Markdown Export**: Generate shareable review reports in Markdown format
- **Error Handling**: Robust retry logic with exponential backoff for AWS rate limits
- **Flexible Configuration**: Customize file size limits, exclusion patterns, and AWS regions

## Installation

### Prerequisites

- Python 3.14+
- AWS account with Bedrock access
- Claude Opus 4.5 model access in AWS Bedrock

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

## Usage

### Basic Usage

```bash
codereview /path/to/your/codebase
```

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

### Specify AWS Region

```bash
codereview /path/to/code --aws-region us-east-1
```

### Verbose Mode

```bash
# Show detailed progress and error traces
codereview /path/to/code --verbose
```

### All Options Combined

```bash
codereview /path/to/code \
  --output report.md \
  --severity medium \
  --max-files 100 \
  --max-file-size 15 \
  --exclude "**/vendor/**" \
  --aws-region us-west-2 \
  --aws-profile production \
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

### AWS Credentials Not Found

```
Error: AWS credentials not found
```

**Solution**: Configure AWS credentials using `aws configure` or set environment variables.

### Access Denied

```
Error: AccessDeniedException
```

**Solutions**:
1. Verify Bedrock access in AWS Console
2. Check IAM permissions include `bedrock:InvokeModel`
3. Ensure Claude Opus 4.5 model access is enabled in your region

### Model Not Available

```
Error: ResourceNotFoundException
```

**Solution**: Claude Opus 4.5 may not be available in your region. Try using `--aws-region us-west-2`.

### Rate Limiting

```
Error: ThrottlingException
```

**Solution**: The tool automatically retries with exponential backoff. If issues persist:
- Reduce batch size by analyzing fewer files (`--max-files`)
- Use smaller file size limit (`--max-file-size`)
- Wait a few minutes before retrying

### No Files Found

```
Warning: No files found to review
```

**Reasons**:
- Directory is empty
- All files are excluded by default patterns
- File size limits are too restrictive

**Solution**: Check exclusion patterns and adjust `--max-file-size` if needed.

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
│   ├── analyzer.py       # LLM-based code analysis
│   ├── batcher.py        # Smart file batching
│   ├── cli.py            # CLI entry point
│   ├── config.py         # Configuration constants
│   ├── models.py         # Pydantic data models
│   ├── renderer.py       # Terminal and Markdown rendering
│   └── scanner.py        # File system scanning
├── tests/
│   ├── test_*.py         # Unit tests
│   └── fixtures/         # Test fixtures
├── docs/
│   ├── usage.md          # Detailed usage guide
│   └── examples.md       # Example commands and workflows
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

### Code Quality

The codebase follows:
- Python 3.14+ modern syntax
- Type hints throughout
- Pydantic for data validation
- Rich for terminal UI
- Click for CLI interface
- Comprehensive test coverage

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the [Usage Guide](docs/usage.md)
- Review [Examples](docs/examples.md)

## Version History

### v0.1.0 (Current)
- Initial release
- Support for Python and Go
- Claude Opus 4.5 integration
- Smart batching and token management
- Terminal and Markdown output
- Comprehensive error handling
- Full test coverage

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- AWS Bedrock for model hosting
- Rich library for beautiful terminal output
