# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangChain-based CLI tool for AI-powered code reviews via AWS Bedrock, Azure OpenAI, and NVIDIA NIM. Supports multiple models including Claude (Opus, Sonnet, Haiku), GPT-5.2 Codex, Devstral 2, Minimax M2, Mistral Large 3, Kimi K2 Thinking, and Qwen3 Coder. Reviews **Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript** codebases with structured output (categories, severity levels, line numbers, suggested fixes).

**Tech Stack:** Python 3.14, LangChain, AWS Bedrock, Azure OpenAI, NVIDIA NIM, Pydantic V2, Click, Rich

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

### Code Quality & Static Analysis
```bash
# Install development dependencies (if not already installed)
uv pip install ruff mypy black isort vulture types-PyYAML

# Run all static analysis tools
uv run ruff check codereview/ tests/        # Linting
uv run mypy codereview/ --ignore-missing-imports  # Type checking
uv run black codereview/ tests/             # Code formatting
uv run isort codereview/ tests/             # Import sorting
uv run vulture codereview/ --min-confidence 80  # Dead code detection

# Auto-fix issues where possible
uv run ruff check --fix codereview/ tests/
uv run black codereview/ tests/
uv run isort codereview/ tests/

# Verify all tools pass (run before committing)
uv run ruff check codereview/ tests/ && \
uv run black --check codereview/ tests/ && \
uv run isort --check-only codereview/ tests/ && \
uv run mypy codereview/ --ignore-missing-imports && \
uv run vulture codereview/ --min-confidence 80 && \
echo "✓ All static analysis checks passed"
```

**Quality Standards:**
- All code must pass ruff, black, isort, mypy, and vulture checks
- Type hints required for all public APIs
- Minimum 80% confidence for vulture (dead code detection)
- Unused imports/variables must be removed
- All provider implementations must include `get_pricing()` method

### Running the Tool
```bash
# Basic usage (uses Claude Opus 4.5 by default)
uv run codereview /path/to/code

# With model selection (use short names!)
uv run codereview /path/to/code --model sonnet
uv run codereview /path/to/code -m haiku
uv run codereview /path/to/code -m gpt  # Azure OpenAI
uv run codereview /path/to/code -m devstral  # NVIDIA NIM
uv run codereview /path/to/code -m qwen
uv run codereview /path/to/code -m mistral

# With static analysis (runs tools in parallel for speed)
uv run codereview /path/to/code --static-analysis

# With severity filtering (show only high and above)
uv run codereview ./src --severity high

# Dry run (preview files and estimated cost without API calls)
uv run codereview ./src --dry-run

# With all options
uv run codereview ./src -m sonnet --output report.md --severity medium --verbose

# Direct Python invocation (for debugging)
uv run python -m codereview.cli /path/to/code
```

### CLI Options
| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Model to use (see Model Names below) | opus |
| `--output, -o` | Export report to Markdown file | None |
| `--severity, -s` | Minimum severity to display (critical/high/medium/low/info) | info |
| `--temperature` | Model temperature (0.0-2.0) | Model-specific |
| `--static-analysis` | Run static analysis tools (parallel) | False |
| `--dry-run` | Preview files and cost without API calls | False |
| `--verbose, -v` | Show detailed progress | False |
| `--exclude, -e` | Additional exclusion patterns | None |
| `--max-files` | Maximum files to analyze | None |
| `--max-file-size` | Maximum file size in KB | 500 |
| `--aws-profile` | AWS CLI profile to use | None |
| `--list-models` | List all available models and exit | - |

### Model Names
Use primary model IDs (case-insensitive). Run `codereview --list-models` to see all available models.

| Model ID | Name | Provider | Aliases |
|----------|------|----------|---------|
| `opus` | Claude Opus 4.5 | bedrock | claude-opus |
| `sonnet` | Claude Sonnet 4.5 | bedrock | claude-sonnet |
| `haiku` | Claude Haiku 4.5 | bedrock | claude-haiku |
| `gpt-5.2-codex` | GPT-5.2 Codex | azure_openai | gpt, gpt52, codex |
| `devstral` | Devstral 2 123B | nvidia | devstral-2 |
| `minimax-bedrock` | Minimax M2 (Bedrock) | bedrock | mm2-bedrock |
| `minimax-nvidia` | MiniMax M2 (NVIDIA) | nvidia | mm2-nvidia |
| `qwen-nvidia` | Qwen3 Coder 480B (NVIDIA) | nvidia | qwen3-nvidia, qwen-coder-nvidia |
| `kimi-nvidia` | Kimi K2 Instruct (NVIDIA) | nvidia | kimi-k2-nvidia |
| `deepseek-nvidia` | DeepSeek V3.2 (NVIDIA) | nvidia | deepseek-v3-nvidia, ds-nvidia |
| `mistral` | Mistral Large 3 | bedrock | mistral-large |
| `kimi` | Kimi K2 Thinking | bedrock | kimi-k2 |
| `qwen` | Qwen3 Coder 480B | bedrock | qwen-coder |

**Note:** All models are displayed in `--list-models` regardless of provider credentials. Credentials are only required when actually using a model.

### Static Analysis Integration
```bash
# Install Python static analysis tools (including security scanner)
pip install ruff mypy black isort vulture bandit

# Install Go static analysis tools (requires Go installed)
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install github.com/securego/gosec/v2/cmd/gosec@latest

# Install Shell script static analysis (Ubuntu/Debian)
sudo apt install shellcheck
pip install bashate
# Or on macOS: brew install shellcheck && pip install bashate

# Install C++ static analysis tools (Ubuntu/Debian)
sudo apt install clang-tidy cppcheck clang-format
# Or on macOS: brew install llvm cppcheck

# Install Java static analysis (requires Java)
# Download checkstyle JAR from https://checkstyle.org/

# Install JavaScript/TypeScript static analysis
npm install -g eslint prettier typescript

# Run combined AI + static analysis review
uv run codereview ./src --static-analysis --output comprehensive-review.md
```

**Supported Static Analysis Tools:**
- **Python:** ruff (linter), mypy (type checker), black (formatter), isort (import sorter), vulture (dead code finder), bandit (security scanner)
- **Go:** golangci-lint (meta-linter), go vet (static analyzer), gofmt (formatter), gosec (security scanner)
- **Shell:** shellcheck (static analyzer), bashate (style checker)
- **C++:** clang-tidy (linter), cppcheck (static analysis), clang-format (formatter)
- **Java:** checkstyle (style checker)
- **JavaScript/TypeScript:** eslint (linter), prettier (formatter), tsc (TypeScript type checker), npm audit (security scanner)

**Note:** Static analysis tools run in parallel using `ThreadPoolExecutor` for faster execution. Only installed tools are run. See `docs/static-analysis.md` for detailed tool documentation.

## Configuration System

The tool is **fully configurable via YAML files** - no code changes required for most customizations.

### Configuration File Location
```
codereview/config/
├── models.yaml      # All models, providers, pricing, inference params, scanning rules
├── models.py        # Pydantic validation (rarely needs modification)
├── loader.py        # YAML parser with env var expansion
└── prompts.py       # AI code review rules and system prompt
```

### What You Can Configure Without Code Changes

| Configuration | File | Example |
|---------------|------|---------|
| Add new models | `models.yaml` | Add entry under provider's `models:` list |
| Model pricing | `models.yaml` | Update `pricing.input_per_million` |
| Inference parameters | `models.yaml` | Set `temperature`, `top_p`, `max_output_tokens` |
| Default AWS region | `models.yaml` | Change `bedrock.region` |
| File exclusion patterns | `models.yaml` | Add to `scanning.exclude_patterns` |
| Excluded file extensions | `models.yaml` | Add to `scanning.exclude_extensions` |
| Max file size | `models.yaml` | Change `scanning.max_file_size_kb` |
| API credentials | Environment variables | `AZURE_OPENAI_API_KEY`, `NVIDIA_API_KEY` |
| Code review rules | `prompts.py` | Modify `SYSTEM_PROMPT` |

### Environment Variable Expansion

Secrets are configured via environment variables with `${VAR_NAME}` syntax:

```yaml
# models.yaml
azure_openai:
  endpoint: "${AZURE_OPENAI_ENDPOINT}"
  api_key: "${AZURE_OPENAI_API_KEY}"
  api_version: "2025-04-01-preview"

nvidia:
  api_key: "${NVIDIA_API_KEY}"
```

### Adding a New Model (No Code Changes)

Simply add to `models.yaml`:
```yaml
bedrock:
  models:
    - id: my-new-model
      full_id: vendor.model-name-v1
      name: My New Model
      aliases: [mnm, new-model]
      pricing:
        input_per_million: 1.00
        output_per_million: 5.00
      inference_params:
        default_temperature: 0.1
        max_output_tokens: 8192
```

Then use immediately: `codereview ./src --model my-new-model`

### Customizing File Scanning

Edit `scanning` section in `models.yaml`:
```yaml
scanning:
  max_file_size_kb: 500        # Skip files larger than this
  warn_file_size_kb: 100       # Warn about files larger than this
  exclude_patterns:
    - "**/node_modules/**"     # Skip directories
    - "**/generated/**"        # Skip generated code
    - "**/*.min.js"            # Skip minified files
  exclude_extensions:
    - ".json"
    - ".lock"
    - ".svg"
```

### Modifying Code Review Behavior

Edit `SYSTEM_PROMPT` in `config/prompts.py` to:
- Add new severity rules
- Include/exclude specific issue types
- Add language-specific guidelines
- Customize false positive prevention

## Architecture

### Pipeline Flow
```
FileScanner → FileBatcher → CodeAnalyzer → ProviderFactory → BedrockProvider/AzureOpenAIProvider/NVIDIAProvider → Aggregation → TerminalRenderer/MarkdownExporter
```

1. **FileScanner** (`scanner.py`): Discovers code files (.py, .go, .sh, .bash, .cpp, .cc, .cxx, .h, .hpp, .java, .js, .jsx, .mjs, .ts, .tsx), applies exclusion patterns, validates paths
2. **FileBatcher** (`batcher.py`): Groups files into batches (default 10 files/batch) for token efficiency
3. **CodeAnalyzer** (`analyzer.py`): Orchestrates analysis using provider abstraction
4. **ProviderFactory** (`providers/factory.py`): Auto-detects provider based on model name
5. **Providers** (`providers/`):
   - **BedrockProvider**: AWS Bedrock implementation (Claude, Mistral, Minimax, Kimi, Qwen)
   - **AzureOpenAIProvider**: Azure OpenAI implementation (GPT models)
   - **NVIDIAProvider**: NVIDIA NIM API implementation (Devstral, MiniMax M2)
6. **Aggregation** (`cli.py`): Merges results from all batches (issues, suggestions, design insights)
7. **Renderers** (`renderer.py`): Outputs to Rich terminal UI or Markdown file

### Key Architectural Patterns

**Provider Abstraction Pattern:**
- `ModelProvider` abstract base class defines interface for all LLM providers
- Required methods: `analyze_batch()`, `get_model_display_name()`, `get_pricing()`
- Optional methods: `reset_state()`, `estimate_cost()`, token tracking properties
- **ProviderFactory** auto-detects provider based on model name (ID or alias)
- Creates appropriate provider instance (Bedrock, Azure, or NVIDIA)
- Uses ConfigLoader to resolve model configuration

**Benefits:**
- Easy to add new providers (implement `ModelProvider` interface)
- Clean separation between orchestration (CodeAnalyzer) and provider-specific logic
- Simplified testing (mock at provider level)
- Transparent to CLI users (just specify model name)

**Configuration System:**
- **models.yaml** (`config/models.yaml`): Central configuration for all models and providers
- Defines model IDs, names, aliases, pricing, inference parameters
- Provider-specific settings (AWS region, Azure endpoint/key)
- Environment variable expansion for secrets (`${AZURE_OPENAI_API_KEY}`)
- **ConfigLoader** (`config/loader.py`): Parses YAML with Pydantic validation
- Resolves model names (IDs and aliases) to provider and ModelConfig
- Provides access to provider-specific configuration
- **Pydantic Models** (`config/models.py`): Type-safe configuration with Field validation

**LangChain Structured Output Integration:**
- Uses `.with_structured_output(CodeReviewReport)` on provider-specific LLM clients
- Returns Pydantic models directly from LLM (no manual JSON parsing)
- System prompt in `config.py` specifies JSON schema expectations
- Category normalization handles non-Claude model variations (see `models.py`)

**Pydantic Data Models:**
- `ReviewIssue`: Single finding with category normalization for LLM compatibility
- `CodeReviewReport`: Aggregated results (summary, metrics, issues list, recommendations)
- Models have validators (e.g., `line_end >= line_start`, category mapping)

**Category Normalization:**
Non-Claude models may return non-standard category names. The `ReviewIssue` model includes a `@field_validator` that maps variations to valid categories:
- "error handling", "errorhandling" → "Code Quality"
- "architecture", "design" → "System Design"
- Unknown categories default to "Code Quality"

**Retry Logic with Exponential Backoff:**
Provider-specific retry logic:
- **BedrockProvider**: Handles `ThrottlingException` and `TooManyRequestsException`
- **AzureOpenAIProvider**: Handles `RateLimitError`
- Both use exponential backoff: 2^attempt seconds (1s, 2s, 4s)
- Max 3 retries by default (configurable)

**Parallel Static Analysis:**
- `StaticAnalyzer.run_all(parallel=True)` runs tools concurrently
- Uses `ThreadPoolExecutor` for I/O-bound subprocess calls
- Reduces total time from sum of tools to ~slowest tool

### Code Review Rules

The AI code review behavior is defined in `config/prompts.py`. Key rule sections:

**Severity Classification:**
- **Critical**: Security vulnerabilities, data loss risk, crashes, memory corruption
- **High**: Significant bugs, resource leaks, race conditions, missing error handling
- **Medium**: Code quality issues, minor bugs, suboptimal patterns
- **Low**: Style inconsistencies, naming improvements, minor optimizations
- **Info**: Best practices, documentation improvements, alternative approaches

**Security Analysis:**
- Command/SQL/code injection detection
- Unsafe deserialization, missing input validation
- Insecure cryptographic practices, XSS vulnerabilities
- Missing authentication/authorization checks

**Sensitive Information Detection (Critical Priority):**
- Detects hardcoded secrets: passwords, API keys, tokens, private keys
- Pattern matching for provider-specific keys (Stripe, Slack, GitHub, AWS, Google)
- High-entropy string detection (40+ chars)
- Connection strings with embedded credentials
- Excludes false positives: empty strings, placeholders, env lookups, test files

**Typo and Spelling Detection:**
- Comments and docstrings (primary focus)
- User-facing string literals (error messages, logs, UI text)
- Obviously misspelled identifiers
- 35+ common programming typos (e.g., `recieve`, `occured`, `seperate`)
- Excludes: domain terms, abbreviations (usr, cfg, env), library-specific names

**False Positive Prevention:**
- Only reports issues with >80% confidence
- Context-aware analysis (checks for intentional patterns, comments, test files)
- Excludes defensive code patterns, context managers, glob patterns
- Proportionality guidelines to avoid over-engineering suggestions

**Language-Specific Rules:**
Based on Google Style Guides for Python, Go, Shell/Bash, C++, Java, JavaScript, and TypeScript.

### Supported Models

Models defined in `codereview/config/models.yaml`:

**AWS Bedrock Models:**
| Model | Model ID | Input $/M | Output $/M | Defaults |
|-------|----------|-----------|------------|----------|
| Claude Opus 4.5 | `global.anthropic.claude-opus-4-5-20251101-v1:0` | $5.00 | $25.00 | temp=0.1 |
| Claude Sonnet 4.5 | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | $3.00 | $15.00 | temp=0.1 |
| Claude Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | $1.00 | $5.00 | temp=0.1 |
| Minimax M2 (Bedrock) | `minimax.minimax-m2` | $0.30 | $1.20 | temp=0.3, top_p=0.9, top_k=40, max=8192 |
| Mistral Large 3 | `mistral.mistral-large-3-675b-instruct` | $2.00 | $6.00 | temp=0.1, top_p=0.5, top_k=5 |
| Kimi K2 Thinking | `moonshot.kimi-k2-thinking` | $0.50 | $2.00 | temp=1.0, max=16K-256K |
| Qwen3 Coder 480B | `qwen.qwen3-coder-480b-a35b-v1:0` | $0.22 | $1.40 | temp=0.7, top_p=0.8, top_k=20, max=65536 |

**Azure OpenAI Models:**
| Model | Deployment Name | Input $/M | Output $/M | Defaults |
|-------|-----------------|-----------|------------|----------|
| GPT-5.2 Codex | `gpt-5.2-codex` | $1.75 | $14.00 | temp=0.0, top_p=0.95, max=16000 |

**Note:** GPT-5.2 Codex uses OpenAI's Responses API (not ChatCompletion). This is configured automatically via `use_responses_api: true` in `models.yaml`.

**NVIDIA NIM Models:**
| Model | Model ID | Input $/M | Output $/M | Defaults |
|-------|----------|-----------|------------|----------|
| Devstral 2 123B | `mistralai/devstral-2-123b-instruct-2512` | $0.00* | $0.00* | temp=0.15, top_p=0.95, max=8192 |
| MiniMax M2 (NVIDIA) | `minimaxai/minimax-m2` | $0.00* | $0.00* | temp=0.3, top_p=0.9, max=8192 |
| Qwen3 Coder 480B (NVIDIA) | `qwen/qwen3-coder-480b-a35b-instruct` | $0.00* | $0.00* | temp=0.3, top_p=0.8, max=16384 |
| Kimi K2 Instruct (NVIDIA) | `moonshotai/kimi-k2-instruct-0905` | $0.00* | $0.00* | temp=0.5, top_p=0.9, max=16384 |
| DeepSeek V3.2 (NVIDIA) | `deepseek-ai/deepseek-v3.2` | $0.00* | $0.00* | temp=0.3, top_p=0.9, max=16384 |

**Note:** *NVIDIA models are currently in free tier. Pricing will be updated when NVIDIA announces production pricing.

**Default model:** Claude Opus 4.5

### Configuration Constants

**Core configuration** (`config/`):
- `models.yaml`: All model and provider definitions
- `models.py`: Pydantic data models for validation
- `loader.py`: YAML parsing and model resolution (uses `@lru_cache` singleton pattern)
- `prompts.py`: Code review rules and SYSTEM_PROMPT (primary location for review behavior)

**File processing** (`scanner.py`, `config/`):
- `DEFAULT_EXCLUDE_PATTERNS`: File patterns to skip (.venv, __pycache__, etc.)
- `DEFAULT_EXCLUDE_EXTENSIONS`: Extensions to skip (.json, .md, binaries, etc.)
- `SYSTEM_PROMPT`: Instructions for LLM including language-specific rules
- `MAX_FILE_SIZE_KB`: File size limit (default 500KB)

## Testing Patterns

### Mocking Providers
All tests mock provider calls to avoid real API usage:

```python
from unittest.mock import Mock, patch

# Mock at provider level (preferred)
with patch('codereview.providers.factory.ProviderFactory.create') as mock_factory:
    mock_provider = Mock()
    mock_provider.analyze_batch.return_value = mock_report
    mock_factory.return_value = mock_provider
    # Test code here

# Mock specific provider implementation
with patch('codereview.providers.bedrock.ChatBedrockConverse') as mock_bedrock:
    mock_instance = Mock()
    mock_instance.with_structured_output.return_value.invoke.return_value = mock_report
    mock_bedrock.return_value = mock_instance
    # Test code here

# Reset ConfigLoader singleton between tests
from codereview.config import get_config_loader
get_config_loader.cache_clear()
```

### Test Fixtures
- `tests/fixtures/sample_code/`: Test files for scanner (.py, .go, .sh, .cpp, .java, .js, .ts, .json)
- Fixtures verify inclusion (code files) and exclusion logic (venv, pycache, json files excluded)

### Pydantic Validation Tests
- Category normalization tested (unknown categories map to "Code Quality")
- Line number constraints (ge=1) enforced
- Custom validators (line_end >= line_start) tested

See `tests/test_models.py` for validation patterns.

## Provider Configuration Requirements

### AWS Bedrock
**Runtime Prerequisites:**
1. AWS credentials configured (via `aws configure`, env vars, or IAM role)
2. Bedrock access enabled in AWS region (configured in `models.yaml`)
3. Model access approved for chosen model
4. IAM permissions: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`

**Configuration:**
- Set AWS region in `config/models.yaml` under `bedrock.region`
- Or use `AWS_PROFILE` env var for specific profiles
- Timeout settings in `config/models.py`: `read_timeout` (default 300s), `connect_timeout` (default 60s)

### Azure OpenAI
**Runtime Prerequisites:**
1. Azure OpenAI subscription with approved model access
2. Azure OpenAI endpoint and API key
3. Deployed model (e.g., `gpt-5.2-codex`)

**Configuration:**
- Set `AZURE_OPENAI_API_KEY` environment variable
- Configure endpoint in `config/models.yaml` under `azure_openai.endpoint`
- Or set `AZURE_OPENAI_ENDPOINT` environment variable

### NVIDIA NIM
**Runtime Prerequisites:**
1. NVIDIA API key from https://build.nvidia.com
2. Free tier available for development and testing

**Configuration:**
- Set `NVIDIA_API_KEY` environment variable
- API key format: `nvapi-xxxxx...`
- Optional: Set `NVIDIA_BASE_URL` for self-hosted NIMs

**Getting Started:**
```bash
# Get API key from NVIDIA
# 1. Visit https://build.nvidia.com/explore/discover
# 2. Sign in and generate API key
# 3. Export the key
export NVIDIA_API_KEY="nvapi-your-key-here"

# Run code review with Devstral
uv run codereview ./src --model devstral
```

**For Development/Testing:**
- All provider calls are mocked in tests (no real credentials needed)
- Mock at provider level using `ProviderFactory.create`

## Code Modifications

### Adding New Models
1. Add model entry to `config/models.yaml` under appropriate provider section:
   - `id`: Full model identifier
   - `name`: Display name
   - `aliases`: List of alternative names
   - `pricing`: Input and output prices per million tokens
   - `inference_params`: Temperature, top_p, top_k, max_tokens, etc.
2. No code changes needed - ConfigLoader automatically picks up new models
3. Update this document's Supported Models table

**Example (AWS Bedrock model):**
```yaml
bedrock:
  models:
    - id: "new.model-id"
      name: "New Model Name"
      aliases: ["short-name", "alt-name"]
      pricing:
        input_per_million: 1.00
        output_per_million: 5.00
      inference_params:
        temperature: 0.1
```

**Example (Azure OpenAI model):**
```yaml
azure_openai:
  models:
    - id: "gpt-6-turbo"
      name: "GPT-6 Turbo"
      aliases: ["gpt6", "turbo"]
      pricing:
        input_per_million: 2.00
        output_per_million: 8.00
      inference_params:
        temperature: 0.0
      use_responses_api: true  # Required if model doesn't support ChatCompletion API
```

**Example (NVIDIA NIM model):**
```yaml
nvidia:
  api_key: "${NVIDIA_API_KEY}"
  models:
    - id: "new-model"
      full_id: "vendor/new-model-id"
      name: "New NVIDIA Model"
      aliases: ["new-nim"]
      pricing:
        input_per_million: 0.00   # Free tier, update when priced
        output_per_million: 0.00
      inference_params:
        default_temperature: 0.15
        default_top_p: 0.95
        max_output_tokens: 8192
```

### Adding New Providers
1. Create new provider class in `codereview/providers/` implementing `ModelProvider` interface:
   - **Required methods:**
     - `analyze_batch(files, batch_context)`: Main analysis method
     - `get_model_display_name()`: Return display name for UI
     - `get_pricing()`: Return pricing information dict
   - **Optional methods:** `reset_state()`, `estimate_cost()`, token properties
2. Add provider configuration section to `config/models.yaml`
3. Update `ProviderFactory.create_provider()` to detect and instantiate new provider
4. Add provider-specific tests in `tests/`

**Example provider skeleton:**
```python
from codereview.providers.base import ModelProvider
from codereview.config.models import ModelConfig

class NewProvider(ModelProvider):
    def __init__(self, model_config: ModelConfig, provider_config: dict):
        self.model_config = model_config
        self.provider_config = provider_config

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        # Implementation here
        pass

    def get_model_display_name(self) -> str:
        return self.model_config.name

    def get_pricing(self) -> dict[str, float]:
        return {
            "input_price_per_million": self.model_config.pricing.input_per_million,
            "output_price_per_million": self.model_config.pricing.output_per_million,
        }
```

### Adding New Review Categories
1. Update `models.py`: Add to `ReviewIssue.category` Literal and `VALID_CATEGORIES`
2. Update `CATEGORY_MAPPING` in `models.py` for common variations
3. Update `config.py`: Add category description to SYSTEM_PROMPT

### Changing Default Exclusions
Edit `DEFAULT_EXCLUDE_PATTERNS` or `DEFAULT_EXCLUDE_EXTENSIONS` in `config.py`

### Modifying Code Review Rules
Edit `SYSTEM_PROMPT` in `config/prompts.py`. Key sections:
- **SEVERITY CLASSIFICATION CRITERIA**: Define what constitutes each severity level
- **CONFIDENCE AND FALSE POSITIVE PREVENTION**: Control what issues are reported
- **LANGUAGE-SPECIFIC RULES**: Add or modify language style guides
- **SECURITY ANALYSIS**: Add new vulnerability patterns to detect
- **SENSITIVE INFORMATION DETECTION**: Add new secret patterns or false positives
- **TYPO AND SPELLING DETECTION**: Add common typos or exclusions
- **PERFORMANCE GUIDELINES**: Define when to report performance issues
- **OUTPUT FORMAT**: Keep JSON schema specification intact for structured output

### Adding Support for New Languages
1. Add extension to `scanner.py` target_extensions set
2. Update SYSTEM_PROMPT in `config/prompts.py` with language-specific rules
3. Add language to `LANGUAGE_EXTENSIONS` in `renderer.py`
4. Test with fixture files in `tests/fixtures/`

### Adding New Static Analysis Tools
1. Add tool config to `StaticAnalyzer.TOOLS` in `static_analysis.py`
2. Include: name, description, command, language, optional version_command
3. Handle tool-specific output parsing in `run_tool()` if needed

## Common Issues

**Category Validation Errors with Non-Claude Models:**
- Non-Claude models may return non-standard category names
- The `normalize_category` validator in `models.py` handles this automatically
- Unknown categories default to "Code Quality"

**Pydantic V1 Compatibility Warning (Python 3.14):**
- LangChain uses Pydantic V1 compat layer
- Shows warning but code works fine

**Test Failures Related to Pydantic:**
- Category validation now normalizes instead of raising errors
- Use `test_category_normalization` pattern for testing

**LangChain Structured Output Not Working:**
- Verify Pydantic model has proper `Field()` descriptions
- Check SYSTEM_PROMPT mentions JSON output format
- Ensure model returned from `.with_structured_output()` is used

**Provider Errors in Tests:**
- Mock at provider level using `ProviderFactory.create` for cleaner tests
- For provider-specific tests, mock the provider implementation (e.g., `ChatBedrockConverse`, `AzureChatOpenAI`)
- Ensure mocks are set up BEFORE importing modules that use them

**Configuration Issues:**
- Verify `config/models.yaml` exists and is valid YAML
- Check environment variables are set for provider credentials
- Use `--list-models` to verify model configuration is loaded correctly

**Reusing CodeAnalyzer Instance:**
- Provider state is managed internally
- CodeAnalyzer delegates to provider, no need to manually reset
- Create new analyzer instance if you need fresh state
