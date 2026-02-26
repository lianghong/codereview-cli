# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangChain-based CLI tool for AI-powered code reviews via AWS Bedrock, Azure OpenAI, NVIDIA NIM, and Google Generative AI. Supports multiple models including Claude (Opus, Sonnet, Haiku), GPT-5.2 Codex, Grok 4 Fast Reasoning, Gemini 3.1 Pro, Gemini 3 (Pro, Flash), Devstral 2, MiniMax M2, MiniMax M2.1, Kimi K2.5, Qwen3 Coder, Qwen3 Coder Next, Qwen3.5, DeepSeek-R1, DeepSeek V3.2, GLM 4.7, GLM 4.7 Flash, and GLM-5. Reviews **Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript** codebases with structured output (categories, severity levels, line numbers, suggested fixes).

**Tech Stack:** Python 3.14, LangChain, AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google Generative AI, Pydantic V2, Click, Rich

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
uv pip install ruff mypy isort vulture types-PyYAML

# Run all static analysis tools
uv run ruff check codereview/ tests/        # Linting
uv run ruff format --check codereview/ tests/  # Code formatting (PEP 758 aware)
uv run mypy codereview/                          # Type checking
uv run isort --check-only codereview/ tests/   # Import sorting
uv run vulture codereview/ vulture_whitelist.py --min-confidence 80  # Dead code detection

# Auto-fix issues where possible
uv run ruff check --fix codereview/ tests/
uv run ruff format codereview/ tests/
uv run isort codereview/ tests/

# Verify all tools pass (run before committing)
uv run ruff check codereview/ tests/ && \
uv run ruff format --check codereview/ tests/ && \
uv run isort --check-only codereview/ tests/ && \
uv run mypy codereview/ && \
uv run vulture codereview/ vulture_whitelist.py --min-confidence 80 && \
echo "✓ All static analysis checks passed"
```

**Quality Standards:**
- All code must pass ruff (check + format), isort, mypy, and vulture checks
- Type hints required for all public APIs
- Minimum 80% confidence for vulture (dead code detection)
- Unused imports/variables must be removed
- All provider implementations must include `get_pricing()` method

### Running the Tool
```bash
# Basic usage (uses Claude Opus 4.6 by default)
uv run codereview /path/to/code

# With model selection (use short names!)
uv run codereview /path/to/code --model sonnet
uv run codereview /path/to/code -m haiku
uv run codereview /path/to/code -m gpt  # Azure OpenAI
uv run codereview /path/to/code -m devstral  # NVIDIA NIM
uv run codereview /path/to/code -m qwen
uv run codereview /path/to/code -m gemini-3.1-pro  # Google GenAI (best reasoning)
uv run codereview /path/to/code -m gemini-3-pro   # Google GenAI
uv run codereview /path/to/code -m gemini-3-flash  # Google GenAI (fast)

# With static analysis (runs tools in parallel for speed)
uv run codereview /path/to/code --static-analysis

# With severity filtering (show only high and above)
uv run codereview ./src --severity high

# Dry run (preview files and estimated cost without API calls)
uv run codereview ./src --dry-run

# Export as JSON for CI/CD pipelines
uv run codereview ./src --output report.json --format json

# With all options
uv run codereview ./src -m sonnet --output report.md --severity medium --verbose

# Direct Python invocation (for debugging)
uv run python -m codereview.cli /path/to/code
```

### README Context

The tool automatically discovers your project's README.md to provide context for code reviews:

```bash
# Auto-discover README (prompts for confirmation, auto-confirms after 3s)
uv run codereview ./src

# Specify README explicitly
uv run codereview ./src --readme ./docs/PROJECT.md

# Skip README context
uv run codereview ./src --no-readme
```

The README content is included in each batch sent to the LLM, helping it understand project conventions and requirements. The tool searches the target directory and parent directories for README.md, stopping at the git repository root. The prompt auto-confirms "Y" after 3 seconds if no input is received.

### CLI Options
| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Model to use (see Model Names below) | opus |
| `--output, -o` | Export report to file (Markdown or JSON) | None |
| `--format, -f` | Output format: markdown, json | markdown |
| `--severity, -s` | Minimum severity to display (critical/high/medium/low/info) | info |
| `--temperature` | Model temperature (0.0-2.0) | Model-specific |
| `--batch-size` | Max files per batch (1-50, acts as file-count cap alongside token budget) | 10 |
| `--static-analysis` | Run static analysis tools (parallel) | False |
| `--dry-run` | Preview files and cost without API calls | False |
| `--verbose, -v` | Show detailed progress (includes token budget breakdown) | False |
| `--exclude, -e` | Additional exclusion patterns | None |
| `--max-files` | Maximum files to analyze | None |
| `--max-file-size` | Maximum file size in KB | 500 |
| `--aws-profile` | AWS CLI profile to use | None |
| `--list-models` | List all available models and exit | - |
| `--readme <path>` | Specify README file for project context | None |
| `--no-readme` | Skip README context entirely | False |
| `--no-color` | Disable ANSI colors/styles for copy-paste friendly output | False |

### Model Names
Use primary model IDs (case-insensitive). Run `codereview --list-models` to see all available models.

| Model ID | Name | Provider | Aliases |
|----------|------|----------|---------|
| `opus` | Claude Opus 4.6 | bedrock | claude-opus, opus4.6, claude-opus-4.6 |
| `sonnet` | Claude Sonnet 4.6 | bedrock | claude-sonnet, sonnet4.6, claude-sonnet-4.6 |
| `haiku` | Claude Haiku 4.5 | bedrock | claude-haiku |
| `gpt-5.2-codex` | GPT-5.2 Codex | azure_openai | gpt, gpt52, codex |
| `kimi-k2.5-azure` | Kimi K2.5 (Azure) | azure_openai | kimi25-azure, kimi-azure |
| `grok-4-fast` | Grok 4 Fast Reasoning (Azure) | azure_openai | grok, grok4, grok-fast, g4fast |
| `devstral` | Devstral 2 123B | nvidia | devstral-2 |
| `minimax-nvidia` | MiniMax M2 (NVIDIA) | nvidia | mm2-nvidia |
| `minimax-m2.1-nvidia` | MiniMax M2.1 (NVIDIA) | nvidia | mm2.1-nvidia, minimax-m2.1, mm21 |
| `qwen-bedrock` | Qwen3 Coder 480B (Bedrock) | bedrock | qwen, qwen-coder |
| `deepseek-r1-bedrock` | DeepSeek-R1 (Bedrock) | bedrock | deepseek, deepseek-r1, ds-bedrock, deepseek-bedrock |
| `deepseek-v3.2-bedrock` | DeepSeek V3.2 (Bedrock) | bedrock | deepseek-v3-bedrock, ds-v3-bedrock |
| `minimax-m2.1-bedrock` | MiniMax M2.1 (Bedrock) | bedrock | mm2.1-bedrock |
| `glm47-bedrock` | GLM 4.7 (Bedrock) | bedrock | glm4-bedrock |
| `glm47-flash-bedrock` | GLM 4.7 Flash (Bedrock) | bedrock | glm4-flash, glm47f, glm47-flash |
| `kimi-k2.5-bedrock` | Kimi K2.5 (Bedrock) | bedrock | kimi, kimi-bedrock, kimi25-bedrock |
| `qwen-next-bedrock` | Qwen3 Coder Next (Bedrock) | bedrock | qwen-next, qwen3-next, qwen-coder-next |
| `qwen-nvidia` | Qwen3 Coder 480B (NVIDIA) | nvidia | qwen3-nvidia, qwen-coder-nvidia |
| `qwen3.5-nvidia` | Qwen3.5 397B A17B (NVIDIA) | nvidia | qwen3.5, qwen35, qwen35-nvidia |
| `kimi-k2.5-nvidia` | Kimi K2.5 (NVIDIA) | nvidia | kimi-k2.5, kimi25 |
| `deepseek-v3.2-nvidia` | DeepSeek V3.2 (NVIDIA) | nvidia | deepseek-v3-nvidia, ds-nvidia, deepseek-nvidia |
| `glm47` | GLM 4.7 (NVIDIA) | nvidia | glm4, glm-nvidia |
| `glm5` | GLM-5 (NVIDIA) | nvidia | glm-5, glm5-nvidia |
| `gemini-3.1-pro` | Gemini 3.1 Pro Preview | google_genai | gemini31-pro, g31pro |
| `gemini-3-pro` | Gemini 3 Pro Preview | google_genai | gemini-pro, gemini3-pro, g3pro |
| `gemini-3-flash` | Gemini 3 Flash Preview | google_genai | gemini-flash, gemini3-flash, g3flash |

**Note:** All models are displayed in `--list-models` regardless of provider credentials. Credentials are only required when actually using a model.

### Static Analysis Integration
```bash
# Install Python static analysis tools
# Option 1: Install with optional dependencies
uv pip install -e ".[static-analysis]"

# Option 2: Install manually
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
| Tool use support | `models.yaml` | Set `supports_tool_use: false` for models without tool calling |
| Context window size | `models.yaml` | Set `context_window` per model for token-budget-aware batching |
| API credentials | Environment variables | `AZURE_OPENAI_API_KEY`, `NVIDIA_API_KEY`, `GOOGLE_API_KEY` |
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

google_genai:
  api_key: "${GOOGLE_API_KEY}"
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
      context_window: 128000  # Enables token-budget-aware batching
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
FileScanner → FileBatcher → CodeAnalyzer → ProviderFactory → BedrockProvider/AzureOpenAIProvider/NVIDIAProvider/GoogleGenAIProvider → Aggregation → TerminalRenderer/MarkdownExporter
```

1. **FileScanner** (`scanner.py`): Discovers code files (.py, .go, .sh, .bash, .cpp, .cc, .cxx, .h, .hpp, .java, .js, .jsx, .mjs, .ts, .tsx), applies exclusion patterns, validates paths, tracks all skipped files with reasons (too large, stat failures, etc.)
2. **FileBatcher** (`batcher.py`): Groups files into token-budget-aware batches. When a model's `context_window` is configured, computes a token budget (`context_window - max_output - system_prompt - readme - safety_margin`) and packs files greedily within that budget. Falls back to count-only batching (default 10 files/batch) when no context window is set or budget is non-positive. `--batch-size` always acts as a file-count cap
3. **CodeAnalyzer** (`analyzer.py`): Orchestrates analysis using provider abstraction
4. **ProviderFactory** (`providers/factory.py`): Auto-detects provider based on model name
5. **Providers** (`providers/`):
   - **BedrockProvider**: AWS Bedrock implementation (Claude, Kimi K2.5, Qwen, DeepSeek, MiniMax M2.1, GLM)
   - **AzureOpenAIProvider**: Azure OpenAI implementation (GPT, Kimi K2.5, Grok models)
   - **NVIDIAProvider**: NVIDIA NIM API implementation (Devstral, MiniMax M2, MiniMax M2.1, Qwen3, Qwen3.5, DeepSeek, GLM 4.7, GLM-5)
   - **GoogleGenAIProvider**: Google Generative AI implementation (Gemini 3 Pro, Gemini 3 Flash)
6. **Aggregation** (`cli.py`): Merges results from all batches (issues, suggestions, design insights)
7. **Renderers** (`renderer.py`): Outputs to Rich terminal UI or Markdown file

### Key Architectural Patterns

**Provider Abstraction Pattern:**
- `ModelProvider` abstract base class defines interface for all LLM providers
- Required methods: `analyze_batch()`, `get_model_display_name()`, `get_pricing()`
- Optional methods: `reset_state()`, `estimate_cost()`, token tracking properties
- **ProviderFactory** auto-detects provider based on model name (ID or alias)
- Creates appropriate provider instance (Bedrock, Azure, NVIDIA, or Google GenAI)
- Uses ConfigLoader to resolve model configuration

**Benefits:**
- Easy to add new providers (implement `ModelProvider` interface)
- Clean separation between orchestration (CodeAnalyzer) and provider-specific logic
- Simplified testing (mock at provider level)
- Transparent to CLI users (just specify model name)

**Configuration System:**
- **models.yaml** (`config/models.yaml`): Central configuration for all models and providers
- Defines model IDs, names, aliases, pricing, inference parameters
- Provider-specific settings (AWS region, Azure endpoint/key, NVIDIA/Google API keys)
- Environment variable expansion for secrets (`${AZURE_OPENAI_API_KEY}`, `${GOOGLE_API_KEY}`)
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
- **NVIDIAProvider**: Handles gateway errors (502/503/504) and rate limits (429)
- **GoogleGenAIProvider**: Handles `ResourceExhausted` (429) and `ServiceUnavailable` (503)
- All use exponential backoff capped at 60 seconds with configurable max retries
- NVIDIA uses longer initial wait (4s) for 504 gateway timeouts

**Parallel Static Analysis:**
- `StaticAnalyzer.run_all(parallel=True)` runs tools concurrently
- Uses `ThreadPoolExecutor` for I/O-bound subprocess calls
- Reduces total time from sum of tools to ~slowest tool

**Token-Budget-Aware Batching:**
- `FileBatcher` supports an optional `token_budget` parameter computed from the model's `context_window`
- Budget formula: `context_window - max_output_tokens - system_prompt_tokens - readme_tokens - safety_margin`
- Safety margin: `clamp(context_window // 10, 1000, 20000)` — 10% of context, covers estimation error
- Token estimation heuristic: `file_size_bytes // 4 + 50` (50 tokens overhead per file for headers/separators)
- Greedy packing: adds files to current batch until next file would exceed token budget or file-count cap
- Oversized files (exceeding budget alone) get their own single-file batch with a warning
- Falls back to count-only batching when `context_window` is not set or budget computes to non-positive
- `--batch-size` CLI option still works as a max file-count cap alongside the token budget
- `--verbose` displays the full token budget breakdown

### Code Review Rules

The AI code review behavior is defined in `config/prompts.py` (~310 lines). The prompt is structured for maximum instruction adherence with critical constraints near the top.

**Prompt Structure (in order):**
1. Role & task definition
2. Core constraints (8 critical rules — positioned first for attention)
3. Prompt injection defense (ignores adversarial instructions in user code)
4. Severity classification (with concrete examples per level)
5. False positive prevention (consolidated from multiple sections)
6. Security analysis (with CWE references)
7. Architecture & production readiness (boundary violations, coupling, observability)
8. Performance guidelines
9. Testing quality (anti-patterns, coverage gaps)
10. Language-specific rules (Python, Go, Shell, C++, Java, JS, TS)
11. Typo detection (concise instructions, not exhaustive lists)
12. Output requirements (categories, severities, format)
13. Self-verification step (line number check, false positive check, context check)
14. Two concrete examples (good finding + false positive to avoid)

**Severity Classification (with examples):**
- **Critical**: Security vulnerabilities, data loss risk, crashes — e.g., `subprocess.call(user_input, shell=True)`
- **High**: Significant bugs, resource leaks, race conditions — e.g., file handle not closed in error path
- **Medium**: Code quality issues, minor bugs — e.g., bare `except Exception` swallowing errors
- **Low**: Style inconsistencies, naming improvements, minor optimizations
- **Info**: Best practices, documentation improvements, alternative approaches

**Security Analysis (with CWE references):**
- CWE-78 (OS Command Injection), CWE-89 (SQL Injection), CWE-94 (Code Injection)
- CWE-798 (Hardcoded Credentials), CWE-327 (Weak Cryptography)
- CWE-79 (XSS), CWE-502 (Unsafe Deserialization), CWE-400 (Resource Exhaustion)
- Sensitive information detection with false positive exclusions

**False Positive Prevention:**
- Only reports issues with >80% confidence
- Context-aware self-verification before each issue
- Consolidated exclusion rules (defensive patterns, test files, glob patterns, etc.)
- Proportionality guidelines to avoid over-engineering suggestions
- Prompt injection defense prevents user code from altering review behavior

**Language-Specific Rules:**
Based on Google Style Guides for Python, Go, Shell/Bash, C++, Java, JavaScript, and TypeScript.

### Supported Models

Models defined in `codereview/config/models.yaml`:

**AWS Bedrock Models:**
| Model | Model ID | Input $/M | Output $/M | Defaults |
|-------|----------|-----------|------------|----------|
| Claude Opus 4.6 | `global.anthropic.claude-opus-4-6-v1` | $5.00 | $25.00 | temp=0.1, max=128000 |
| Claude Sonnet 4.6 | `global.anthropic.claude-sonnet-4-6` | $3.00 | $15.00 | temp=0.1 |
| Claude Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | $1.00 | $5.00 | temp=0.1 |
| Qwen3 Coder 480B (Bedrock) | `qwen.qwen3-coder-480b-a35b-v1:0` | $0.22 | $1.40 | temp=0.3, top_p=0.8, top_k=20, max=65536 |
| DeepSeek-R1 (Bedrock) | `us.deepseek.r1-v1:0` | $1.35 | $5.40 | temp=0.6, max=32000 |
| DeepSeek V3.2 (Bedrock) | `deepseek.v3.2` | $0.62 | $1.85 | temp=0.3, top_p=0.9, max=16384 |
| MiniMax M2.1 (Bedrock) | `minimax.minimax-m2.1` | $0.30 | $1.20 | temp=1.0, top_p=0.95, top_k=40, max=128000, thinking=on |
| GLM 4.7 (Bedrock) | `zai.glm-4.7` | $0.00* | $0.00* | temp=0.5, top_p=0.95, max=16384, thinking=on |
| GLM 4.7 Flash (Bedrock) | `zai.glm-4.7-flash` | $0.00* | $0.00* | temp=0.5, top_p=0.95, max=8192 |
| Kimi K2.5 (Bedrock) | `moonshotai.kimi-k2.5` | $0.60 | $3.00 | temp=0.6, top_p=0.95, top_k=40, max=65536 |
| Qwen3 Coder Next (Bedrock) | `qwen.qwen3-coder-next` | $0.50 | $1.20 | temp=0.7, top_p=0.95, top_k=40, max=16384 |

**Note:** *GLM Bedrock pricing TBD - update when AWS publishes official pricing.

**Azure OpenAI Models:**
| Model | Deployment Name | Input $/M | Output $/M | Defaults |
|-------|-----------------|-----------|------------|----------|
| GPT-5.2 Codex | `gpt-5.2-codex` | $1.75 | $14.00 | temp=0.0, top_p=0.95, max=128000 |
| Kimi K2.5 (Azure) | `Kimi-K2.5` | $0.60 | $3.00 | temp=0.6, top_p=0.95, max=65536 |
| Grok 4 Fast Reasoning (Azure) | `grok-4-fast-reasoning` | $0.20 | $0.50 | temp=0.1, top_p=0.95, max=32000 |

**Note:** GPT-5.2 Codex uses OpenAI's Responses API (not ChatCompletion). This is configured automatically via `use_responses_api: true` in `models.yaml`. Kimi K2.5 and Grok 4 Fast Reasoning use the standard ChatCompletion API.

**Note:** DeepSeek-R1 doesn't support tool use, so it uses prompt-based JSON parsing instead of structured output. This is configured via `supports_tool_use: false` in `models.yaml`.

**NVIDIA NIM Models:**
| Model | Model ID | Input $/M | Output $/M | Defaults |
|-------|----------|-----------|------------|----------|
| Devstral 2 123B | `mistralai/devstral-2-123b-instruct-2512` | $0.00* | $0.00* | temp=0.15, top_p=0.95, max=8192 |
| MiniMax M2 (NVIDIA) | `minimaxai/minimax-m2` | $0.00* | $0.00* | temp=0.3, top_p=0.9, max=8192 |
| MiniMax M2.1 (NVIDIA) | `minimaxai/minimax-m2.1` | $0.00* | $0.00* | temp=1.0, top_p=0.95, top_k=40, max=128000, thinking=on |
| Qwen3 Coder 480B (NVIDIA) | `qwen/qwen3-coder-480b-a35b-instruct` | $0.00* | $0.00* | temp=0.3, top_p=0.8, max=16384, thinking=on |
| Qwen3.5 397B A17B (NVIDIA) | `qwen/qwen3.5-397b-a17b` | $0.00* | $0.00* | temp=0.6, top_p=0.95, top_k=20, max=16384, thinking=on |
| Kimi K2.5 (NVIDIA) | `moonshotai/kimi-k2.5` | $0.00* | $0.00* | temp=0.6, top_p=0.95, top_k=40, max=16384 |
| DeepSeek V3.2 (NVIDIA) | `deepseek-ai/deepseek-v3.2` | $0.00* | $0.00* | temp=0.3, top_p=0.9, max=16384, thinking=on |
| GLM 4.7 (NVIDIA) | `z-ai/glm4.7` | $0.00* | $0.00* | temp=0.5, top_p=0.95, max=16384, thinking=on |
| GLM-5 (NVIDIA) | `z-ai/glm5` | $0.00* | $0.00* | temp=1.0, top_p=0.95, max=128000 |

**Note:** *NVIDIA models are currently in free tier. Pricing will be updated when NVIDIA announces production pricing. Models with `thinking=on` use interleaved thinking mode for deeper reasoning.

**Google Generative AI Models:**
| Model | Model ID | Input $/M | Output $/M | Defaults |
|-------|----------|-----------|------------|----------|
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | $2.00 | $12.00 | temp=0.1, top_p=0.95, max=65536 |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` | $2.00 | $12.00 | temp=0.1, top_p=0.95, max=65536 |
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | $0.50 | $3.00 | temp=0.1, top_p=0.95, max=65536 |

**Note:** Google GenAI models have 1M token context windows. Uses `method="json_schema"` for structured output.

**Default model:** Claude Opus 4.6

### Configuration Constants

**Core configuration** (`config/`):
- `models.yaml`: All model and provider definitions (includes `context_window` per model)
- `models.py`: Pydantic data models for validation (ModelConfig includes `context_window` for token-budget-aware batching)
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

### Google Generative AI
**Runtime Prerequisites:**
1. Google API key from https://aistudio.google.com/apikey
2. Gemini model access (available immediately with API key)

**Configuration:**
- Set `GOOGLE_API_KEY` environment variable
- Models use structured output with `method="json_schema"`

**Getting Started:**
```bash
# Get API key from Google AI Studio
# 1. Visit https://aistudio.google.com/apikey
# 2. Create an API key
# 3. Export the key
export GOOGLE_API_KEY="your-api-key-here"

# Run code review with Gemini 3.1 Pro (best reasoning)
uv run codereview ./src --model gemini-3.1-pro

# Run with Gemini 3 Pro
uv run codereview ./src --model gemini-3-pro

# Run with Gemini 3 Flash (faster, cheaper)
uv run codereview ./src --model gemini-3-flash
```

## Code Modifications

### Adding New Models
1. Add model entry to `config/models.yaml` under appropriate provider section:
   - `id`: Full model identifier
   - `name`: Display name
   - `aliases`: List of alternative names
   - `pricing`: Input and output prices per million tokens
   - `inference_params`: Temperature, top_p, top_k, max_tokens, etc.
   - `context_window`: Context window size in tokens (enables token-budget-aware batching)
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
      context_window: 200000
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
      context_window: 400000
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
      context_window: 128000
```

**Example (Google GenAI model):**
```yaml
google_genai:
  api_key: "${GOOGLE_API_KEY}"
  models:
    - id: "gemini-new"
      full_id: "gemini-new-model-id"
      name: "Gemini New Model"
      aliases: ["gnew"]
      pricing:
        input_per_million: 1.00
        output_per_million: 5.00
      inference_params:
        default_temperature: 0.1
        default_top_p: 0.95
        max_output_tokens: 65536
      context_window: 1000000
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
- **CORE CONSTRAINTS**: 8 critical rules positioned at top for maximum adherence
- **SEVERITY CLASSIFICATION**: Severity levels with concrete examples
- **FALSE POSITIVE PREVENTION**: Consolidated exclusion rules and confidence threshold
- **SECURITY ANALYSIS**: Vulnerability patterns with CWE references
- **SENSITIVE INFORMATION DETECTION**: Secret patterns with false positive exclusions
- **ARCHITECTURE & PRODUCTION READINESS**: Boundary violations, coupling, observability
- **LANGUAGE-SPECIFIC RULES**: Per-language style guide rules
- **SELF-VERIFICATION**: Pre-output validation step
- **EXAMPLES**: Good finding and false positive examples (update to match new patterns)

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
- For provider-specific tests, mock the provider implementation (e.g., `ChatBedrockConverse`, `AzureChatOpenAI`, `ChatGoogleGenerativeAI`)
- Ensure mocks are set up BEFORE importing modules that use them

**Configuration Issues:**
- Verify `config/models.yaml` exists and is valid YAML
- Check environment variables are set for provider credentials
- Use `--list-models` to verify model configuration is loaded correctly

**Reusing CodeAnalyzer Instance:**
- Provider state is managed internally
- CodeAnalyzer delegates to provider, no need to manually reset
- Create new analyzer instance if you need fresh state
