# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

LangChain-based CLI for AI code review across **7 providers**: AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google GenAI, DeepSeek direct, Z.AI (Zhipu international), and Moonshot (Kimi). Reviews **Python, Go, Shell, C++, Java, JS, TS** with structured output (severity, line numbers, suggested fixes).

**Stack:** Python 3.14, LangChain (1.3+), Pydantic V2, Click, Rich, AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google GenAI, DeepSeek (`langchain-deepseek`), Z.AI (`langchain-openai` + custom base_url), Moonshot (`langchain-moonshot`).

For the live model list with pricing/aliases run `uv run codereview --list-models` — that output is authoritative; the YAML in `codereview/config/models.yaml` is the source of truth. Default model: **Claude Opus 4.7**.

## Development commands

```bash
# Setup
uv venv --python 3.14
uv pip install -e .

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_models.py::test_review_issue_creation -v
uv run pytest tests/ --cov=codereview --cov-report=html

# All quality gates (run before committing)
uv run ruff check codereview/ tests/ && \
  uv run ruff format --check codereview/ tests/ && \
  uv run isort --check-only codereview/ tests/ && \
  uv run mypy codereview/ && \
  uv run vulture codereview/ vulture_whitelist.py --min-confidence 80

# Auto-fix
uv run ruff check --fix codereview/ tests/
uv run ruff format codereview/ tests/
uv run isort codereview/ tests/

# Run the tool
uv run codereview /path/to/code                           # default: opus4.7
uv run codereview ./src --model sonnet --output report.md
uv run codereview ./src --static-analysis --severity high
uv run codereview ./src --dry-run                          # preview cost/files
uv run codereview ./src --output report.json --format json # CI-friendly
```

**Quality bar:** ruff (check + format) + isort + mypy + vulture (≥80% confidence) all clean. Type hints on all public APIs. Python 3.14: PEP 758 unparenthesized exceptions, PEP 765 no control flow in `finally`. Every provider must implement `get_pricing()`.

## CLI options

| Option | Description | Default |
|---|---|---|
| `--model, -m` | Model ID or alias (`--list-models` to see) | opus4.7 |
| `--output, -o` | Export report (md or json) | None |
| `--format, -f` | `markdown` or `json` | markdown |
| `--severity, -s` | Min severity: critical/high/medium/low/info | info |
| `--temperature` | 0.0-2.0 | model default |
| `--batch-size` | Max files per batch (file-count cap atop token budget) | 10 |
| `--static-analysis` | Run installed linters in parallel | False |
| `--dry-run` | Preview without API calls | False |
| `--verbose, -v` | Detailed progress + token-budget breakdown | False |
| `--exclude, -e` | Extra glob patterns | None |
| `--max-files` / `--max-file-size` | File caps | None / 500 KB |
| `--aws-profile` | AWS profile name | None |
| `--readme <path>` / `--no-readme` | README context override | auto-discover |
| `--no-color` | Strip ANSI for paste-friendly output | False |

## Architecture

```
FileScanner → FileBatcher → CodeAnalyzer → ProviderFactory → {Bedrock|Azure|NVIDIA|GoogleGenAI}Provider
            → Aggregation (cli.py) → TerminalRenderer / MarkdownExporter
```

- **scanner.py** — discovers code files, applies exclusions, tracks skips with reasons.
- **batcher.py** — token-budget-aware batching when `context_window` is set; falls back to count-only. Budget = `context_window − max_output − system_prompt − readme − safety_margin`. Safety margin = `clamp(context_window // 10, 1000, 20000)`. Token estimate = `bytes // 4 + 50`. Greedy packing; oversized files are skipped with a warning.
- **analyzer.py** — orchestration. Delegates to provider; tracks `skipped_files` (lock-guarded for concurrent batches).
- **providers/factory.py** — auto-detects provider from model name (id or alias).
- **providers/{bedrock,azure_openai,nvidia,google_genai}.py** — share `ModelProvider` ABC with template-method hooks: `_is_retryable_error`, `_calculate_backoff`, `_extract_token_usage`. Token tracking via `TokenTrackingMixin` (lock-guarded `+=`).
- **cli.py** — runs batches concurrently via `ThreadPoolExecutor` (≤4 workers); `--stream` and single-batch runs stay sequential. Aborts when all batches fail; warns about partial results when some fail.

### Key patterns

- **Structured output:** `.with_structured_output(CodeReviewReport, include_raw=True)` — `include_raw` is required to read real token counts from the raw `AIMessage`. MiniMax M2.5 (Bedrock), DeepSeek-V4-Pro (Azure), and Kimi K2.6 (Moonshot) set `supports_tool_use: false` and use `PydanticOutputParser` instead. Bedrock, Azure, and Moonshot providers all honor this flag (mirrored implementations: `_use_prompt_parsing` flag, format-instructions injection in `analyze_batch`). For K2.6 the trigger is server-side: Moonshot rejects `tool_choice='specified'` while thinking mode is enabled, which is exactly what `.with_structured_output()` would set.
- **Category normalization** (`models.py`): non-Claude models return varying category names; `@field_validator` maps them (e.g., `"error handling" → "Code Quality"`). Unknown → `"Code Quality"`.
- **Retry/backoff:** per-provider, exponential, capped at 60s. Azure honours `Retry-After`. NVIDIA uses 4s base for 504. Google uses 10s base for `ResourceExhausted` on preview models.
- **Prompt injection defense:** `SYSTEM_PROMPT` instructs the model to treat code AND README content as data, never instructions. Don't add new "trusted" message paths without extending that defense.
- **Parallel static analysis:** `StaticAnalyzer.run_all(parallel=True)` uses `ThreadPoolExecutor`; rglob helpers skip symlinks defensively.
- **Pricing display:** zero-priced models (placeholder for unannounced rates) render `Estimated cost: TBD`, not `$0.0000`. See `_is_pricing_tbd` in `cli.py`.

## Configuration

```
codereview/config/
├── models.yaml   # All models, providers, pricing, inference params, scanning rules
├── models.py     # Pydantic schema (ModelConfig, ProviderConfig, etc.)
├── loader.py     # YAML + ${VAR} env expansion, @lru_cache singleton
└── prompts.py    # SYSTEM_PROMPT — code review behavior lives here
```

**Configurable via `models.yaml` (no code changes):** model registration, pricing, inference params (`temperature`, `top_p`, `top_k`, `max_output_tokens`), `context_window`, `supports_tool_use`, `use_responses_api`, AWS region, scanning patterns/extensions, max file size.

**Secrets via env vars** (expanded with `${VAR}` syntax in YAML): `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `NVIDIA_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `KIMI_API_KEY`. AWS uses standard credential chain.

## Provider credentials

| Provider | Env var(s) | Sign-up |
|---|---|---|
| AWS Bedrock | standard credential chain (or `--aws-profile`) — needs `bedrock:InvokeModel`/`InvokeModelWithResponseStream` and model access in the YAML region | aws.amazon.com/bedrock |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | portal.azure.com |
| NVIDIA NIM | `NVIDIA_API_KEY` (`nvapi-...`); free tier available | build.nvidia.com |
| Google GenAI | `GOOGLE_API_KEY` | aistudio.google.com/apikey |
| DeepSeek direct | `DEEPSEEK_API_KEY`; via `langchain-deepseek` | platform.deepseek.com/api_keys |
| Z.AI (Zhipu) | `ZAI_API_KEY`; via OpenAI-compat (`ChatOpenAI` + `base_url`) | z.ai |
| Moonshot (Kimi) | `KIMI_API_KEY`; default endpoint is `.cn` (Chinese platform). Override `base_url` to `https://api.moonshot.ai/v1` for international keys | platform.moonshot.cn |

Tests mock at the provider level — no credentials needed for `pytest`.

**Canonical-owner convention** for the model registry: when the same model is exposed by both a vendor's direct API and a re-hoster (Bedrock/NVIDIA/Azure), the **direct API owns the canonical aliases**. E.g. `deepseek-v4-pro` routes to DeepSeek direct, not NVIDIA's free re-host (`dsv4-nvidia`); `kimi` and `kimi-k2.6` route to Moonshot direct, not Bedrock's K2.5 (`kimi-bedrock`) or NVIDIA's K2.6 (`kimi-nvidia-26`). Re-host entries keep provider-suffixed aliases only.

## Adding things

**New model:** add an entry under the matching provider in `codereview/config/models.yaml`. Fields: `id`, `full_id` (provider's identifier), `name`, `aliases`, `pricing.input_per_million`/`output_per_million`, `inference_params`, `context_window`. Use immediately: `codereview ./src --model <id-or-alias>`. Existing entries in `models.yaml` are the best reference.

**New provider:** subclass `ModelProvider` in `codereview/providers/`, implement `analyze_batch`, `_create_model`, `_create_chain`, `_extract_token_usage`, `_is_retryable_error`, `_calculate_backoff`, `validate_credentials`. Register in `ProviderFactory.create_provider` (and add the `<Name>Config` Pydantic class to `config/models.py`, plus a parsing branch in `config/loader.py`). Add the env-var to `cli.py`'s Provider Setup table. **Reference implementations**:
- OpenAI-compatible vendor with dedicated langchain package → mirror `providers/deepseek.py` (uses `ChatDeepSeek`, `api_base` parameter)
- OpenAI-compatible vendor without dedicated package → mirror `providers/zai.py` (uses `ChatOpenAI` with custom `base_url`)
- BaseChatOpenAI subclass with vendor-specific quirks → mirror `providers/moonshot.py` (uses `ChatMoonshot`, accepts `base_url` alias)
- Tool-use-less endpoints → mirror Bedrock's prompt-based JSON path (set `_use_prompt_parsing=True`, append `PydanticOutputParser` to the chain, inject format instructions via `_system_prompt_with_format_instructions`).

**New review category:** add to `ReviewIssue.category` Literal + `VALID_CATEGORIES` + `CATEGORY_MAPPING` in `models.py`, then mention it in `SYSTEM_PROMPT` (`config/prompts.py`).

**New language:** add extension to `FileScanner.target_extensions`, add language section to `SYSTEM_PROMPT`, add to `LANGUAGE_EXTENSIONS` in `renderer.py`, add a fixture under `tests/fixtures/`.

**New static-analysis tool:** add to `StaticAnalyzer.TOOLS` (name, description, command, language, optional `version_command`); handle tool-specific output parsing in `run_tool` if it has unusual exit codes.

## Static analysis

```bash
uv pip install -e ".[static-analysis]"           # Python tools
uv run codereview ./src --static-analysis        # Run alongside AI review
```

Tools detected at runtime; only installed ones run. Python (ruff/mypy/black/isort/vulture/bandit), Go (golangci-lint/go vet/gofmt/gosec), Shell (shellcheck/bashate), C++ (clang-tidy/cppcheck/clang-format), Java (checkstyle), JS/TS (eslint/prettier/tsc/npm audit). See `docs/static-analysis.md` for install per-language.

## Testing patterns

Mock at the provider boundary:

```python
with patch('codereview.providers.factory.ProviderFactory.create') as f:
    f.return_value.analyze_batch.return_value = mock_report

# For provider-specific tests, mock the LLM client itself:
with patch('codereview.providers.bedrock.ChatBedrockConverse') as m:
    m.return_value.with_structured_output.return_value.invoke.return_value = mock_report

# Reset the ConfigLoader singleton between tests that mutate config:
from codereview.config import get_config_loader
get_config_loader.cache_clear()
```

Fixtures live in `tests/fixtures/sample_code/` (verifies inclusion + exclusion logic). Validation rules (`line_end >= line_start`, category normalization) are tested in `tests/test_models.py`.

## Gotchas

- **Pydantic V1 compat warning** under Python 3.14 is upstream from LangChain — harmless.
- **Reasoning models** (Claude Opus 4.7, GPT-5.4, GPT-5.4 Pro, DeepSeek-V4-Pro) don't accept `temperature`/`top_p`. Bedrock and Azure providers both pass `allow_none=True` to `_resolve_temperature`; omit `default_temperature` from `inference_params` for new reasoning models.
- **MiniMax M2.5 on Bedrock, DeepSeek-V4-Pro on Azure, and Kimi K2.6 on Moonshot** lack tool-use → use prompt-based JSON parsing via `supports_tool_use: false`. All three providers branch in `_create_model` and append a `PydanticOutputParser` to the chain. K2.6's case is unusual: the model *can* tool-call, but Moonshot's server rejects `tool_choice='specified'` (HTTP 400) whenever thinking mode is enabled — which is K2.6's whole value proposition for code review. `.with_structured_output()` would set exactly that tool_choice, so we route around it.
- **`use_responses_api: true`** for GPT-5.x in `models.yaml` — ChatCompletion API does not support reasoning summaries for these.
- **Concurrent batches:** `TokenTrackingMixin._track_tokens` and `CodeAnalyzer.skipped_files` are lock-guarded. Don't add other shared mutable state to providers without a lock.
- **`--list-models`** shows everything regardless of credentials; credentials are only validated when a model is actually used.
- **DeepSeek-V4-Pro on Azure / SGLang null-model bug**: the Foundry endpoint validates `body.model` strictly. langchain-openai's `AzureChatOpenAI` defaults `model_name=None` and serializes `"model": null`, which real Azure-OpenAI ignores but SGLang rejects with HTTP 400. The Azure provider explicitly sets `model=deployment_name` to satisfy both backends.
- **Moonshot has two platforms**: `platform.moonshot.cn` (Chinese, default in our YAML) and `platform.moonshot.ai` (international). Keys are NOT interchangeable. `KIMI_API_KEY` typically maps to `.cn`; users with `.ai` keys must override `base_url` in the moonshot section.
- **Determinism for static analysis**: when `MAX_FILES_PER_TOOL=500` truncation triggers, file lists must be `sorted(...)[:N]`. Locked in by `tests/test_static_analysis.py::test_truncation_is_deterministic`. `MAX_FILES_PER_TOOL`'s docstring documents this as a design guarantee.
- **`--tool-timeout`** plumbs through to `subprocess.run(timeout=...)` for static-analysis tools (default 120s). `--include-hidden` opts into `.github/scripts/` etc. Both are off the FileScanner / StaticAnalyzer constructors as well, not just the CLI.
