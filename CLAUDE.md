# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

LangChain-based CLI for AI code review across **8 providers**: AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google GenAI, DeepSeek direct, Z.AI (Zhipu international), Moonshot (Kimi), and OpenAI-on-Bedrock (GPT-5.x via Bedrock's OpenAI-compatible endpoint). Reviews **Python, Go, Shell, C++, Java, JS, TS** with structured output (severity, line numbers, suggested fixes).

**Stack:** Python 3.14, LangChain (1.3+), Pydantic V2, Click, Rich, AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google GenAI, DeepSeek (`langchain-deepseek`), Z.AI (`langchain-openai` + custom base_url), Moonshot (`langchain-moonshot`).

For the live model list with pricing/aliases run `uv run codereview --list-models` — that output is authoritative; the YAML in `codereview/config/models.yaml` is the source of truth. Default model: **Claude Opus 4.8**.

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
uv run codereview /path/to/code                           # default: opus4.8
uv run codereview ./src --model sonnet --output report.md
uv run codereview ./src --static-analysis --severity high
uv run codereview ./src --dry-run                          # preview cost/files
uv run codereview ./src --output report.json --format json # CI-friendly
```

**Quality bar:** ruff (check + format) + isort + mypy + vulture (≥80% confidence) all clean. Type hints on all public APIs. Python 3.14: PEP 758 unparenthesized exceptions, PEP 765 no control flow in `finally`. Every provider must implement `get_pricing()`.

## CLI options

| Option | Description | Default |
|---|---|---|
| `--model, -m` | Model ID or alias (`--list-models` to see) | opus4.8 |
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
FileScanner → FileBatcher → CodeAnalyzer → ProviderFactory → {Bedrock|Azure|NVIDIA|GoogleGenAI|ZAI|DeepSeek|Moonshot|BedrockOpenAI}Provider
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
- **Retry/backoff:** per-provider, exponential, capped at 60s. Azure honours `Retry-After`. NVIDIA uses 4s base for 504. Google uses 10s base for `ResourceExhausted` on preview models. **Output-parsing failures are retried under `enable_output_fixing`** via a dedicated `except` in `_execute_with_retry` that names three shapes: `ValidationError` (tool-use schema violation), `OutputParsingRetryError` (include_raw `parsed` is None), and `OutputParserException` (prompt-parsing path got malformed JSON — a `ValueError` subclass but NOT a `ValidationError`, so it must be named explicitly or it falls into the generic non-retryable `except`). Reasoning models on the prompt-parsing path (e.g. GPT-5.5/5.4 on Bedrock) intermittently emit invalid JSON on think-heavy batches; the retry is what makes those runs complete.
- **Prompt injection defense:** `SYSTEM_PROMPT` instructs the model to treat code AND README content as data, never instructions. Don't add new "trusted" message paths without extending that defense.
- **Parallel static analysis:** `StaticAnalyzer.run_all(parallel=True)` uses `ThreadPoolExecutor`; rglob helpers skip symlinks defensively.
- **Pricing display:** zero-priced models (placeholder for unannounced rates) render `Estimated cost: TBD`, not `$0.0000`. See `_is_pricing_tbd` in `cli.py`.

#### Structured-output path matrix

Which structured-output path a model uses, and why. **When adding a reasoning/thinking model, assume the prompt-parsing path until a live run proves tool-use works** — the failure is often intermittent (only on batches where the model thinks). Set `supports_tool_use: false` in `models.yaml` to opt into prompt parsing; the provider appends a `PydanticOutputParser` and injects format instructions.

| Model (provider) | Thinking | `supports_tool_use` | Path | Why prompt-parsing (if so) |
|---|---|---|---|---|
| Claude Opus 4.8 / 4.7 (Bedrock) | adaptive (server-side) | `false` | prompt | Forced `tool_choice` while thinking → tool call returned as **literal text** → `list_type` error (intermittent) |
| Claude Opus 4.6 (Bedrock) | none | `true` | tool-use | Not adaptive-thinking — forced `tool_choice` works |
| GPT-5.5 / GPT-5.4 (**Bedrock** OpenAI-compat) | adaptive (server-side) | `false` | prompt | Think-heavy batches return reasoning-only (`tool_calls=[]`, no `parsed`) → "no 'parsed' field" (intermittent) |
| GPT-5.4 / 5.4 Pro (**Azure**) | reasoning | `true` | tool-use | Azure deployment tolerates forced `tool_choice`; Bedrock's endpoint does not |
| Kimi K2.6 (Moonshot) | enabled (server-side) | `false` | prompt | Moonshot rejects `tool_choice='specified'` (HTTP 400) while thinking |
| DeepSeek-V4-Pro (**Azure** Foundry) | — | `false` | prompt | SGLang backend lacks tool calling |
| DeepSeek V4 family (**DeepSeek direct**) | off by default | `true` | tool-use | Tool calling works when thinking disabled. **`inference_params.thinking: enabled` flips this entry to the prompt path at runtime** (see `deepseek._create_model`) |
| MiniMax M2.5 (Bedrock) | — | `false` | prompt | No usable tool-based structured output |
| GLM-5.1 (Z.AI) | — | `false` | prompt | Endpoint ignores `json_schema` response_format, returns markdown-fenced JSON; `PydanticOutputParser` strips fences |
| Everything else (Claude Sonnet, GPT-OSS, Qwen, Gemini, …) | — | `true` (default) | tool-use | Standard `.with_structured_output()` |

Two distinct failure shapes drive the `false` cases: **"can't tool-call at all"** (MiniMax, DeepSeek-on-Azure, GLM-5.1 fenced JSON) and **"can tool-call but not *while thinking*"** (Opus 4.7/4.8, GPT-5.5/5.4-on-Bedrock, K2.6) — the latter are intermittent. The Gotchas section has the full per-model detail.

## Configuration

```
codereview/config/
├── models.yaml   # All models, providers, pricing, inference params, scanning rules
├── models.py     # Pydantic schema (ModelConfig, ProviderConfig, etc.)
├── loader.py     # YAML + ${VAR} env expansion, @lru_cache singleton
└── prompts.py    # SYSTEM_PROMPT — code review behavior lives here
```

**Configurable via `models.yaml` (no code changes):** model registration, pricing, inference params (`temperature`, `top_p`, `top_k`, `max_output_tokens`), `context_window`, `supports_tool_use`, `use_responses_api`, AWS region, scanning patterns/extensions, max file size.

**Secrets via env vars** (expanded with `${VAR}` syntax in YAML): `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `NVIDIA_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `KIMI_API_KEY`, `OPENAI_API_KEY` + `OPENAI_BASE_URL` (OpenAI-on-Bedrock). AWS Bedrock (Converse path) uses the standard credential chain; OpenAI-on-Bedrock uses a Bedrock API key (bearer token) instead.

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
| OpenAI-on-Bedrock | `OPENAI_API_KEY` (an **Amazon Bedrock API key** / bearer token, *not* an openai.com key) + `OPENAI_BASE_URL` (region's Bedrock OpenAI endpoint); via OpenAI-compat (`ChatOpenAI` + `base_url`). Set `BEDROCK_OPENAI_MODEL_ID` into each model's `full_id`. | Bedrock console → API keys |

Tests mock at the provider level — no credentials needed for `pytest`.

### `validate_credentials` semantics (`--validate`)

Every provider's `validate_credentials` returns a `ValidationResult`. Keep the hard-failure (`valid=False`) vs warning distinction **consistent across providers** — an inconsistency here is what let a bad Azure key report success once. The contract:

- **Hard failure** (`result.valid = False`, via `add_check(..., False, msg)`) — a problem that *will* break the run: missing/placeholder API key, non-HTTPS `base_url`, unparseable endpoint, and an **explicit auth rejection from the connection test (HTTP 401/403)**. Bedrock additionally fails on AWS identity/credential-chain errors.
- **Warning** (`add_warning(msg)`) — non-fatal or inconclusive: unusually short key, missing/defaulted API version, and **inconclusive connection tests** (timeout, DNS/TLS/connection refused, or a non-200/401/403 status). These don't flip `valid` because the run may still succeed.

The connection test is best-effort and skippable via `CODEREVIEW_SKIP_CONNECTION_TEST=1`. The 401/403→hard-fail rule applies to every provider that runs a connection test (Azure, NVIDIA); providers without one (DeepSeek, Moonshot, Z.AI, OpenAI-on-Bedrock) validate key presence + HTTPS only and defer auth verification to the first call. When adding a provider with a connection test, follow this same mapping.

**Canonical-owner convention** for the model registry: when the same model is exposed by both a vendor's direct API and a re-hoster (Bedrock/NVIDIA/Azure), the **direct API owns the canonical aliases**. E.g. `deepseek-v4-pro` routes to DeepSeek direct, not NVIDIA's free re-host (`dsv4-nvidia`); `kimi` and `kimi-k2.6` route to Moonshot direct, not Bedrock's K2.5 (`kimi-bedrock`) or NVIDIA's K2.6 (`kimi-nvidia-26`). Re-host entries keep provider-suffixed aliases only.

## Adding things

**New model:** add an entry under the matching provider in `codereview/config/models.yaml`. Fields: `id`, `full_id` (provider's identifier), `name`, `aliases`, `pricing.input_per_million`/`output_per_million`, `inference_params`, `context_window`. Use immediately: `codereview ./src --model <id-or-alias>`. Existing entries in `models.yaml` are the best reference.

**New provider:** subclass `ModelProvider` in `codereview/providers/`, implement `analyze_batch`, `_create_model`, `_create_chain`, `_extract_token_usage`, `_is_retryable_error`, `_calculate_backoff`, `validate_credentials`. Register in `ProviderFactory.create_provider` (and add the `<Name>Config` Pydantic class to `config/models.py`, plus a parsing branch in `config/loader.py`). Add the env-var to `cli.py`'s Provider Setup table. **Reference implementations**:
- OpenAI-compatible vendor with dedicated langchain package → mirror `providers/deepseek.py` (uses `ChatDeepSeek`, `api_base` parameter)
- OpenAI-compatible vendor without dedicated package → mirror `providers/zai.py` (uses `ChatOpenAI` with custom `base_url`); for a reasoning model on an OpenAI-compatible endpoint that needs the Responses API, mirror `providers/bedrock_openai.py` (adds `use_responses_api` + temperature/top_p opt-out)
- BaseChatOpenAI subclass with vendor-specific quirks → mirror `providers/moonshot.py` (uses `ChatMoonshot`, accepts `base_url` alias)
- Tool-use-less endpoints → mirror Bedrock's prompt-based JSON path (set `_use_prompt_parsing=True`, append `PydanticOutputParser` to the chain, inject format instructions via `_system_prompt_with_format_instructions`).

**Provider contract — public API vs internal hooks.** When implementing a provider, know which methods callers invoke versus which the base class calls into:

| Method | Role | Notes |
|---|---|---|
| `analyze_batch` | **public** | The single entry point `CodeAnalyzer` calls. Build `chain_input`, then delegate to `_execute_with_retry` (don't reimplement the retry loop). |
| `validate_credentials` | **public** | Called by `--validate`. Follow the hard-fail vs warning contract above. |
| `get_pricing` / `get_model_display_name` | **public** | Used by cost reporting and the renderer; `get_pricing` is mandatory for every provider. |
| `_create_model` / `_create_chain` | **hook (required)** | Build the LangChain client and chain. Enforce HTTPS here via `require_https` (fail closed before any network call). Set `_use_prompt_parsing` for tool-use-less models. |
| `_extract_token_usage` | **hook (required)** | OpenAI-compatible providers should delegate to `extract_openai_token_usage` (mixins.py). |
| `_is_retryable_error` / `_calculate_backoff` | **hook (required)** | OpenAI-compatible providers should use `is_openai_retryable_error` + `parse_retry_after` (mixins.py); keep any provider-specific base-wait local (see Azure). |
| `_execute_with_retry`, `_prepare_batch_context`, `_build_batch_system_prompt`, `_resolve_temperature`, `_build_rate_limiter` | **base-provided** | Inherited from `ModelProvider`; call them, don't override unless you have a specific reason. |

Don't add shared mutable state to a provider without a lock (see the concurrency gotcha).

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
- **Reasoning models** (Claude Opus 4.8, Claude Opus 4.7, GPT-5.4, GPT-5.4 Pro, DeepSeek-V4-Pro) don't accept `temperature`/`top_p`. Bedrock and Azure providers both pass `allow_none=True` to `_resolve_temperature`; omit `default_temperature` from `inference_params` for new reasoning models.
- **MiniMax M2.5 on Bedrock, DeepSeek-V4-Pro on Azure, Kimi K2.6 on Moonshot, Claude Opus 4.7/4.8 on Bedrock, and GLM-5.1 on Z.AI** lack usable tool-based structured output → use prompt-based JSON parsing via `supports_tool_use: false`. GLM-5.1's case: Z.AI's OpenAI-compat endpoint ignores OpenAI's `json_schema` response_format that `.with_structured_output()` sets and returns markdown-fenced JSON (```` ```json … ``` ````), which the json_schema parser rejects with "Invalid JSON: expected value at line 1 column 1"; `PydanticOutputParser` strips the fences. The providers branch in `_create_model` and append a `PydanticOutputParser` to the chain. Two of these are "can tool-call but not while thinking" cases, not "no tool-call at all": **K2.6** — Moonshot's server rejects `tool_choice='specified'` (HTTP 400) when thinking is enabled; **Opus 4.7/4.8** — these support only `thinking.type: "adaptive"` and engage thinking server-side per request, and Anthropic allows only `tool_choice: auto/none` while thinking, so a forced `tool_choice` returns the tool call as **literal text** (`<invoke name="issues">…`) → `CodeReviewReport.issues` fails with a Pydantic `list_type` error on the batches where the model thinks (intermittent). `.with_structured_output()` sets exactly that forced `tool_choice`, so we route around it. Opus 4.6 is NOT adaptive-thinking and keeps `supports_tool_use: true`.
- **`use_responses_api: true`** for GPT-5.x in `models.yaml` — ChatCompletion API does not support reasoning summaries for these.
- **Concurrent batches:** `TokenTrackingMixin._track_tokens` and `CodeAnalyzer.skipped_files` are lock-guarded. Don't add other shared mutable state to providers without a lock.
- **`--list-models`** shows everything regardless of credentials; credentials are only validated when a model is actually used.
- **DeepSeek-V4-Pro on Azure / SGLang null-model bug**: the Foundry endpoint validates `body.model` strictly. langchain-openai's `AzureChatOpenAI` defaults `model_name=None` and serializes `"model": null`, which real Azure-OpenAI ignores but SGLang rejects with HTTP 400. The Azure provider explicitly sets `model=deployment_name` to satisfy both backends.
- **Moonshot has two platforms**: `platform.moonshot.cn` (Chinese, default in our YAML) and `platform.moonshot.ai` (international). Keys are NOT interchangeable. `KIMI_API_KEY` typically maps to `.cn`; users with `.ai` keys must override `base_url` in the moonshot section.
- **OpenAI-on-Bedrock is NOT the `bedrock` provider**: GPT-5.5/GPT-5.4 on Bedrock go through Bedrock's *OpenAI-compatible* endpoint, which authenticates with an Amazon Bedrock **API key (bearer token)** via `ChatOpenAI` + `base_url` — not the SigV4 `ChatBedrockConverse` path. It lives in the separate `bedrock_openai` provider. Underlying transport is the `openai` SDK (already pulled by `langchain-openai`; no new dep). The `bedrock_openai` model entries' `full_id` is a **literal**, not `${BEDROCK_OPENAI_MODEL_ID}` — an unset env var expands to `""` and fails `full_id`'s `min_length=1`, breaking `--list-models`; paste the wire id from the console instead. These are reasoning models (Responses API via `use_responses_api: true`, no temperature/top_p) and use `supports_tool_use: false` — **verified against the live endpoint**: GPT-5.5/5.4 engage adaptive server-side thinking per request, and on think-heavy batches return a reasoning-only response (`tool_calls=[]`, no `parsed` field → "Structured Output response does not have a 'parsed' field"), which breaks the forced `tool_choice` that `.with_structured_output()` sets. Intermittent (only the batches where it thinks). Same failure mode as Opus 4.7/4.8 on Bedrock, so they route through prompt-based JSON parsing. Note GPT-5.4 on *Azure* keeps `supports_tool_use: true` — that deployment doesn't exhibit this; the Bedrock OpenAI-compatible endpoint does.
- **Determinism for static analysis**: when `MAX_FILES_PER_TOOL=500` truncation triggers, file lists must be `sorted(...)[:N]`. Locked in by `tests/test_static_analysis.py::test_truncation_is_deterministic`. `MAX_FILES_PER_TOOL`'s docstring documents this as a design guarantee.
- **`--tool-timeout`** plumbs through to `subprocess.run(timeout=...)` for static-analysis tools (default 120s). `--include-hidden` opts into `.github/scripts/` etc. Both are off the FileScanner / StaticAnalyzer constructors as well, not just the CLI.
