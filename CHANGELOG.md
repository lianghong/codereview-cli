# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### New Providers (3)
- **DeepSeek direct API** — 6th provider, via dedicated `langchain-deepseek`
  package (small single-purpose dep, not the heavy langchain-community)
  - Models: `deepseek-v4-pro` (1M context, $1.74/$3.48 per M),
    `deepseek-v4-flash` (1M context, $0.14/$0.28 per M)
  - Aliases: `dsv4-pro`, `ds-v4-pro`, `dsv4-flash`, `ds-v4-flash`
  - Reads `${DEEPSEEK_API_KEY}`; base URL `https://api.deepseek.com`
  - Native tool calling and structured output (no prompt-based JSON fallback)
  - **Naming cleanup**: NVIDIA's `deepseek-v4-pro-nvidia` lost aliases
    `deepseek-v4`, `deepseek-v4-pro`, `dsv4` (now claimed by direct API as
    canonical owner); kept as `ds-v4-nvidia`, `dsv4-nvidia`,
    `deepseek-v4-nvidia`. Azure entry's collision comment also updated.
- **Z.AI (Zhipu international)** — 5th provider, via OpenAI-compatible adapter
  (`ChatOpenAI` + custom `base_url`, no langchain-community heavy dep)
  - Model: `zhipuai/glm-5.1` (203K context, $1.40/$4.40 per M, function calling)
  - Aliases: `zai-glm`, `glm-zai`, `glm5.1-zai`
  - Reads `${ZAI_API_KEY}`; base URL `https://api.z.ai/api/paas/v4/`
- **Moonshot AI (Kimi)** — 7th provider, via dedicated `langchain-moonshot`
  package (extends `BaseChatOpenAI`)
  - Model: `kimi-k2.6` (1T MoE, 32B active, 256K context, $0.60/$2.50 per M)
  - Aliases: `kimi`, `kimi26`, `kimi-moonshot`
  - Reads `${KIMI_API_KEY}`; base URL `https://api.moonshot.cn/v1` (Chinese
    platform; override to `https://api.moonshot.ai/v1` for international keys)
  - **Naming cleanup**: NVIDIA's `kimi-k2.6-nvidia` lost aliases `kimi-k2.6`
    and `kimi26` (claimed by direct provider); kept as `kimi-nvidia-26`,
    `kimi26-nvidia`. Bedrock's `kimi-k2.5-bedrock` lost the bare `kimi`
    alias (now routes to canonical Moonshot K2.6); kept as `kimi-bedrock`,
    `kimi25-bedrock`.

#### CLI Features
- **`--tool-timeout`** — override the static-analysis subprocess timeout
  (default 120s, range 1–3600). Useful for `cppcheck --enable=all` on
  large C++ repos and `mypy` strict on big Python codebases.
- **`--include-hidden`** — opt-in scanning of `.github/scripts`, `.config/`,
  and other dotfile directories. Default behavior (skip hidden) is unchanged.
- **Python version check at package import** — `RuntimeError` raised for
  Python < 3.14 before sub-modules parse. Avoids confusing `SyntaxError`
  from PEP 758 leaf modules when run under a wrong venv.

#### Reliability and reproducibility
- **Deterministic file truncation** — when `MAX_FILES_PER_TOOL` (500)
  triggers, file lists are sorted before slicing. Previously the analyzed
  subset depended on filesystem walk order; now CI runs are reproducible.
  Locked in by a regression test that was confirmed to fail without the sort.
- **Worker-thread traceback context** — `_render_batch_error` now uses
  `traceback.format_exception(type(e), e, e.__traceback__)` instead of
  `format_exc()`. Inside the `ThreadPoolExecutor.future.result()` re-raise
  path the latter only shows the re-raise frame, hiding the real failure.
- **`StaticAnalyzer` fail-fast directory validation** — invalid paths
  raise `ValueError` at construction instead of returning per-tool errors
  on every `run_tool` call.
- **`run_all` systemic-failure log** — when every tool errors with no
  output, log an `ERROR` distinguishing infrastructure failure from
  code-quality findings (CI pipelines reading `passed` can now tell the
  difference).
- **`skipped_oversized` accumulation fix** — `FileBatcher.create_batches`
  now resets the list at the top of each call. Reusing a batcher
  instance no longer conflates skip lists across runs.
- **Markdown export now includes batcher-skipped files** — files dropped
  for exceeding the per-batch token budget are surfaced in the report's
  `skipped_files` section alongside scanner and analyzer skips. Previously
  they were only printed to terminal.
- **Provider validation: differentiated httpx errors** —
  `TimeoutException`, `ConnectError`, `HTTPStatusError` get specific
  messages (DNS / TLS / refused / status code) instead of a generic
  catch-all. The strict-redaction guarantee for the catch-all is preserved:
  no `str(e)` for the generic branch where the message can carry api-key
  headers.

#### Security hardening
- **Static-analysis tool resolution** — already shipped via `shutil.which`
  but the `gofmt` cache fallback bypassed the check. Fixed: when a tool's
  primary executable (`gofmt`) differs from its version-check executable
  (`go`) and can't be safely resolved on PATH, the tool is excluded from
  `available_tools` instead of caching the bare name.
- **AWS credential validation redaction** — three paths in `bedrock.py`
  no longer surface raw AWS error messages or `str(e)` to users. STS and
  Bedrock model-list errors now report only the AWS error code; generic
  exceptions report only the class name. Mirrors prior Azure/NVIDIA work
  to prevent IAM/SCP fragments from leaking into user terminals.

#### Observability
- **npm-audit JSON parse fallback log** — `logging.warning` with output
  preview when JSON parsing falls through to line counting. Without this,
  an HTML proxy error page would silently inflate "issue count" to one-
  per-line.
- **ValidationError retry log** — `logging.debug` when output-fixing
  burns a retry, with attempt counter and parse error. Helps operators
  diagnose why retries get exhausted.
- **Renderer markdown: drop misleading hardcoded tool list** — the
  static-analysis section previously claimed "ruff, mypy, black, isort
  (when available)" regardless of what actually ran (the codebase now
  supports 19+ tools across 6 languages).

### Fixed
- **DeepSeek-V4-Pro on Azure: SGLang `null` model body** — the Foundry
  endpoint is served by SGLang, which validates `body.model` as a
  required string. langchain-openai's `AzureChatOpenAI` defaults
  `model_name=None` and serializes `"model": null`, which real Azure-OpenAI
  ignores but SGLang rejects with HTTP 400. Provider now passes
  `model=deployment_name` explicitly.
- **Moonshot endpoint default** — initially defaulted to
  `https://api.moonshot.ai/v1` (international); flipped to
  `https://api.moonshot.cn/v1` (Chinese platform) since `KIMI_API_KEY`
  almost always targets the latter. International keys can override per
  models.yaml or via `MOONSHOT_API_BASE`.

### Existing entries below ↓
- **GPT-5.4 (Azure OpenAI)** — frontier reasoning model, 1.05M context, 128K output
  - Model ID: `gpt-5.4`, aliases: `gpt`, `gpt54`, `gpt5.4`
  - Reasoning model: no `temperature`/`top_p`; uses Responses API for reasoning summaries
  - Pricing: $2.50/M input, $0.25/M cached, $15.00/M output (Azure standard tier ≤272K)
  - Now the default Azure model (replaces retired GPT-5.3 Codex)
- **DeepSeek-V4-Pro (Azure OpenAI Foundry)** — 1M context, chain-of-thought reasoning
  - Model ID: `deepseek-v4-pro-azure`, aliases: `dsv4-azure`, `deepseek-v4-azure`, `ds-v4-azure`
  - `supports_tool_use: false` — Foundry doesn't expose tool calling; provider routes
    to prompt-based JSON parsing via `PydanticOutputParser` (matches Bedrock pattern)
  - Pricing: $1.74/M input, $0.174/M cached, $3.48/M output (DeepSeek list price)
- **Azure provider: `supports_tool_use: false` support** — mirrors the Bedrock implementation
  - New `_use_prompt_parsing` flag; `_create_chain` appends `PydanticOutputParser`
  - `analyze_batch` injects format instructions into the system prompt
  - `_resolve_temperature(allow_none=True)` so reasoning models (DeepSeek-V4-Pro,
    GPT-5.4 family) skip `temperature`/`top_p` cleanly
- **Supply-chain hardening for static analysis** — `shutil.which()` resolves tool
  binaries to absolute paths and rejects any binary that resolves inside the
  analyzed directory (defends against `node_modules/.bin/eslint` shadowing)
- **`--include-hidden` CLI flag** — opt-in scanning of `.github/scripts`,
  `.config/`, etc. New `FileScanner.exclude_hidden: bool = True` parameter;
  default behavior unchanged
- **Deterministic file truncation** — `MAX_FILES_PER_TOOL` slicing now sorts
  before `[:N]` for shellcheck/cpp/java/prettier/tsc, so the analyzed subset
  is reproducible across runs (CI quality gates can rely on it)
- **Accurate static-analysis issue counts** — per-tool summary-line regex for
  ruff (`Found N errors.`), mypy (`Found N errors in M files`), and bandit
  (`>> Issue:` markers); falls back to substring counting only for unknown
  tools. Previously the substring scan double-counted: each per-issue line
  AND the summary line both tripped indicators
- **`_safe_rglob_suffixes()` helper** — eslint branch consolidated from 4
  separate rglob loops into one single-pass call (consistent with prettier
  and clang-tidy)
- **Fail-fast directory validation** — `StaticAnalyzer.__init__` now raises
  `ValueError` when the directory is missing/not-a-dir/contains null bytes,
  instead of returning per-tool errors at every `run_tool` call
- **Kimi K2.6 (NVIDIA NIM)** — 1T MoE, 32B active, 262K context, thinking mode
  - Model ID: `kimi-k2.6-nvidia`, aliases: `kimi-k2.6`, `kimi26`, `kimi-nvidia-26`
  - Fixed temperature=1.0 and top_p=0.95 required by Moonshot serving backend
- **Mistral Medium 3.5 128B (NVIDIA NIM)** — dense 128B, 256K context, 77.6% SWE-Bench
  - Model ID: `mistral-medium-nvidia`, aliases: `mistral-medium`, `mistral-medium-3.5`, `mm35`, `mmed`
  - Per-request `reasoning_effort` parameter (`none`/`low`/`medium`/`high`)
  - New `InferenceParams.reasoning_effort` field in the config schema; passed
    through `NVIDIAProvider` via `ChatNVIDIA.model_kwargs`
- **GLM-5.1 (NVIDIA NIM)** — 744B/40B active MoE, 131K context, interleaved thinking
  - Model ID: `glm51`, aliases: `glm-5.1`, `glm51-nvidia`, `glm5.1`
  - Replaces the GLM-5 endpoint deprecated by NVIDIA on 2026-04-20
- **Logged Pydantic coercion drift** — `ReviewIssue.normalize_severity` /
  `normalize_category` now emit a deduplicated warning when an unknown value
  is coerced to the default, surfacing LLM schema drift instead of silently
  absorbing it
- **`_safe_rglob_suffixes()` helper** in `StaticAnalyzer` — single tree walk
  across multiple extensions (C++ went from 5 rglobs to 1, prettier from 8 to 1)

### Changed
- **Default Azure model** is now `gpt-5.4` (was `gpt-5.3-codex`, now removed).
  `defaults.azure_default` updated in `models.yaml`. The `gpt` short alias
  routes to GPT-5.4.
- **API-key redaction in connection-test paths** — Azure and NVIDIA providers
  now surface `type(e).__name__` only (never `str(e)`), since lower layers
  (httpx, urllib3) can include the `Authorization` header or URL-encoded key
  variants in error messages. The earlier prefix-scrub heuristic was fragile.
- **Thread-safe warning dedup** in `models.py` — `_warn_once` wraps the
  check-then-add in a `threading.Lock`. Validators run inside provider
  threads (`ThreadPoolExecutor` in `cli.py`); without the lock two threads
  could both pass the membership test and emit the same warning twice.
- **Google GenAI provider wiring** — `rate_limiter` is now actually passed
  to `ChatGoogleGenerativeAI` (was constructed in `__init__` but unused),
  and `google_api_key` is wrapped in `pydantic.SecretStr` to match Azure
  and NVIDIA providers.
- **Exception chaining** in `ConfigLoader._load_config` — `raise ValueError(...)
  from e` for `FileNotFoundError`, `yaml.YAMLError`, and `PermissionError`
  (preserves traceback chain per PEP 3134).
- **GLM-5 (NVIDIA)** marked deprecated in YAML and docs; NVIDIA deprecated
  the `z-ai/glm5` endpoint on 2026-04-20. Entry kept until NVIDIA fully
  removes the endpoint so existing `--model glm5` invocations keep working.
- **Provider boilerplate consolidated** — `ModelProvider` base class gained
  `_resolve_temperature()`, `_build_rate_limiter()`, and
  `_system_prompt_with_format_instructions()` helpers plus concrete-default
  `get_model_display_name()` / `get_pricing()`. Each of the four providers
  (Bedrock, Azure OpenAI, NVIDIA, Google GenAI) dropped the duplicate
  implementations.
- **Env var expansion** in `ConfigLoader` now warns once per missing variable
  (deduplicated) with a clearer message pointing at provider-registration
  impact.
- **Line counting in CLI** switched to chunked binary newline counting
  (~2-3× faster than UTF-8 decode-and-iterate) in `cli.py`.

### Fixed
- **AWS Bedrock error messages** no longer hardcode "Claude Opus 4.6" —
  access-denied / model-access troubleshooting now names the actual resolved
  model. Falls back to the raw `--model` argument if resolution itself failed.
- **Invalid `--exclude` patterns** are now named individually when rejected,
  so users can identify and fix typos instead of seeing a generic warning.
- **Azure API key redaction** in `validate_credentials` connection-error path
  now scrubs 16-char key prefixes in addition to full-key matches.
- **Rich callback cleanup** — removed `__del__` finalizers in
  `StreamingCallbackHandler` / `ProgressCallbackHandler`. The CLI already
  calls `cleanup()` in its `finally` block; the destructors produced noisy
  tracebacks during interpreter shutdown when Rich internals were already
  torn down.
- **`next()` / `StopIteration` pattern** in ESLint file discovery replaced
  with `next(gen, None)` sentinel form.

### Removed
- **12 model entries** pruned from `models.yaml`:
  - **AWS Bedrock**: DeepSeek-R1, DeepSeek V3.2, MiniMax M2.1, GLM 4.7,
    GLM 4.7 Flash
  - **Azure OpenAI**: GPT-5.3 Codex, Grok 4 Fast Reasoning
  - **NVIDIA NIM**: DeepSeek V3.2, GLM 4.7, MiniMax M2.1, MiniMax M2,
    Devstral 2 123B
  - Doc references in `README.md`, `docs/usage.md`, `docs/examples.md`,
    and `CLAUDE.md` updated to point at surviving equivalents
  - `tests/test_config.py` aliases assertion updated from `minimax-nvidia`
    (deleted) to `mistral-medium-nvidia`
- **Dead duplicate severity-count fields** (`critical`, `high`, `medium`,
  `low`, `info`) from `ReviewMetrics`. They were populated by `cli.py` but
  never read anywhere in the codebase or tests. Canonical `*_issues` fields
  are unchanged.
- **Obsolete release-note files** — `RELEASE_NOTES_v0.3.0.md` (363 lines)
  and `RELEASE_NOTES_v0.3.1.md` (328 lines). Content is fully covered by
  this CHANGELOG.
- **`docs/MIGRATION.md`** (321 lines) — v0.1.x → v0.2.0 migration guide,
  long obsolete.
- **Per-release history block** from `README.md` (~112 lines) — replaced
  with a 3-line pointer to this CHANGELOG as the single source of truth.
- **Shipped-feature planning docs** in `docs/plans/` (~1,100 lines) — the
  readme-context feature they designed landed in `codereview/readme_finder.py`
  long ago.

### Documentation
- Trimmed overall documentation footprint from ~5,600 to ~4,500 lines
- Updated `CLAUDE.md` and `README.md` to v0.3.1-current state: correct
  default model (Opus 4.7), correct test count (319), new model tables
- Added `.ruff_cache/` to `.gitignore` to match existing cache ignores

### Quality
- Test suite: **368 passing** (up from 319; +2 scanner `exclude_hidden`,
  +3 supply-chain `_resolve_tool_binary`, +5 ruff/mypy/bandit counters,
  +2 Azure `supports_tool_use=false`, +9 Z.AI provider, +11 DeepSeek
  provider, +10 Moonshot provider, +5 truncation/timeout/batcher resets);
  two pre-existing fixtures fixed — they passed kwargs that Pydantic
  silently dropped
- `ruff check`, `ruff format --check`, `mypy`: clean
- New runtime dependencies: `langchain-deepseek>=1.0.1`,
  `langchain-moonshot>=0.1.0` (both small single-purpose packages, not
  the heavy `langchain-community`)

## [0.3.1] - 2026-04-18

### Added
- **Claude Opus 4.7** support via AWS Bedrock (`us.anthropic.claude-opus-4-7`)
  - Latest reasoning model with adaptive thinking capability
  - Max output tokens: 32,000
  - Model ID: `opus4.7`, aliases: `claude-opus-4.7`, `opus-4.7`, `claude-opus-47`
  - Available in US East (N. Virginia) and Asia Pacific (Tokyo)
  - Reasoning model - does not support temperature parameter
  - Automatically configured as the new default model
- **PEP 758 clarification comments** to exception handlers (8 locations)
  - Added comments explaining Python 3.14+ unparenthesized multi-exception syntax
  - Prevents confusion for contributors unfamiliar with PEP 758
  - Files: `callbacks.py` (2), `azure_openai.py` (1), `readme_finder.py` (3), `static_analysis.py` (2)

### Fixed
- **--no-color flag consistency** across all CLI commands
  - `--list-models` now respects `--no-color` flag (outputs plain text without ANSI codes)
  - `--validate` now respects `--no-color` flag
  - Removed module-level `Console()` instance that ignored user flags
  - Console instance now created early in main() and passed to all helper functions
  - Affected functions: `display_available_models()`, `validate_provider_credentials()`
- **Bedrock provider temperature handling** for reasoning models
  - Conditionally omits temperature parameter for models that don't support it
  - Prevents `ValidationException: temperature is deprecated for this model` errors
  - Dynamically detects when model config omits temperature (reasoning models)
  - Builds model kwargs dict and only includes temperature when appropriate

### Changed
- **Default model**: Updated from Claude Opus 4.6 to Claude Opus 4.7
  - Set in `models.yaml`: `bedrock_default: opus4.7`
  - Users can still explicitly use Opus 4.6 with `--model opus`
- **Dependencies**: Updated all packages to latest versions
  - Core: `langchain>=1.2.15`, `langchain-aws>=1.4.4`, `langchain-openai>=1.1.14`, `langchain-nvidia-ai-endpoints>=1.2.1`, `langchain-google-genai>=4.2.2`
  - Providers: `boto3>=1.42.91`, `google-api-core>=2.30.3`, `google-genai>=1.73.1`
  - UI: `click>=8.3.2`, `rich>=15.0.0`
  - Data: `pydantic>=2.13.2`
  - Dev tools: `pytest>=9.0.3`, `ruff>=0.15.11`, `mypy>=1.20.1`, `black>=26.3.1`, `vulture>=2.16`
  - All transitive dependencies updated
- **Documentation updates**:
  - CLAUDE.md: Added "Recent Updates" section with v0.3.1 changes
  - README.md: Added "What's New" section highlighting Opus 4.7
  - Updated model tables with correct Opus 4.7 model ID and parameters
  - Added notes about reasoning model characteristics (no temperature support)

### Technical Details
- **Files modified**: 9 files
  - Configuration: `models.yaml`, `CLAUDE.md`, `README.md`, `CHANGELOG.md`
  - Code: `cli.py`, `bedrock.py`, `callbacks.py`, `azure_openai.py`, `readme_finder.py`, `static_analysis.py`
- **Quality metrics**:
  - All 311 tests passing (100%)
  - Zero security vulnerabilities (Bandit: 5,656 lines scanned)
  - Zero linting issues (Ruff)
  - Zero type errors (Mypy: 22 source files)
  - Zero dead code (Vulture at 80% confidence)
- **Testing verified**:
  - ✅ `codereview --list-models` (shows Opus 4.7)
  - ✅ `codereview --list-models --no-color` (no ANSI codes)
  - ✅ `codereview --model opus4.7 --dry-run` (validates successfully)
  - ✅ `codereview /path/to/code --model opus4.7` (analyzes successfully)

### Migration Guide

**No breaking changes.** All existing functionality continues to work.

To use Claude Opus 4.7 (new default):
```bash
codereview ./src                    # Uses Opus 4.7 automatically
codereview ./src --model opus4.7    # Explicit
```

To continue using Claude Opus 4.6:
```bash
codereview ./src --model opus       # Opus 4.6
```

### References
- [AWS Blog: Introducing Claude Opus 4.7](https://aws.amazon.com/blogs/aws/introducing-anthropics-claude-opus-4-7-model-in-amazon-bedrock/)
- [PEP 758: Allow except without parentheses](https://peps.python.org/pep-0758/)

---

## [0.3.0] - 2026-03-31

### Added
- **MiniMax M2.5 (Bedrock)**: Agent-native frontier model via AWS Bedrock
  - Model ID: `minimax.minimax-m2.5`
  - Aliases: `minimax-m2.5-bedrock`, `mm2.5-bedrock`
  - Architecture: MoE (230B total, 10B active parameters)
  - Context: 196K tokens, Max output: 128K tokens
  - 80.2% SWE-Bench Verified, 37% faster than M2.1
  - Temperature: 0.5 (optimized for code review without thinking mode)
  - Optimized for task decomposition and complex workflows
- **GLM 5 (Bedrock)**: Frontier-class model for systems engineering via AWS Bedrock
  - Model ID: `zai.glm-5`
  - Aliases: `glm5-bedrock`, `glm-5-bedrock`, `glm5b`
  - Context: 200K tokens, Max output: 128K tokens
  - Temperature: 0.5 (per Zhipu AI recommendations for structured tasks)
  - Optimized for complex systems engineering and long-horizon agentic tasks
  - Multi-step reasoning, AIME-style math, advanced coding, tool-augmented workflows

### Fixed
- **Critical: Azure Provider Syntax Error**: Fixed Python 2 style exception handling (`except ValueError, TypeError:` → `except (ValueError, TypeError):`) that completely blocked Azure OpenAI provider functionality
- **Security: ReDoS Prevention**: Added input validation for user-provided `--exclude` patterns to prevent Regular Expression Denial of Service attacks
  - Max pattern length: 200 characters
  - Max `**` recursion depth: 3
  - Disallow null bytes and malicious patterns
  - Invalid patterns are filtered with warning message

### Changed
- **Model Configuration**: Updated `models.yaml` with comprehensive parameter documentation
  - Added detailed rationale for temperature settings (MiniMax M2.5: why Bedrock uses 0.5 vs NVIDIA's 1.0)
  - Documented thinking mode availability differences between providers
  - Architecture specifications and capability tags for model selection
- **Documentation**: Updated CLAUDE.md with new model information
  - Added MiniMax M2.5 and GLM 5 to model lists and pricing tables
  - Updated supported models count (109 total models)
  - Enhanced parameter documentation with cross-provider comparisons
- **Python 3.14 Compliance**: Adopted PEP 758 unparenthesized exception syntax
  - Updated 7 exception handlers to use modern syntax: `except E1, E2:` instead of `except (E1, E2):`
  - Files updated: `callbacks.py` (2), `readme_finder.py` (3), `static_analysis.py` (2)
  - Correctly retained parentheses for exception handlers using `as` clause (required by PEP 758)
  - Verified PEP 765 compliance: no control flow issues in `finally` blocks
  - All 311 tests pass with Python 3.14.2

### Technical Details
- Total models: 109 (up from 107)
- All 311 tests passing
- Zero static analysis issues (ruff, mypy, isort, vulture)
- Full backward compatibility maintained

### Provider Comparison
**MiniMax M2.5: Bedrock vs NVIDIA**
| Parameter | Bedrock | NVIDIA | Reason |
|-----------|---------|--------|--------|
| Temperature | 0.5 | 1.0 | Bedrock lacks thinking mode |
| Top-p | 0.95 | 0.95 | Same |
| Context | 196K | 196K | Same |
| Thinking Mode | ❌ | ✅ | Platform limitation |

**GLM 5: Bedrock vs NVIDIA**
| Parameter | Bedrock | NVIDIA | Reason |
|-----------|---------|--------|--------|
| Temperature | 0.5 | 0.5 | Model docs recommendation |
| Top-p | 0.95 | 0.95 | Same |
| Context | 200K | 200K | Same |

### Notes
- MiniMax M2.5 and GLM 5 Bedrock pricing TBD (awaiting AWS publication)
- Parameter research based on AWS blog announcement, NVIDIA configurations, and model documentation
- Temperature differences for MiniMax M2.5 are architectural (thinking mode availability), not arbitrary

## [0.2.9] - 2026-03-20

### Added
- **Mistral Small 4 Model**: Added Mistral Small 4 119B via NVIDIA NIM
  - MoE architecture with 256K context, 16K max output
  - Prompt-based JSON parsing (no tool use support)
- **MiniMax M2.5 Model (NVIDIA)**: Added MiniMax M2.5 via NVIDIA NIM
  - 80.2% SWE-Bench Verified
  - 192K context, 128K output
  - Interleaved thinking mode
  - 37% faster than M2.1
- **Prompt-Based JSON Parsing**: Fallback parsing for models without tool use support
  - DeepSeek-R1, Mistral Small 4
  - Maintains structured output reliability

### Fixed
- **Oversized File Handling**: Files exceeding token budget now skipped with warning instead of creating doomed batches
- **Batch Failure Handling**: Clear error messages when all batches fail (rate limits, auth errors)
  - Partial failures now warn that results are incomplete
  - No more misleading "0 issues found" reports
- **Grok 4 Fast Context Fix**: Corrected context window from 2M to 128K to match Azure deployment limit

### Improved
- **Retry Backoff**: Enhanced retry logic for Azure OpenAI and Google GenAI providers
  - 5 retries with longer backoff (10s/20s/40s/60s/60s progression)
  - Total wait time: ~190 seconds
  - Azure respects `Retry-After` headers
- **Plain Text Suggestions**: Improvement Suggestions section renders as plain text without box-drawing characters for clean copy-paste

### Upgraded
- **Dependencies**: Updated to latest versions
  - langchain-aws 1.3.1
  - langsmith 0.7.7
  - google-genai 1.65.0
  - openai 2.24.0
  - websockets 16.0
  - isort 8.0.0
  - ruff 0.15.4
  - mypy 1.19.1

### Testing
- 311 tests passing

## [0.2.8] - 2026-03-15

### Added
- Token-budget-aware batching for efficient context window utilization
- Step 3.5 Flash model via NVIDIA NIM
- GLM-5 model via NVIDIA NIM

### Changed
- Improved file batching logic with token estimation
- Enhanced error messages for provider issues

## [0.2.7] - 2026-03-10

### Added
- Qwen3.5 397B model support via NVIDIA NIM
- Kimi K2.5 model support (Bedrock, Azure, NVIDIA)
- DeepSeek V3.2 model support (Bedrock, NVIDIA)

### Fixed
- Provider credential validation improvements
- Rate limit handling for multiple providers

## [0.2.6] - 2026-03-05

### Added
- Google Generative AI provider (Gemini 3.1 Pro, Gemini 3 Pro, Gemini 3 Flash)
- `--no-color` flag for copy-paste friendly output
- README context discovery with auto-confirmation

### Changed
- Enhanced structured output with `method="json_schema"` for Google GenAI
- Improved retry logic with adaptive backoff

## [0.2.5] - 2026-02-28

### Added
- NVIDIA NIM provider support
- Devstral 2 123B model (72.2% SWE-Bench Verified)
- MiniMax M2, M2.1 models via NVIDIA
- GLM 4.7 model via NVIDIA

### Changed
- Parallel static analysis execution for faster performance
- Improved token estimation for batching

## [0.2.4] - 2026-02-20

### Added
- Azure OpenAI provider support
- GPT-5.3 Codex and GPT-5.4 Pro models
- Grok 4 Fast Reasoning model via Azure
- Responses API support for GPT models

### Fixed
- Rate limit handling for Azure OpenAI
- Retry-After header respect

## [0.2.3] - 2026-02-15

### Added
- Qwen3 Coder 480B model (Bedrock)
- DeepSeek-R1 model (Bedrock)
- MiniMax M2.1 model (Bedrock)

### Changed
- Enhanced prompt engineering for code review
- Improved category normalization

## [0.2.2] - 2026-02-10

### Added
- GLM 4.7 and GLM 4.7 Flash models (Bedrock)
- Context window configuration per model
- Token budget calculation with safety margins

## [0.2.1] - 2026-02-05

### Fixed
- Pydantic V2 compatibility issues
- Category validation for non-Claude models

### Changed
- Enhanced error messages with actionable suggestions

## [0.2.0] - 2026-02-01

### Added
- Multi-provider support (AWS Bedrock foundation)
- Claude Opus 4.6, Sonnet 4.6, Haiku 4.5
- Structured output with Pydantic V2
- Rich terminal UI
- Markdown and JSON export
- Static analysis integration

### Changed
- Migrated from direct Anthropic API to LangChain
- Improved batching logic
- Enhanced retry mechanisms

## [0.1.0] - 2026-01-15

### Added
- Initial release
- Basic code review functionality with Claude
- Python, Go, Shell Script support
- Terminal output
- File scanning and filtering

---

## Release Notes Format

Each release includes:
- **Added**: New features and capabilities
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future releases
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes and corrections
- **Security**: Security vulnerability fixes

## Versioning Strategy

- **Major version (X.0.0)**: Breaking changes, major architectural updates
- **Minor version (0.X.0)**: New features, model additions, provider additions
- **Patch version (0.0.X)**: Bug fixes, security patches, documentation updates

---

**Maintained by:** lianghong  
**Repository:** https://github.com/lianghong/codereview-cli  
**License:** MIT
