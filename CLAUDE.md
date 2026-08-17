# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

LangChain-based CLI for AI code review across **8 providers**: AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google GenAI, DeepSeek direct, Z.AI (Zhipu international), Moonshot (Kimi), and OpenAI-on-Bedrock (GPT-5.x **and xAI Grok** via Bedrock's OpenAI-compatible `bedrock-mantle` endpoint). Reviews **Python, Go, Shell, C++, Java, JS, TS** with structured output (severity, line numbers, suggested fixes).

**Stack:** Python 3.14, LangChain (1.3+), Pydantic V2, Click, Rich, AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google GenAI, DeepSeek (`langchain-deepseek`), Z.AI (`langchain-openai` + custom base_url), Moonshot (`langchain-moonshot`).

For the live model list with pricing/aliases run `uv run codereview --list-models` — that output is authoritative; the YAML in `codereview/config/models.yaml` is the source of truth. Default model: **Claude Opus 5**.

## Deep-dive docs

This file holds the **invariants** — the rule and the prohibition. The reproduction, the exact
wrong numbers, and why the obvious fix is also wrong live in `docs/`. Read the relevant one
before changing that subsystem; a rule here that looks arbitrary is explained there.

| File | Covers |
|---|---|
| `docs/architecture.md` | scanner exclusion, batcher token cache, factory registry, `run_review`, callbacks, Markdown export |
| `docs/providers.md` | method contract, retry classifiers, backoff, token accounting, streaming, per-provider quirks |
| `docs/structured-output.md` | the full tool-use-vs-prompt-parsing matrix, both failure shapes, why `prompt_prefill` was rejected |
| `docs/validation-contract.md` | `--validate` hard-fail vs warning, placeholder keys, model-access scope, the `mixins.py` predicates |
| `docs/model-registry.md` | `models.yaml` conventions: config-key forwarding, aliases, removals, profile-drift cross-check |
| `docs/static-analysis.md` | tool install per language, plus implementation notes (config-execution gate, exit codes) |

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
uv run codereview /path/to/code                           # default: opus5
uv run codereview ./src --model sonnet --output report.md
uv run codereview ./src --static-analysis --severity high
uv run codereview ./src --dry-run                          # preview cost/files
uv run codereview ./src --output report.json --format json # CI-friendly
```

**Quality bar:** ruff (check + format) + isort + mypy + vulture (≥80% confidence) all clean. Type hints on all public APIs. Python 3.14: PEP 758 unparenthesized exceptions, PEP 765 no control flow in `finally`. Every provider must implement `get_pricing()`.

**The ruff rule set is pinned in `pyproject.toml` (`[tool.ruff.lint] select`) and must stay pinned.** Ruff's *default* selection isn't stable across releases and the `ruff>=…` floor resolves without consulting `uv.lock`, so inheriting the default reported 135 errors on an unmodified checkout and the gate could no longer distinguish "this change is clean" from "the toolchain moved". Widening it is fine — in one commit that also lands the resulting fixes, not implicitly by upgrading a tool.

## CLI options

| Option | Description | Default |
|---|---|---|
| `--model, -m` | Model ID or alias (`--list-models` to see) | opus5 |
| `--output, -o` | Export report (md or json) | None |
| `--format, -f` | `markdown` or `json` | markdown |
| `--severity, -s` | Min severity to **display**: critical/high/medium/low/info | info |
| `--fail-on` | Exit 2 if any issue at this severity or above was found (CI gate) | None |
| `--temperature` | 0.0-2.0 | model default |
| `--batch-size` | Max files per batch (file-count cap atop token budget) | 10 |
| `--static-analysis` | Run installed linters in parallel | False |
| `--trust-repo-config` | Run mypy/ESLint/Prettier even when the reviewed repo ships a config that makes them execute code from the tree | False |
| `--dry-run` | Preview without API calls | False |
| `--stream` | Real-time token display; runs batches sequentially. **Ignored (with a notice, keeping concurrency) on Bedrock/NVIDIA/Google**, which never stream a token | False |
| `--verbose, -v` | Detailed progress + token-budget breakdown; with `--list-models`, also spells out deprecated aliases | False |
| `--exclude, -e` | Extra glob patterns | None |
| `--max-files` / `--max-file-size` | File caps | None / 500 KB |
| `--aws-profile` | AWS profile name | None |
| `--readme <path>` / `--no-readme` | README context override | auto-discover |
| `--no-color` | Strip ANSI for paste-friendly output | False |
| `--tool-timeout` | Plumbs to `subprocess.run(timeout=...)` for static-analysis tools | 120s |
| `--include-hidden` | Opt into `.github/scripts/` etc. | False |

`--tool-timeout` and `--include-hidden` are on the `FileScanner` / `StaticAnalyzer` constructors as well, not just the CLI.

## Architecture

```
FileScanner → FileBatcher → CodeAnalyzer → ProviderFactory → {Bedrock|Azure|NVIDIA|GoogleGenAI|ZAI|DeepSeek|Moonshot|BedrockOpenAI}Provider
            → Aggregation (cli.py) → TerminalRenderer / MarkdownExporter
```

- **scanner.py** — discovers code files, applies exclusions, tracks skips with reasons. Exclusion runs at **two** non-interchangeable levels: `_is_excluded` glob-matches each candidate, `_get_excluded_dir_names` feeds an `os.walk` prune set. **A prune name may only come from an *unanchored* pattern** — pruning matches a bare directory name at any depth, so `docs/api/*` contributing `api` also skips an unrelated `app/api/`, silently reviewing less code than the run claims; write a recursive `**` to drop a subtree. **`_is_excluded` must test `PurePath.match` *and* `PurePath.full_match`** — neither alone covers the patterns this project ships. Four pruning tests in `tests/test_scanner.py`. → `docs/architecture.md`
- **batcher.py** — token-budget-aware batching when `context_window` is set, else count-only. Budget = `context_window − max_output − system_prompt − readme − safety_margin`; margin = `clamp(context_window // 10, 1000, 20000)`; estimate = `bytes // 3 + 50` (tiktoken when available, ≤2MB). Greedy packing; oversized files skipped with a warning. Counts memoized on `(path, size, mtime_ns)`: **cache the count, never the content** (concurrent batches — every file's text held live is unbounded memory), and **`_TOKEN_CACHE_SIZE` is a run bound, not a "typical repo" bound** — below the file count the memo degrades to nothing on exactly the large repos it exists for (`test_token_cache_bound_exceeds_a_large_repo_file_count`). `lru_cache` is already thread-safe — don't add a lock. → `docs/architecture.md`
- **analyzer.py** — orchestration. Delegates to provider; tracks `skipped_files` (lock-guarded for concurrent batches).
- **providers/factory.py** — auto-detects provider from model name. Dispatch is a **table, not an if/elif chain**: `_PROVIDER_REGISTRY` maps each provider to `(config_type, module, class_name)` and derives the guard, the lazy import, the constructor call and the error list, so a new provider is one row. The module/class are **strings** to keep imports lazy — an eager registry pulls all eight vendor client packages into every run, including `--list-models`. Two tests in `tests/test_factory_smoke.py` keep the table honest.
- **providers/{bedrock,azure_openai,nvidia,google_genai}.py** — share `ModelProvider` ABC with template-method hooks: `_is_retryable_error`, `_calculate_backoff`, `_extract_token_usage`. Token tracking via `TokenTrackingMixin` (lock-guarded `+=`).
- **cli.py** — `main()` is Click parsing plus the three flags that exit before any review work (`--list-models`, `--validate`, no-directory help); everything from scanning to the `--fail-on` gate lives in **`run_review(directory, *, console, ...)`**, callable without `CliRunner.invoke`. Batches run concurrently (`ThreadPoolExecutor`, ≤4 workers); `--stream` and single-batch runs stay sequential, but `--stream` is downgraded with a notice for a provider that never streams a token. **`run_review`'s keyword-only params mirror the Click options including their defaults** — add a new option to both signatures and the `run_review(...)` call (`test_run_review_defaults_match_the_click_option_defaults`).

### Key patterns

- **Structured output:** `.with_structured_output(CodeReviewReport, include_raw=True)`; `include_raw` is what makes real token counts readable. `supports_tool_use: false` switches to `PydanticOutputParser`, and **the routing lives once in `base.py`** — providers just `return self._apply_structured_output(base_model)`. Three prohibitions: **assume the prompt path for a new reasoning/thinking model until a live run proves tool-use** (the failure is intermittent — only on think-heavy batches); **don't cite Anthropic's docs for the "can't tool-call while thinking" failure** — the vendor documents the opposite for adaptive thinking, so it's observed behavior (`de5e2fc`); **don't swap in `method="prompt_prefill"`** — rejected 2026-07-26, its unconditional backtick stop sequence truncates JSON on any fenced prose. → `docs/structured-output.md`
- **Category normalization** (`models.py`): `@field_validator` maps the varying names non-Claude models emit; unknown → `"Code Quality"`. **A missing category is worse than a mismapped one** — before `Correctness` existed, every bug word coerced to `Code Quality` *and* bumped `category_coerced`, so correct model behavior incremented the prompt-drift counter. Map a new category's synonyms in the same commit and assert the counter stays 0 (`test_correctness_variations_map_without_drift`). `"error handling"` deliberately stays on `Code Quality` — as a bare name it's ambiguous.
- **`--fail-on` is the CI gate, `--severity` is a display filter** — never conflate them. `_evaluate_fail_on` counts `report.issues`, not the rendered subset, or a display preference becomes a silent hole in the gate. Exit **1** = the run failed, **2** = `EXIT_QUALITY_GATE_FAILED`. The gate is the **last** statement in `run_review`, after export, so a failing build still leaves its artifact. Locked by `test_severity_filter_does_not_affect_fail_on` and `test_run_review_applies_the_gate_after_writing_the_report` (which asserts *order*).
- **Classify retryability on the HTTP *status*, not on an exception class** — an `isinstance` against a type the *installed* client never raises is dead code, and dead retry logic is invisible: a misclassified throttle looks exactly like a lost batch. Two classifiers were dead this way and aborted real 429/503/504s on attempt 1. Provider status sets differ **on purpose** — recorded in `_RETRY_MATRIX`, not normalized (`tests/test_retry_contract.py`). → `docs/providers.md`
- **Retry/backoff:** per-provider, exponential, capped at 60s; Azure honours `Retry-After`, NVIDIA 4s base for 504, Google 10s for 429. `parse_retry_after` reads the header off **any** `APIStatusError`. **`max_retries=None` means "the provider decides"** — resolve with `_resolve_max_retries` (`override > provider_config > provider default`), and **never give `analyze_batch` a concrete signature default**, which makes every provider default dead code (`test_every_provider_analyze_batch_defaults_max_retries_to_none`). **Output-parsing failures retry under `enable_output_fixing`**, and that `except` must name `OutputParserException` explicitly — a `ValueError`, not a `ValidationError`, so it otherwise lands in the non-retryable branch. → `docs/providers.md`
- **Token accounting counts what was billed, not what parsed.** `_extract_token_usage` runs on the raw `AIMessage` even when `parsed` is `None`; the prompt-parsing path needs its own `_track_usage_from_parse_failure` because an `OutputParserException` raises past the message. **Read `usage_metadata` first everywhere** — `response_metadata["token_usage"]` is filled only on Chat Completions, so reading it alone substituted the byte heuristic for every `use_responses_api: true` entry (Azure `gpt-5.4` under-reported ~13x). Swallow accounting failures to `logging.debug`. → `docs/providers.md`
- **Streaming: `streaming=bool(callbacks)` was wrong twice, `--stream` a third time.** Use `wants_token_streaming(callbacks)` — an **override check** on `on_llm_new_token`, not class identity — paired with `openai_stream_params(callbacks)`, since without `stream_usage` a `base_url` client sends no usage chunk and reintroduces the under-reporting above. `supports_token_streaming()` is a **classmethod** (Bedrock/NVIDIA/Google → `False`) that must stay answerable without constructing anything: worker count and which handler to attach are one decision, and both feed the constructor. → `docs/providers.md`
- **Aggregation must collect every list field the model returns** (`run_review`): `issues`, `improvement_suggestions`, `system_design_insights`, `recommendations`. `recommendations` was being dropped in favour of severity/category *counts* naming no file, line or title — the field `SYSTEM_PROMPT` works hardest for. Model text now wins; counts are the **fallback**. Dedup on `_dedupe_design_insights`' normalization, capped at 5. **Add a new model-written list field to the batch loop in the same commit** (four recommendation tests in `tests/test_cli.py`).
- **`files_analyzed` counts what was *sent*, not what was scanned** — `sum(len(batch.files) …)`, because the batcher drops oversized files after the scan; `len(files)` claimed coverage the review doesn't have and `--dry-run` billed the dropped bytes (`test_files_analyzed_excludes_files_the_batcher_dropped`, `test_dry_run_does_not_bill_files_the_batcher_will_drop`).
- **Provider error prose is withheld unless `--verbose`** (`_aws_error_detail`; CWE-209). AWS puts the denying SCP statement, role ARNs and account ids in `Error.Message`, and `str(ClientError)` splices it into CI logs. The **code** is what a user acts on, so it always prints and the message is gated — never render a raw `ClientError` or `error.response[...]["Message"]` unconditionally.
- **One `ProgressCallbackHandler` serves every concurrent batch**, so it's refcounted by `run_id` under a lock with at most one `Status` alive (`callbacks.py`) — using a `set[UUID]`, not an int. Rich's `Console.set_live` *returns a bool* instead of raising, so an overwritten `_status` left a permanently-`_started` `Live` on the stack and corrupted the terminal for the rest of the process. → `docs/architecture.md`
- **Every model-generated string reaching the Markdown export must pass through `balance_code_fences`** (renderer.py) — one unclosed ```` ``` ```` in a prose field swallows every following section, and the export still "succeeds". Applied at `_summary`, `_format_issue` (description + rationale), `_system_design`, `_recommendations`, `_improvement_suggestions`; `suggested_code` gets a *wider* fence. **Wrap any new model-written field here too.** Structural fix, not escaping — deliberately no HTML/Markdown escaping.
- **Markdown export tolerates raw-dict metrics:** `metrics_to_dict` returns `report.metrics` unchanged when it isn't a `ReviewMetrics`, so values may be stringified or `None`. `isinstance(..., int)`-guard before any `:,` format or cost division; locked by the raw-dict tests in `tests/test_markdown_export.py`.
- **Prompt injection defense:** `SYSTEM_PROMPT` instructs the model to treat code AND README content as data, never instructions. Don't add new "trusted" message paths without extending that defense.
- **Parallel static analysis:** `StaticAnalyzer.run_all(parallel=True)` uses `ThreadPoolExecutor`; rglob helpers skip symlinks defensively.
- **Pricing display:** zero-priced models (placeholder for unannounced rates) render `Estimated cost: TBD`, not `$0.0000` (`_is_pricing_tbd`, cli.py).

## Configuration

```
codereview/config/
├── models.yaml   # All models, providers, pricing, inference params, scanning rules
├── models.py     # Pydantic schema (ModelConfig, ProviderConfig, etc.)
├── loader.py     # YAML + ${VAR} env expansion, @lru_cache singleton
└── prompts.py    # SYSTEM_PROMPT — code review behavior lives here
```

**Configurable via `models.yaml` (no code changes):** model registration, `aliases` / `deprecated_aliases`, pricing, inference params, `context_window`, `supports_tool_use`, `use_responses_api`, AWS region (provider-level plus a per-model `region` override for region-restricted Bedrock models), Bedrock `read_timeout`, NVIDIA's provider-level `max_retries`, scanning patterns/extensions, max file size. Two per-model overrides carry reasons worth knowing: fable5 pins `us-east-1` because its `provider_data_share` opt-in is per-region account state, and **any new Bedrock model whose thinking is on without being asked for needs `read_timeout: 1800`** (the non-streaming Converse path emits no bytes until the full response is generated, so think-heavy batches hit `ReadTimeoutError` against the 300s default).

**Secrets via env vars** (expanded with `${VAR}` syntax in YAML): `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `NVIDIA_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `KIMI_API_KEY`, `OPENAI_API_KEY` + `OPENAI_BASE_URL` (OpenAI-on-Bedrock). AWS Bedrock (Converse path) uses the standard credential chain; OpenAI-on-Bedrock uses a Bedrock API key (bearer token) instead.

**Doc-only YAML:** the `defaults:` block (`zai_default`, `bedrock_default`, …) and a model's `capabilities`/`architecture`/`notes` keys are **informational only** — nothing reads them and `ModelConfig` isn't `extra="forbid"`. The CLI's real default `--model` is hardcoded (`opus5`) in `cli.py`.

**`ConfigLoader` must forward every key it parses** (`loader.py`). A key in `models.yaml` but absent from the `<Name>Config` construction is invisible: no error, no warning, and the setting appears to work because the field has a default. `NVIDIAConfig.max_retries` was unreachable this way, and the same hazard shipped **sixteen times** on model-entry `pricing`/`inference_params` keys — an unread *pricing* number is the worst version, because the next reader trusts it. Wire a new knob through `_parse_model_config` **and** the Pydantic model (`test_every_pricing_and_inference_key_in_the_yaml_is_actually_read`); a comment in the YAML is not configuration.

**Registry conventions** — **canonical owner**: a vendor's direct API owns the bare aliases, re-hosts keep provider-suffixed ones. **Generation-neutral aliases** track the current generation: rename the old entry's `id`, don't just add the alias, since `_register_model` is last-write-wins within a provider. **Removal**: verify against the live endpoint, then *migrate* version-neutral names as `deprecated_aliases` and *delete* version-explicit ones. Six guards in `tests/test_config.py` lock these in, including `test_documented_model_names_all_resolve`, which scrapes `--model X` out of `README.md`/`docs/usage.md`/`docs/examples.md`. **`models.yaml` is cross-checked against the partner packages' `_MODEL_PROFILES` tables but never overwritten from them** — neither side is authoritative (`tests/test_model_profile_drift.py`). → `docs/model-registry.md`

## Provider credentials

Env vars per provider are listed under Configuration above; `README.md` has the sign-up links and
per-provider setup walkthroughs. Three things worth knowing here:

- **AWS Bedrock** uses the standard credential chain (or `--aws-profile`) and needs `bedrock:InvokeModel`/`InvokeModelWithResponseStream` *plus* model access in the YAML region.
- **`OPENAI_API_KEY` for OpenAI-on-Bedrock is an Amazon Bedrock API key** (bearer token), *not* an openai.com key, and `OPENAI_BASE_URL` must be a region that actually serves the model.
- **Tests mock at the provider level** — no credentials needed for `pytest`.

### `validate_credentials` semantics (`--validate`)

Keep the hard-fail vs warning split **consistent across providers** — an inconsistency here is what let a bad Azure key report success once. **Hard-fail** (`valid = False`) only for what *will* break the run: missing/placeholder key, non-HTTPS `base_url`, unparseable endpoint, a connection-test **401/403**, and for Bedrock an AWS identity/credential-chain error. **Warn** for everything non-fatal or inconclusive: short key, missing API version, timeout/DNS/TLS/refused, any other status. The connection test is best-effort and skippable (`CODEREVIEW_SKIP_CONNECTION_TEST=1`); DeepSeek/Moonshot/Z.AI/OpenAI-on-Bedrock run none and check key presence + HTTPS only. → `docs/validation-contract.md`

- **Placeholder keys** must include the README's export strings **verbatim, punctuation included** — `<your-amazon-bedrock-api-key>` *with* the angle brackets, or the documented placeholder passes `--validate` and 401s later. Use `is_placeholder_api_key(key, extra)` (`mixins.py`); a new provider adds its env var to `_README_KEY_EXTRAS`, since `tests/test_placeholder_keys.py` scrapes `README.md`.
- **Model-access validation checks *catalog visibility only*** — never invocation permission or inference-profile routing (Bedrock is the only provider with one), so it's the only branch reporting success and a miss is a **warning, never `valid = False`**. Exact-match the prefix-stripped base id: substring matching let `minimax.minimax-m2.5` report green against a catalog holding only `m2`/`m2.1`.
- **The four `mixins.py` predicates — never re-spell one inline** (each was a per-provider one-liner first and each drifted): `is_blank`, `is_https_url` (schemes are case-insensitive *and* a hostname is required), `require_https` (fail-closed, defined *in terms of* `is_https_url` so constructor and pre-flight can't disagree), `is_short_api_key` (warning only). Locked by `test_every_url_checking_provider_uses_the_shared_https_predicate` and `test_validate_accepts_any_url_the_constructor_accepted`.

## Adding things

**New model:** add an entry under the matching provider in `codereview/config/models.yaml`. Fields: `id`, `full_id` (provider's identifier), `name`, `aliases` (never repeat the `id` — `_check_alias_hygiene` rejects it), `deprecated_aliases` (only for names inherited from a removed entry), `pricing.input_per_million`/`output_per_million`, `inference_params`, `context_window`. For a reasoning/thinking model also set `supports_tool_use: false` (→ `docs/structured-output.md`), omit `default_temperature`, and on Bedrock add `read_timeout: 1800`.

**New provider:** subclass `ModelProvider` in `codereview/providers/`, implementing `analyze_batch`, `_create_model`, `_create_chain`, `_extract_token_usage`, `_is_retryable_error`, `_calculate_backoff`, `validate_credentials`. Add one row to `_PROVIDER_REGISTRY` (`providers/factory.py`), the `<Name>Config` class to `config/models.py`, a parsing branch to `config/loader.py`, and the env-var to `cli.py`'s Provider Setup table — that table is hand-written prose where `models.yaml`'s `${VAR}` references are authoritative, so `test_provider_setup_table_covers_every_configured_provider` and `test_provider_setup_table_names_the_env_vars_models_yaml_actually_reads` tie them together (rows match by substring; `bedrock` is exempt from the second direction, documenting boto3's variables). A URL-taking provider also needs a `_cleartext_<name>` builder in `_CLEARTEXT_BUILDERS` (`tests/test_provider_result_shape_contract.py`).

Three rules that are easy to get wrong; the full method table is in `docs/providers.md`:

- `analyze_batch` keeps `max_retries: int | None = None` and resolves it with `_resolve_max_retries` — a concrete default here overrides the provider layer for every caller.
- `_create_model` enforces HTTPS via `require_https` **before** constructing the client (it runs from `__init__`, so a caller that never runs `--validate` still can't ship a key over `http://`; CWE-319), and ends with `return self._apply_structured_output(base_model)`.
- `supports_token_streaming` is a **classmethod** — override it to `False`, with the reason in the docstring, if the client never delivers a token to a callback. `tests/test_streaming_contract.py::test_the_non_streaming_provider_set_is_exactly_the_documented_one` fails until the new provider is classified either way.

Don't add shared mutable state to a provider without a lock (see the concurrency gotcha). **Reference implementations:** `deepseek.py` (OpenAI-compatible vendor *with* a dedicated langchain package), `zai.py` (one without — `ChatOpenAI` + custom `base_url`), `bedrock_openai.py` (reasoning model on the Responses API, temperature/top_p opt-out), `moonshot.py` (`BaseChatOpenAI` subclass with vendor quirks). A tool-use-less endpoint needs nothing extra — `supports_tool_use: false` routes the prompt path.

**New review category:** add to `ReviewIssue.category` Literal + `VALID_CATEGORIES` + `CATEGORY_MAPPING` (**including every synonym a model might emit**) in `models.py`, mention it in `SYSTEM_PROMPT` (`config/prompts.py`) with an explicit boundary against the neighbouring category, and add it to `_generate_recommendations`'s `category_configs` in `cli.py` if it warrants a headline recommendation. Update the count in README's "Review Categories" heading.

**New language:** add extension to `FileScanner.target_extensions`, add language section to `SYSTEM_PROMPT`, add to `LANGUAGE_EXTENSIONS` in `renderer.py`, add a fixture under `tests/fixtures/`.

**New static-analysis tool:** add to `StaticAnalyzer.TOOLS` (name, description, command, language, optional `version_command`); handle tool-specific output parsing in `run_tool` if it has unusual exit codes.

## Static analysis

```bash
uv pip install -e ".[static-analysis]"           # Python tools
uv run codereview ./src --static-analysis        # Run alongside AI review
```

Tools detected at runtime; only installed ones run. Python (ruff/mypy/black/isort/vulture/bandit), Go (golangci-lint/go vet/gofmt/gosec), Shell (shellcheck/bashate), C++ (clang-tidy/cppcheck/clang-format), Java (checkstyle), JS/TS (eslint/prettier/tsc/npm audit). See `docs/static-analysis.md` for install per-language **and for the implementation notes behind the rules below**.

- **mypy, ESLint and Prettier execute code the *reviewed repository* supplies, and are gated off by default.** `run_tool` consults `_find_executable_config` (against `_CONFIG_EXECUTION_RISK`) **before the command is built** and returns a failed result naming the config rather than running; `--trust-repo-config` opts back in. Two design rules: **detect on *content*, not presence**, and **fail closed** (unreadable, or over 512 KB, counts as risky). The skip message must say the tool's **findings are missing from this review** — a silently-absent tool reads as "clean". The `_CONFIG_EXECUTION_RISK` block in `tests/test_static_analysis.py` asserts `subprocess.run` was *not called*; asserting on the result would pass even if the tool ran first.
- **"I ran and found problems" ≠ "I never started"** — `_OPERATIONAL_FAILURE_EXIT_CODES` maps the exit codes meaning a tool *couldn't run* (ruff/black/mypy: 2) so the result carries `issues_count=0` plus an explicit error, not a fabricated count. The map is deliberately short and **verified against the installed binaries, not the docs**; where a tool's two meanings are ambiguous, keep treating the exit as a finding.
- **Parse structured output from `stdout` only** — `_count_npm_audit_issues` was fed `stdout + stderr`, and one routine `npm warn` made valid `--json` unparseable, silently reporting 0 advisories. Keep `result.stderr` for the human-readable fields.
- **Determinism:** when `MAX_FILES_PER_TOOL=500` truncation triggers, file lists must be `sorted(...)[:N]` (`test_truncation_is_deterministic`).
- **Path filtering for prompt condensation matches whole components, from the right.** `_line_mentions_any_path` compares component tuples and `_path_match_token` keeps the **whole** normalized path — a two-component `parent/basename` token can't disambiguate same-named files under same-named parents, and `api/`, `utils/`, `models/`, `tests/` are exactly the names that repeat. Don't reintroduce a fixed-width token, and don't `lstrip("./")` — it eats a leading `.` from a real name.

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

**Two test rules this codebase learned the hard way**, both because a hand-built fixture invented the exact field the broken code read and so passed for as long as the code was wrong:

- **Build the error/response the way the real client builds it** — route a real `requests.Response` through the vendor's own error path; drive extractors with an `AIMessage` the vendor's client produced from a recorded wire payload (`tests/test_retry_contract.py`, `tests/test_token_usage_contract.py`).
- **Add a reflective guard so coverage can't lapse** — enumerate every `ModelProvider` subclass (or every provider module matching a pattern) and fail when a new one isn't classified: `_USAGE_MATRIX`, `_RETRY_MATRIX`, `_CLEARTEXT_BUILDERS`, the streaming provider set, the factory registry.

## Gotchas

- **Pydantic V1 compat warning** under Python 3.14 is upstream from LangChain — harmless.
- **Reasoning models don't accept `temperature`/`top_p`** (Claude Opus 5 / 4.8 / Sonnet 5 / Fable 5, GPT-5.4 / 5.4 Pro on Azure, GPT-5.5 / 5.6 Sol on Bedrock, DeepSeek-V4-Pro) — omit `default_temperature`; Bedrock and Azure pass `allow_none=True` to `_resolve_temperature`. **Gemini sampling params are deprecated from 3.6 Flash onward** — omit all three for every new Gemini entry (3.1 Pro keeps theirs); `test_gemini36_flash_omits_sampling_params` pins 3.6 and `test_every_modern_gemini_entry_omits_sampling_params` catches the next entry.
- **18 of 31 entries ship `supports_tool_use: false`** — the Claude-on-Bedrock family, everything on `bedrock-mantle` (GPT-5.5, GPT-5.6 Sol, Grok 4.3), and the MiniMax / Kimi / GLM / Qwen3.5 / Step / Mistral re-hosts. Only **Opus 5** has vendor confirmation (its model card lists *Structured outputs: Not Supported*); most of the rest are the assume-prompt-parsing rule, not a live failure. The two **Gemini Flash** entries are the only thinking models that won `true` back with a live run — that's the bar for flipping one. An **Azure Foundry deployment of an open-weight model** needs the same flag — SGLang/vLLM reject a forced `tool_choice` unless started with `--enable-auto-tool-choice` — and `tests/test_azure_provider.py::test_supports_tool_use_false_uses_prompt_parsing` keeps that shape as the reference case. Per-entry matrix → `docs/structured-output.md`
- **`use_responses_api: true`** for GPT-5.x in `models.yaml` — the ChatCompletion API does not support reasoning summaries for these.
- **Concurrent batches:** `TokenTrackingMixin._track_tokens` and `CodeAnalyzer.skipped_files` are lock-guarded. Don't add other shared mutable state to providers without a lock, and don't attach a `StreamingCallbackHandler` under `max_workers > 1` (that's the concurrent-`Live` overlap above).
- **OpenAI-on-Bedrock is NOT the `bedrock` provider.** GPT-5.5 / GPT-5.6 Sol / **Grok 4.3** use Bedrock's *OpenAI-compatible* `bedrock-mantle` endpoint — `ChatOpenAI` + `base_url` with a Bedrock **API key (bearer token)**, not SigV4 `ChatBedrockConverse` — in the separate `bedrock_openai` provider. Their `full_id` must be a **literal**: an unset `${BEDROCK_OPENAI_MODEL_ID}` expands to `""`, fails `min_length=1` and breaks `--list-models`. Grok 4.3 and GPT-5.6 Sol are **In-Region only** (Grok: us-west-2/us-east-1/us-east-2; Sol: us-east-1/us-east-2). GPT-5.4 on *Azure* is a separate entry keeping `supports_tool_use: true`. → `docs/providers.md`
- **Two endpoint traps:** Moonshot has two non-interchangeable platforms (`platform.moonshot.cn`, our YAML default, and `platform.moonshot.ai` — `.ai` keys need a `base_url` override); and Azure/SGLang validates `body.model` strictly while `AzureChatOpenAI` serializes `"model": null`, so the Azure provider sets `model=deployment_name` explicitly.
- **Bedrock throttling arrives under several spellings** (`ThrottlingException`, `TooManyRequestsException`, `ServiceUnavailable`, `ModelTimeout`, plus `ClientError` codes) and `_is_retryable_error` classifies on rendered text, so **test retryability from the *outside***: a narrow substring list silently turned a retryable throttle into a lost batch. Add the spelling *and* a case for it.
- **`--list-models`** shows everything regardless of credentials and advertises `aliases` only (`deprecated_aliases` collapse to `+N deprecated`, expanded by `--verbose`). Rich truncates the Aliases column unless it sets both `overflow="fold"` *and* a computed `min_width`, printing alias spellings that are invalid as typed (`test_list_models_never_truncates_an_alias`).
- **Line counts need the last line even without a trailing newline** — count `"\n"` occurrences *plus one* for a non-empty final line, or `total_lines` drifts low across a repo.
- **CommonMark fence widths, not fence parity** (`tests/test_markdown_export.py::_inside_code_block`) — a block opened with N backticks closes only on a fence line of **at least N** and no info string, so a numerically *balanced* document can still trap every following section. A parity assertion passed under mutation for exactly this reason; assert with a real state machine.
