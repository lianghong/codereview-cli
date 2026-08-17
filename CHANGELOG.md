# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### New Models
- **Gemini 3.7 Flash (Google GenAI)** — latest and most capable Flash model,
  built for complex coding, agentic workflows and reliable multi-step execution
  (released August 2026)
  - Model ID: `gemini-3.7-flash` (`full_id: gemini-3.7-flash`)
  - Aliases: `gemini37-flash`, `gemini3.7-flash`, plus the generation-neutral
    `gemini-flash` (moved off 3.6 Flash — see Changed)
  - 1M-token context (card: 1,048,576; registered as the conservative
    1,000,000, matching the 3.6 entry — under-stating only shrinks batches,
    over-stating overflows), up to 64K output
  - Pricing $1.50/$7.50 per M, the same standard rate as 3.6 Flash. Google is
    running an introductory discount ($0.75/$3.75 through 2026-12-31,
    reverting 2027-01-01); the **standard** rate is recorded, so estimates
    over-state cost during the promo rather than under-stating it afterwards
  - Thinking supported at low/medium/high; `minimal` is rejected with an error
  - Sampling params omitted — `temperature`/`top_p`/`top_k` are deprecated for
    every Gemini entry from 3.6 Flash onward
  - **Keeps the tool-use path** (no `supports_tool_use` override), making it
    the second exception to the assume-prompt-parsing rule. Earned the same
    way 3.6 Flash did: the model card lists Structured outputs *and* Function
    calling as Supported, and three live runs on 2026-08-17 each returned a
    valid `CodeReviewReport` with `parsing_error` None — each reporting
    non-zero `output_token_details.reasoning`, so tool-use held *while the
    model was thinking*, which is the exact condition the rule guards against
- **Claude Opus 5 (AWS Bedrock)** — Anthropic's most capable Opus model and
  first of the Claude 5 generation's Opus tier; **new CLI default model**
  (released 2026-07-24)
  - Model ID: `opus5` (`full_id: us.anthropic.claude-opus-5`, geo-US inference
    profile; In-Region us-east-1, Geo/Global route more broadly)
  - Aliases: `claude-opus-5`, `opus-5`, `claude-opus5`, plus the
    generation-neutral `opus` and `claude-opus` (moved off Opus 4.6 — see
    Changed)
  - 1M-token context (both default and maximum), up to 128K output; pricing
    $5/$25 per M, unchanged from Opus 4.8. Cache read $0.50/M, 5-minute cache
    write $6.25/M; minimum cacheable prompt drops to 512 tokens (from 1,024)
  - Step-change gains over Opus 4.8 in deep reasoning, agentic/long-horizon
    coding, and — most relevant here — code review and bug-finding
  - `supports_tool_use: false` (prompt-based JSON parsing). Unlike the other
    adaptive-thinking Claude entries this is not an assume-prompt-parsing
    guess: the Bedrock model card lists *Structured outputs: Not Supported*
    for both the `bedrock-runtime` and `bedrock-mantle` endpoints. Thinking is
    also **on by default** (a breaking change from Opus 4.8, where it was off
    unless requested), reproducing the Opus 4.7/4.8
    forced-`tool_choice`-while-thinking conflict
  - `read_timeout: 1800` — thinking-on-by-default at default effort `high`
    over the non-streaming Converse path emits no bytes until the full
    response is generated, so think-heavy batches would outlast the 300s
    provider default (same condition that forced Fable 5's override)
  - Reasoning model: `temperature`/`top_p`/`top_k` omitted. No
    `provider_data_share` opt-in needed (unlike Fable 5) — zero data retention
    is on by default on Bedrock
  - Registered from the published model card; not yet exercised against the
    live endpoint from this repo
- **Claude Sonnet 5 (AWS Bedrock)** — first Sonnet-tier model of the Claude 5
  generation, near-Opus-4.8 intelligence at Sonnet pricing (announced
  2026-06-30)
  - Model ID: `sonnet5` (`full_id: us.anthropic.claude-sonnet-5`, geo-US
    inference profile; routes us-east-1/us-east-2/us-west-2)
  - Aliases: `claude-sonnet-5`, `sonnet-5`, `claude-sonnet5` (the bare
    `sonnet` alias intentionally stays on Sonnet 4.6)
  - 1M-token context, up to 128K output; pricing $3/$15 per M (standard;
    a launch promo of $2/$10 runs through 2026-08-31 — we register the
    durable standard rate)
  - `supports_tool_use: false` (prompt-based JSON parsing) — first Sonnet
    tier with adaptive thinking on by default, so it hits the same
    forced-`tool_choice`-while-thinking conflict as Opus 4.7/4.8. Also
    rejects `temperature`/`top_p`/`top_k`. Unverified live; flip to `true`
    only if a live run proves tool-use works
- **Gemini 3.6 Flash (Google GenAI)** — Google's new GA workhorse model
  (released 2026-07-21), successor to Gemini 3.5 Flash: better coding and
  multimodal work while using ~17% fewer output tokens, at a lower output price
  - Model ID: `gemini-3.6-flash` (`full_id: gemini-3.6-flash`)
  - Aliases: `gemini36-flash`, `gemini3.6-flash`, plus the
    generation-neutral `gemini-flash` (moved off Gemini 3 Flash — see Changed)
  - 1M-token context, up to 64K output; pricing $1.50/$7.50 per M (flat — no
    >200K-prompt tier), cheaper output than the $9.00/M of the 3.5 Flash it
    replaces
  - Thinking on by default at level `medium`; computer-use and agentic-task
    capable
  - `temperature`/`top_p`/`top_k` omitted deliberately — **all three are
    deprecated from this model onward**: Google's API ignores them today and
    documents an HTTP 400 for future model generations. The Google provider
    already passes `allow_none=True` to `_resolve_temperature` and drops
    `top_p`/`top_k` when unset, so no code change was needed
  - Keeps the tool-use path (`supports_tool_use` defaults to `true`) — Google
    documents both structured outputs and function calling, and a live review
    run against the endpoint confirmed it works
  - Note there is no Gemini 3.6 **Pro**; the Pro tier remains at 3.1
- **MiniMax M3 (NVIDIA NIM)** — multimodal MoE vision-language model
  (428B total / ~22B active, A22B), 1M-token context, text-only output,
  long-form video understanding (to 30 min) and long-horizon coding (8+ hrs)
  - Model ID: `minimax-m3-nvidia` (`full_id: minimaxai/minimax-m3`)
  - Aliases: `minimax-m3`, `mm3-nvidia`, `mm3`
  - Free/non-commercial NVIDIA NIM trial endpoint → cost renders `TBD`
  - Interleaved thinking enabled (temp 1.0 / top_p 0.95 / top_k 40,
    128K max output), mirroring the MiniMax M2.7 house recommendation
  - `supports_tool_use: false` (prompt-based JSON parsing) — per the
    "assume prompt-parsing until a live run proves tool-use works" rule
    for new reasoning models; live-verified working on the prompt path
- **GLM-5.2 (NVIDIA NIM)** — Zhipu flagship, successor to GLM-5.1: 753B-total
  MoE with IndexShare sparse attention, "solid" 1M-token context, multiple
  thinking effort levels (High/Max)
  - Model ID: `glm52` (`full_id: z-ai/glm-5.2`)
  - Aliases: `glm52-nvidia`, `glm5.2-nvidia`, `glm-5.2-nvidia`; also absorbs
    the version-neutral GLM-5 aliases (`glm5`, `glm-5`, `glm5-nvidia`) as
    `deprecated_aliases`. The version-explicit GLM-5.1 names (`glm51`,
    `glm51-nvidia`, `glm-5.1`, `glm5.1`) were deleted, not migrated
  - Free NVIDIA NIM trial endpoint → cost renders `TBD`; thinking enabled
    (temp 0.5 / top_p 0.95 for deterministic review)
  - `supports_tool_use: false` (prompt-based JSON parsing) — NIM re-host emits
    malformed tool-call JSON and it's a thinking model; assume-prompt-parsing
    rule (flip to `true` only if a live run proves tool-use)
- **GPT-5.6 Sol (OpenAI-on-Bedrock)** — flagship of OpenAI's GPT-5.6 family
  (Sol/Terra/Luna), GA on Amazon Bedrock (launched 2026-07-13) via the same
  OpenAI-compatible `bedrock-mantle` endpoint. Sol is OpenAI's best coding
  model to date (SOTA on the Artificial Analysis Coding Agent Index,
  Terminal-Bench 2.1, DeepSWE) — the code-review pick of the family.
  - Model ID: `gpt5.6-sol-bedrock` (`full_id: openai.gpt-5.6-sol`)
  - Aliases: `gpt5.6`, `gpt-5.6`, `gpt5.6-bedrock`
  - 272K context; pricing $5/$30 per M (OpenAI list; cache read $0.50)
  - Reasoning model: **Responses API only** (Chat Completions not supported →
    `use_responses_api: true` required), rejects `temperature`/`top_p`
  - `supports_tool_use: false` (prompt-based JSON parsing) — same adaptive
    server-side reasoning / reasoning-only failure mode as GPT-5.5/5.4 on
    Bedrock
  - In-Region only: us-east-1 / us-east-2 (no Geo/Global; narrower than Grok)
  - Terra (`openai.gpt-5.6-terra`, $2.50/$15) and Luna (`openai.gpt-5.6-luna`,
    $1/$6) are the cheaper balanced / high-volume tiers — not registered, add
    the same way if wanted
- **Grok 4.3 (OpenAI-on-Bedrock)** — xAI reasoning-first frontier model on
  Amazon Bedrock's new OpenAI-compatible `bedrock-mantle` endpoint (NOT the
  SigV4 Converse path — the model card lists Converse/`bedrock-runtime` as
  unsupported). Rides the existing `bedrock_openai` provider (`ChatOpenAI` +
  `base_url` + a Bedrock API-key bearer token).
  - Model ID: `grok-4.3-bedrock` (`full_id: xai.grok-4.3`)
  - Aliases: `grok`, `grok-4.3`, `grok43`, `grok-bedrock`
  - 1M context; pricing $1.25/$2.50 per M (third-party aggregator, pending AWS)
  - Unlike the GPT-5.x entries on this endpoint it **accepts**
    `temperature`/`top_p` (card defaults 0.7/0.95), so it uses Chat Completions
    (no `use_responses_api`) with temp 0.3 for deterministic review
  - `supports_tool_use: false` (prompt-based JSON parsing) — always-on
    reasoning is the highest-risk forced-`tool_choice`-while-thinking profile
  - In-Region only: us-west-2 / us-east-1 / us-east-2 (no Geo/Global)

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
  - Model: `zhipuai/glm-5.2` (1M context, $1.40/$4.40 per M) — shipped as
    `zhipuai/glm-5.1` (203K context) and moved to 5.2 within this cycle;
    the version-neutral 5.1 aliases (`zai-glm`, `glm-zai`) now resolve here
  - Aliases: `glm`, `glm-5.2`, `glm5.2`, `glm5.2-zai`, `zai-glm`, `glm-zai`
  - Reads `${ZAI_API_KEY}`; base URL `https://api.z.ai/api/paas/v4/`
- **Moonshot AI (Kimi)** — 7th provider, via dedicated `langchain-moonshot`
  package (extends `BaseChatOpenAI`)
  - Model: `kimi-k2.6` (1T MoE, 32B active, 256K context, $0.60/$2.50 per M)
  - Aliases: `kimi`, `kimi26`
  - Reads `${KIMI_API_KEY}`; base URL `https://api.moonshot.cn/v1` (Chinese
    platform; override to `https://api.moonshot.ai/v1` for international keys)
  - **Naming cleanup**: NVIDIA's `kimi-k2.6-nvidia` lost aliases `kimi-k2.6`
    and `kimi26` (claimed by direct provider); kept as `kimi-nvidia-26`,
    `kimi26-nvidia`. Bedrock's `kimi-k2.5-bedrock` lost the bare `kimi`
    alias (now routes to canonical Moonshot K2.6); kept as `kimi-bedrock`,
    `kimi25-bedrock`.

#### Review quality
- **New `Correctness` review category** — the taxonomy had no home for "the
  code returns a wrong result", despite the system prompt stating the review
  priority as *security > correctness > maintainability > performance*. Two
  concrete defects this fixes:
  - Real bugs were filed as `Code Quality`, alongside naming and readability
    nits, so severity was the only thing separating a race condition from a
    typo suggestion.
  - Every word a model naturally reaches for when it finds a bug
    (`correctness`, `bug`, `logic`, `logic error`, `reliability`,
    `concurrency`, `race condition`, `thread safety`, `data loss`, `crash`,
    `edge case`) was an unmapped category → coerced to `Code Quality` **and**
    counted in the `category_coerced` drift counter. That counter exists to
    detect prompt/schema drift, so correct model behavior was polluting the
    signal the CLI surfaces to users at ≥5 coercions.
  All of those spellings now map to `Correctness` silently. `"error handling"`
  deliberately still maps to `Code Quality` — as a bare category name it's
  ambiguous between "this path crashes" (correctness) and "use a narrower
  exception type" (quality).
- **`CORRECTNESS ANALYSIS` section added to `SYSTEM_PROMPT`** — placed above
  the architecture section to match the stated priority order. Covers logic
  and control flow (off-by-one, inverted comparison, unreachable branch,
  precedence), edge cases (empty/single/boundary, null, zero and negative,
  unicode, first/last iteration), error paths (half-updated state, resource
  not released on the error path, swallowed error, non-idempotent retry),
  concurrency (unlocked shared mutable state, TOCTOU, deadlock, async result
  read before completion), and resource lifecycle. Includes an explicit
  `Correctness` vs `Code Quality` boundary rule: if the code produces a wrong
  result or crashes for a nameable input it is Correctness; if it works but is
  hard to maintain it is Code Quality. Constraint 10 (name the triggering
  input or don't report) still governs.
- `_generate_recommendations` ranks `Correctness` directly below `Security`.

#### CLI Features
- **`--fail-on <severity>`** — turn a review into a CI merge gate. Exits
  **2** when any issue at the given severity *or above* was found
  (`--fail-on high` trips on High **and** Critical). Opt-in: without the flag,
  findings never affect the exit code, so existing invocations are unchanged.
  - **Exit `2` is deliberately distinct from `1`.** `1` means the *run* failed
    (no results, bad credentials, API error, unwritable output); `2` means the
    review *succeeded* and the code has blocking issues. A pipeline needs to
    respond to those differently, and previously it couldn't tell them apart —
    a run that found 4 Critical security issues exited `0`.
  - **Independent of `--severity`.** `--severity` filters only what is
    displayed; the gate counts `report.issues`, so
    `--severity critical --fail-on high` still fails on a High finding the
    terminal never printed. Gating on the rendered subset would let a display
    preference silently punch a hole in the gate — locked by
    `test_severity_filter_does_not_affect_fail_on`.
  - **Applied after export**, as the last statement of the run, so a failing
    build still leaves its `--output` artifact for upload.
- **`--list-models --verbose`** — expands the deprecated aliases that plain
  `--list-models` collapses to `+N deprecated`, with a footer explaining why
  they're hidden. Advertising a back-compat name is actively misleading:
  `--model gemini-3-flash` resolves, but to Gemini **3.6** Flash.
- **`deprecated_aliases` YAML key on model entries** — back-compat-only names
  (usually inherited from a removed entry) that resolve exactly like `aliases`
  but stay out of `--list-models`. The split is purely a display concern;
  `ConfigLoader._register_all_names` registers both lists identically, so no
  resolution behavior depends on which list a name is in.
- **`--tool-timeout`** — override the static-analysis subprocess timeout
  (default 120s, range 1–3600). Useful for `cppcheck --enable=all` on
  large C++ repos and `mypy` strict on big Python codebases.
- **`--include-hidden`** — opt-in scanning of `.github/scripts`, `.config/`,
  and other dotfile directories. Default behavior (skip hidden) is unchanged.
- **`--trust-repo-config`** — opt back into running mypy / ESLint / Prettier
  against a reviewed repository that ships a config making them execute code
  from the tree (a mypy `plugins =` entry, a JavaScript `eslint.config.*`, a
  `plugins` key in `.prettierrc` / `package.json`). Off by default: that code
  runs with your privileges. Skipped tools say so explicitly, including that
  their findings are absent from the review. See Fixed for the reproduction.
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
- **Model-profile drift detection** — new `tests/test_model_profile_drift.py`
  (9 tests) cross-checks `models.yaml` against the `_MODEL_PROFILES` tables the
  LangChain partner packages now ship, catching a `context_window` or
  `max_output_tokens` above the model's real cap and any `supports_tool_use`
  that disagrees with the profile's `structured_output`. 15 of 30 registry
  entries resolve a profile today (re-hosts — all of NVIDIA, Z.AI, Bedrock's
  OpenAI-compatible endpoint — carry ids the tables don't know).

  It is a **warn-with-allowlist** check, never a source to overwrite the YAML
  from, for two reasons: the tables are generated from the community-curated
  models.dev, and our `supports_tool_use` is *empirical* — the whole
  structured-output matrix in CLAUDE.md exists because models advertising
  `structured_output: true` fail on the forced `tool_choice` anyway. Eight
  deliberate divergences are allowlisted with a one-line reason each; the
  assertions are one-directional where a direction exists (a *conservative*
  limit is a valid cost choice, only exceeding the cap is a bug), and a separate
  test fails when an allowlist entry stops diverging so it can't accumulate
  permission for problems already fixed. The one behavioural fact that makes the
  profiles worth checking against: langchain-aws *acts* on its own table at
  runtime, dropping `temperature`/`top_p` whenever `profile["temperature"] is
  False`.

  Two meta-guards keep it from passing vacuously — the same failure the retry and
  token-usage contracts were written against. `_get_default_model_profile` is
  private API, so one test asserts it still resolves something rather than
  letting every check degrade to "no profile found", and a per-provider coverage
  pin fails when a provider drops from "some coverage" to "none".

### Changed
- **`gemini-flash` now resolves to Gemini 3.7 Flash** (was 3.6 Flash). The
  generation-neutral alias tracks the current Flash generation, per the
  registry convention. Gemini 3.6 Flash stays live and **keeps** its
  version-explicit back-compat names — `gemini-3-flash`, `gemini3-flash` and
  `g3flash` still resolve to 3.6, deliberately: a name that says "3" must not
  jump two generations to different capabilities. Only the generation-neutral
  name travels.
- **The ruff rule set is now pinned in `pyproject.toml`** (`[tool.ruff.lint]
  select = ["E4", "E7", "E9", "F"]`) instead of inheriting ruff's defaults.
  Those defaults are not stable across releases — ruff 0.16.0 enables ~400
  rules where 0.15.22 enabled `E4/E7/E9/F` — and the `ruff>=…` floor is
  resolved by `uv pip install -e .` without consulting `uv.lock`. The
  documented pre-commit gate therefore reported 135 errors on an unmodified
  checkout purely because of which ruff the venv happened to install, which
  left it unable to distinguish "this change is clean" from "the toolchain
  moved". The pinned value is ruff's own historical default, i.e. what this
  codebase was written and reviewed against, so no verdict changes; both
  0.15.22 and 0.16.0 now agree. Widening the set remains a fine idea to do
  deliberately, in a commit that also lands the resulting fixes.
- **`ProviderFactory` dispatch is now a registry table** — `_PROVIDER_REGISTRY`
  maps each provider name to its config type, module and class name, replacing
  an eight-branch if/elif chain (~130 lines) plus a hand-written
  `Supported providers: …` list in the error message. That list is now derived
  from the table, so it can't go stale when a ninth provider lands; adding a
  provider is one row. Module/class stay **strings** to keep the import lazy —
  each provider module imports its vendor's LangChain client at module scope,
  so an eager table would pull all eight client packages into every run,
  including `--list-models`. Two new tests in `tests/test_factory_smoke.py`
  pin the table to the loader's provider set and resolve every row's
  module/class, since a typo in a lazily-imported name would otherwise survive
  until a user selected that provider's model. No behavior change beyond
  ordering: an unknown provider now reports the factory's message rather than
  the loader's, which names the supported set.
- **Default model is now `opus5`** (was `opus4.8`) — Claude Opus 5 supersedes
  Opus 4.8 at identical $5/$25 pricing, with Anthropic specifically calling out
  code review and bug-finding among its largest gains. Runs that relied on the
  implicit default now hit Opus 5; pass `--model opus4.8` to pin the old one.
- **The generation-neutral `opus` and `claude-opus` aliases now resolve to
  Opus 5** (were Claude Opus 4.6). The Opus 4.6 and 4.7 entries were
  subsequently removed (see Removed) and their version-explicit names
  (`opus4.6`, `claude-opus-4.6`, `opus4.7`, `claude-opus-4.7`, …) were
  **deleted** rather than pointed at Opus 5 — use `opus` or `opus5`. Note
  `ConfigLoader._register_model` is last-write-wins
  *within* a provider (it only warns across providers), so a superseded entry
  must not keep a generation-neutral name as its `id` or it silently shadows
  the newer entry's alias depending on YAML ordering.
- **The generation-neutral `gemini-flash` alias now resolves to Gemini 3.6
  Flash** (was Gemini 3 Flash Preview). `gemini-3-flash-preview` is deprecated
  upstream with `gemini-3.6-flash` named as its replacement; the older entry
  was then removed (see Removed) and its names (`gemini-3-flash`,
  `gemini3-flash`, `g3flash`) are now `deprecated_aliases` on the 3.6 Flash
  entry — they don't state a minor version, so they were migrated rather than
  deleted. Same generation-neutral-alias convention as the `opus` move
  above.
- **`supports_tool_use` audit — 6 models moved to prompt-based JSON parsing**
  (`true` → `false`). A per-model research pass found these were on the
  forced-`tool_choice` (tool-use) path despite being thinking models and/or
  having documented tool-call breakage on their endpoints. Moving them to the
  prompt-parsing path is the safe direction (it works even where tool-use
  also would); flip any back to `true` only if a live run proves tool-use.
  - Kimi K2.5 (Bedrock) `kimi-k2.5-bedrock` — Converse leaks Moonshot
    tool-call markers (`<|tool_call_begin|>…`) into text
  - Kimi K2.6 (NVIDIA) `kimi-k2.6-nvidia` — same model/behavior as Kimi K2.6
    on Moonshot (already `false`); thinking on by default
  - Qwen3.5 397B (NVIDIA) `qwen3.5-nvidia` — tool calls emitted as XML inside
    the `<think>` block when thinking is on
  - GLM 5 (Bedrock) `glm5-bedrock` — thinking on by default (reasoning_effort
    defaults to max); forced `tool_choice` conflict
  - GLM-5.2 (NVIDIA) `glm52` — NIM re-host emits malformed/truncated
    tool-call JSON; kept consistent with GLM-5.1/5.2 on Z.AI (replaces the
    now-removed GLM-5.1 NVIDIA entry, which carried the same rationale)
  - Step 3.7 Flash (NVIDIA) `step-3.7-flash` — always-thinking backbone;
    forced `tool_choice` while thinking unproven
- **DeepSeek V4 thinking-default rationale corrected** (docs/comments only,
  no behavior change) — both V4-Pro and V4-Flash default to thinking **on**
  (not "V4-Flash non-thinking by default" as previously documented). Tool-use
  works because the `deepseek` provider explicitly sends `thinking: disabled`,
  not because the model is non-thinking. Updated `deepseek.py` and CLAUDE.md.
- **Retired model aliases: version-neutral names redirected, version-explicit
  names deleted (2026-07-25 registry cleanup)** — the rule for what happens to
  a removed entry's identifiers now depends on whether the name states a
  version. A name that says "4.6" resolving to Opus **5** — different pricing,
  different sampling-param support, a different structured-output path — is
  worse than an error a human reads and fixes, so those are gone; a name that
  says only "the GLM one" still gets the current GLM one. A removed entry's
  **`id`** counts as a `--model` spelling here too, and ids are what get
  forgotten (`glm51`, `kimi-k2.5-azure`, `deepseek-v4-pro-azure`,
  `zhipuai/glm-5.1` had all shipped orphaned).
  - **Redirected** (kept as `deprecated_aliases`, resolve to a live successor,
    locked by `test_retired_model_aliases_redirect_to_live_successors`):
    - `qwen-bedrock` → **Qwen3 Coder Next (Bedrock)** (`qwen-next-bedrock`),
      which already owned `qwen`/`qwen-coder`
    - `kimi-k2.5-azure`, `kimi-azure`, `kimi25-azure` → **Kimi K2.6 (Moonshot
      direct)** (`kimi-k2.6`) — the canonical owner per the direct-API
      convention
    - `deepseek-v4-pro-azure`, `deepseek-v4-azure`, `ds-v4-azure` →
      **DeepSeek-V4-Pro (DeepSeek direct)** (`deepseek-v4-pro`), likewise
      canonical
    - `qwen-nvidia`, `qwen3-nvidia`, `qwen-coder-nvidia` → **Qwen3.5 397B
      (NVIDIA)** (`qwen3.5-nvidia`)
    - `glm5`, `glm-5`, `glm5-nvidia` → **GLM-5.2 (NVIDIA)** (`glm52`)
    - `gemini-3-pro`, `gemini3-pro` → **Gemini 3.1 Pro** (`gemini-3.1-pro`)
    - `gemini-3-flash`, `gemini3-flash`, `g3flash` → **Gemini 3.6 Flash**
      (`gemini-3.6-flash`)
    - `step-flash` stays on **Step 3.7 Flash (NVIDIA)**, `zai-glm`/`glm-zai`
      stay on **GLM-5.2 (Z.AI)** — version-neutral names that were already
      current aliases
  - **Deleted** — see Removed below for the full list and the README's
    *Migrating Deleted Aliases* table for replacements.
  - `tests/test_config.py::test_no_historical_model_id_is_orphaned` reads the
    last 8 revisions of `models.yaml` straight from git and asserts every
    id/alias that ever shipped either still resolves **or** appears in the
    `RETIRED_ALIASES_DELETED_NOT_REDIRECTED` allowlist with a stated reason —
    so a name can never be dropped by accident, only on purpose.

### Removed
- **Sixteen inert prompt-caching pricing keys deleted from `models.yaml`** — six
  `cache_write_per_million` / `cache_read_per_million` pairs (Claude Fable 5,
  Opus 5, Opus 4.8, Sonnet 5, Sonnet 4.6, Haiku 4.5) and four
  `cached_input_per_million` keys (Azure GPT-5.4, GPT-5.4 Pro, DeepSeek-V4-Pro,
  DeepSeek-V4-Flash). `PricingConfig` declares only `input_per_million` and
  `output_per_million`; `ConfigLoader._parse_model_config` copies pricing across
  **field by field, by name**; and neither model is `extra="forbid"`. So each of
  these loaded without error, was dropped on the floor, and could never reach a
  cost figure — while still reading to a human as configuration. Nothing
  referenced them: no `.py`, no doc, no test. This is the
  `NVIDIAConfig.max_retries` bug one level down (present in the YAML, absent
  from the constructor), with a worse failure mode: an unread *pricing* number
  is exactly the kind of thing a future reader trusts, and none of these had
  ever been checked against a billing statement.

  Deleted rather than kept as documentation. Implementing prompt caching is a
  real optimization — a `cache_control` block on the system prompt cuts the
  per-batch input cost ~10x on repeated reviews — but it is a feature with its
  own correctness surface (TTL windows, minimum cacheable length, a second
  token counter in the report), not a YAML edit. The rates can come back
  alongside it, re-verified.

  New guard: `tests/test_config.py::test_every_pricing_and_inference_key_in_the_yaml_is_actually_read`
  scrapes the keys `_parse_model_config` actually reads out of `loader.py` and
  fails on any `pricing`/`inference_params` key in the YAML that isn't among
  them. Scraped rather than hand-listed because the YAML spelling differs from
  the field name (`default_temperature` → `temperature`), so the loader is the
  only place that mapping exists — and the test asserts the scrape found
  something first, so a restructure of `_parse_model_config` fails loudly
  instead of making the check vacuous. It catches both directions: a new dead
  key added to the YAML, *and* a key the loader stops forwarding.

  Also dropped the two now-dangling pricing comments: Opus 5's explained a
  5-minute-TTL cache-write rate the file no longer states, and Azure GPT-5.4's
  long-context tier note quoted a cached rate alongside the input/output pair.

### Fixed
- **⚠️ Retried parse failures were billed by the vendor and recorded as free**
  — `_execute_with_retry` tracked token usage from the raw `AIMessage` (including
  its `parsed is None` branch), but an `OutputParserException` raises from the
  *parser*, past the message, so the prompt-parsing path recorded nothing at all
  for a rejected attempt. That is the path every reasoning model here takes
  (`supports_tool_use: false` — Opus 5, GPT-5.5/5.6 Sol on Bedrock, Grok 4.3,
  GLM-5.2, Kimi K2.6, …), and it is precisely those models that intermittently
  emit invalid JSON on think-heavy batches, so `enable_output_fixing` can burn
  several vendor-billed attempts on one batch and the cost report would show the
  successful attempt only. New `_track_usage_from_parse_failure` estimates the
  attempt from the prompt text and the rejected output the parser attaches as
  `llm_output`. Estimation is not a shortcut here — a `CodeReviewReport` carries
  no usage metadata either, so both counts on the success branch of this path are
  already estimates; the change makes the failures accounted the same way as the
  successes instead of not at all. Failures are swallowed to `logging.debug`, on
  the rule that an accounting problem must never mask the parse error it is
  reporting on.
- **A paragraph of upstream advice printed above the Rich UI on every run with
  the default model** — langchain-aws 1.6.3 added a `logger.warning` whenever it
  has to *infer* `disable_streaming`, which it does for any model absent from its
  hardcoded streaming allowlist. `claude-opus-5` is absent (the list carries
  `claude-opus-4` / `fable-5` / `sonnet-5`), so `opus5` — the CLI's own default —
  tripped it, as did `kimi-k2.5`, `minimax-m2.5` and `glm-5`. The Bedrock
  provider now passes `disable_streaming=True` explicitly. Upstream gates the
  warning on the key being *absent*, so stating any value silences it, and the
  explicit `True` is what the `read_timeout: 1800` overrides on `fable5`/`opus5`
  already assumed: non-streaming Converse emits no bytes until generation
  completes. Behavior is unchanged — this provider never passed
  `streaming=True`, so `_should_stream()` was already `False` for every entry.
  Locked by `tests/test_bedrock_provider.py::test_disable_streaming_is_passed_explicitly`,
  which asserts the key is *present* rather than asserting its value, since
  presence is what suppresses the warning.
- **⚠️ `--stream` cost a 3-5x slowdown on providers that never stream a token**
  — the flag drops the run to one worker (token-by-token output from concurrent
  batches interleaves), but Bedrock passes `disable_streaming=True`, `ChatNVIDIA`
  has no `streaming` field at all, and Google's is deliberately off, so on all
  three the serialization bought output that cannot appear — including on
  `opus5`, the default model. `run_review` now asks the new
  `ProviderFactory.supports_token_streaming(model_name)` and downgrades the flag
  with an explicit notice, keeping the parallel batches. A `classmethod` on
  `ModelProvider` answers it from the class — no credentials, no client — because
  worker count and which callback handler to attach are one decision (a
  `StreamingCallbackHandler` under `max_workers > 1` is the concurrent-`Live`
  overlap `callbacks.py` documents as corrupting terminal state) and both feed
  the provider constructor. A downgraded run still gets the concurrency-safe
  spinner handler.
- **⚠️ `--verbose` alone moved five providers onto the streaming wire path, and
  streaming lost the billed token counts** — two coupled bugs in
  `streaming=bool(self.callbacks)`. `ProgressCallbackHandler` (the `--verbose`
  handler) does not override `on_llm_new_token`, so it cannot observe a single
  streamed token: every `--verbose` run on an OpenAI-compatible provider paid for
  streaming to feed a handler that ignores it. And when streaming *is* wanted,
  `stream_options={"include_usage": True}` is only sent if `stream_usage` is set
  — langchain-openai auto-enables it only when no `base_url` is configured, and
  all five of these providers configure one. Without it a real server sends no
  usage chunk, `usage_metadata` is `None`, `extract_openai_token_usage` returns
  `(0, 0)` and `base.py` substitutes the byte-heuristic estimate, i.e. the same
  silent under-reporting fixed above, reintroduced by the flag meant to show more
  detail. New `wants_token_streaming(callbacks)` and
  `openai_stream_params(callbacks)` in `mixins.py` replace the expression in all
  five providers; detection compares `on_llm_new_token` against
  `BaseCallbackHandler`'s, not class identity, so third-party handlers work and
  `mixins.py` need not import Rich. Locked by the new
  `tests/test_streaming_contract.py` (31 tests), including one that drives a real
  `ChatOpenAI` and asserts `_should_stream()` actually flips.
- **⚠️ Every Responses-API model reported an *estimated* token count as if it
  were the vendor's** — `extract_openai_token_usage` read only
  `response_metadata["token_usage"]`, which **only** langchain-openai's Chat
  Completions converter populates. `AIMessage` carries usage in two independent
  places, and the Responses API path fills just the other one
  (`usage_metadata`), so the helper returned `(0, 0)` for every
  `use_responses_api: true` entry — Azure `gpt-5.4` / `gpt-5.4-pro`, GPT-5.5 and
  GPT-5.6 Sol on Bedrock. `.get(..., 0)` made that indistinguishable from "no
  usage reported", so `base.py` silently substituted its byte-heuristic
  estimate, which cannot see reasoning tokens at all. On a think-heavy Azure
  `gpt-5.4` batch (the tool-use path, where real vendor counts *were* sitting in
  the message) that under-reported by ~13x: 40,000 in / 9,000 out billed,
  6,211 / 145 recorded, $0.2350 of spend printed as $0.0177 — and the reasoning
  models are exactly the expensive ones. The helper now prefers the normalized
  `usage_metadata` and keeps the raw dict as a fallback, so responses carrying
  only `token_usage` still report real numbers. Affects all five OpenAI-client
  providers (Azure, DeepSeek, Moonshot, Z.AI, OpenAI-on-Bedrock); the Bedrock
  Converse, NVIDIA and Google extractors were already correct. Locked by the new
  `tests/test_token_usage_contract.py` (23 tests), which drives each provider's
  extractor with a message built by the **real vendor client** from a recorded
  wire payload — the existing hand-built-`AIMessage` tests passed for exactly as
  long as the extractor was wrong, because they invented the one field it read.
  Two reflective meta-guards keep the coverage from lapsing: every
  `ModelProvider` subclass must appear in the usage matrix, and every provider
  whose module mentions `use_responses_api` must be exercised on that path.
- **`Retry-After` was read on 429 only, so a 503 capacity window got blind
  exponential backoff** — `parse_retry_after` guarded on `RateLimitError`, but
  the header is defined for 503 (RFC 9110 §10.2.3), every OpenAI-client
  provider already *retries* 5xx via `is_openai_retryable_error`, and the openai
  SDK's own `_calculate_retry_timeout` honours the header on every retryable
  status. A server that said "come back in 30s" was ignored and retried on the
  provider's own schedule instead. The guard is now `APIStatusError`, still
  bounded by `max_wait` so a hostile or broken header can't stall a run. The
  five providers' log line reads "backoff" rather than "rate limit" now that a
  503 can reach that branch.
- **⚠️ Recursive exclude patterns silently under-excluded, leaving vendored
  trees eligible for review** — `FileScanner._is_excluded` used
  `PurePath.match` alone, which treats `**` as a *single* segment. Every entry
  in `DEFAULT_EXCLUDE_PATTERNS` has the shape `**/node_modules/**`, so `match`
  read it as literally "one segment, `node_modules`, one segment":
  `a/node_modules/x.py` matched, but `node_modules/x.py` (no leading segment)
  and `a/b/node_modules/deep/x.py` (too many) did not. The `os.walk` prune set
  masked this for an ordinary scan — the directory was never walked — but not
  when a pattern is path-qualified and so contributes no prune name, nor for
  any caller reaching `_is_excluded` directly. Now tests `match` **or**
  `full_match`: `full_match` recurses `**` correctly but requires the whole
  relative path to match, so it cannot replace `match` (it rejects `*.py`
  against `a/b/x.py`) — the union is what covers both spellings. Safe rather
  than merely convenient: with no `**` in the pattern `full_match` is
  *stricter* than `match`, so it can only add matches in the recursive case,
  and it does not widen a path-qualified pattern into another subtree
  (`docs/api/**` still doesn't match `app/api/views.py`). Side effect worth
  knowing: `docs/api/**` now excludes `docs/api/sub/x.py`, which it previously
  did not.
- **⚠️ A failed markdown export printed a bare `✗ Error:` with no diagnosis** —
  `run_review`'s export handler caught only `OSError`, but
  `MarkdownExporter.export` converts `OSError` into `RuntimeError` as its
  documented contract, so only the JSON path was covered. A markdown export to
  an unwritable path fell through to the generic `except Exception`, which lost
  the `escape()` on a repository-controlled path and printed a traceback under
  `--verbose` for a plain permissions problem. The handler now catches both
  spellings and reports `e.__cause__` when present, so the message names the
  actual failure ("Permission denied") rather than repeating the path. A
  companion `except click.Abort: raise` stops the generic handler from
  overwriting an already-printed diagnosis with an empty `✗ Error:` —
  `click.Abort` subclasses `RuntimeError` and its `str()` is empty.
- **A bad `models.yaml` entry raised a bare `KeyError`/`ValidationError` naming
  neither the file nor the entry** — `ConfigLoader._parse_model_config` let
  both propagate raw, so a typo'd key in one model surfaced as
  `KeyError: 'pricing'` with no indication of which of 30 entries was at fault.
  Both are now re-raised as `ValueError` naming the config path and the
  offending entry (by `id`, falling back to `name`/`full_id`), and the
  top-level YAML load names the file too.
- **The five legacy `codereview.config` constants ignored
  `get_config_loader.cache_clear()`** — `DEFAULT_EXCLUDE_PATTERNS`,
  `DEFAULT_EXCLUDE_EXTENSIONS`, `MAX_FILE_SIZE_KB`, `WARN_FILE_SIZE_KB` and
  `MODEL_ALIASES` were eager module-level snapshots taken at first import, so
  after the documented test-reset every accessor function returned the reloaded
  config while these five kept the original values — two spellings of the same
  setting silently disagreeing. Now resolved lazily through a module-level
  `__getattr__` (PEP 562), with `__dir__` keeping them discoverable and a
  `TYPE_CHECKING` block preserving their concrete types for mypy. This does
  **not** defer the YAML load itself: `scanner.py` and `cli.py` bind some of
  these names with a module-level `from codereview.config import …`, which
  copies the value at the importing module's import time.
- **A truncated `GOOGLE_API_KEY` or `NVIDIA_API_KEY` reported all-green from
  `--validate`, then 401'd on the first batch** — both providers omitted the
  "unusually short" warning that the other five emit. NVIDIA's `nvapi-` prefix
  check does not subsume it: a truncated key keeps its prefix. Both now call
  the shared `is_short_api_key`, and the check is enforced across every
  key-taking provider by a parametrized contract test rather than per-provider.
  Deliberately a warning, not a hard failure — no vendor documents a minimum
  length.
- **NVIDIA's `--validate` connection test probed a doubled-slash URL and
  reported the result as a passing check** — the base URL was concatenated
  without normalizing a trailing slash, producing `…/v1//models`. Because a
  non-200/401/403 status is recorded as *inconclusive but passing*, the green
  "Connection" check described a URL the run would never use. Now `rstrip("/")`
  before building the probe URL.
- **DeepSeek computed its `extra_body` twice** — once for the client payload
  and once for the `thinking: enabled` test that routes structured output to
  the prompt-parsing path. Two independent computations of the same value can
  disagree, which here would mean sending a forced `tool_choice` to a
  thinking-mode request (HTTP 400). Computed once and reused.
- **⚠️ NVIDIA NIM retried nothing: every 429/502/503/504 aborted the batch on
  attempt 1** — `_is_retryable_error` tested
  `isinstance(error, httpx.HTTPStatusError)`, but
  `langchain-nvidia-ai-endpoints` runs on `requests`, and its
  `_NVIDIASyncClient._try_raise` *discards* the typed error: it catches
  `requests.HTTPError` and re-raises a bare
  `Exception("[504] Gateway Timeout\n…")` (the client's own source carries a
  `# todo: raise as an HTTPError`). The isinstance test therefore matched
  **nothing**, so NIM's frequent gateway 504s — the exact failure the
  provider-level `max_retries` exists for — were classified as fatal and lost
  the batch, and `NVIDIAConfig.max_retries` had no observable effect. Now
  classified on the status code, read from `.response.status_code` when present
  and otherwise parsed from the `[504] …` message prefix the client actually
  produces. Retryable set is unchanged in intent — exactly the `{429, 502, 503,
  504}` the dead check named — so this makes the existing policy execute rather
  than widening it; a bare NIM 500 stays non-retryable. 504 keeps its 4s backoff
  base, which also only ever applied when the status was readable.
- **⚠️ Google GenAI retried nothing: every 429 and 503 aborted the batch on
  attempt 1** — `_is_retryable_error` tested
  `google.api_core.exceptions.ResourceExhausted` / `ServiceUnavailable`, types
  belonging to the older `google-generativeai` stack.
  `langchain-google-genai` 4.x runs on the `google-genai` SDK, which raises
  `google.genai.errors.ClientError` / `ServerError` carrying a `.code`, so the
  check matched nothing and Gemini's quota throttling — the failure the 10s
  backoff base was written for — went straight to a lost batch. `api_core` is
  still installed as a transitive dependency, which is why the dead branch
  stayed invisible: it imported fine, and the tests constructed
  `ResourceExhausted` by hand. Now classified on the status code (`.code`, with
  a leading-status text fallback for the failures the langchain wrapper
  re-raises as `ChatGoogleGenerativeAIError` with the status only in the
  message). Retryable: `{429, 500, 503, 504}` — 429/503 restore the intended
  policy exactly, while 500/504 are a deliberate addition (the SDK raises them
  as plain `ServerError` with no api_core analogue, and Gemini returns them on
  overload).
- **Transport failures that never reached a server were treated as fatal by the
  two providers whose SDK doesn't wrap them** — a DNS blip or read timeout on
  NIM (`requests`) or Google (`httpx`/`requests`) carries no HTTP status, so
  neither classifier could see it, and a whole batch (plus the tokens already
  spent on it) was discarded on attempt 1. Both now share
  `TRANSPORT_TRANSIENT_ERRORS` (mixins.py), which names the `httpx` and
  `requests` timeout/connection types; the OpenAI-compatible providers already
  covered this via `APIConnectionError`/`APITimeoutError`. Both libraries are
  installed transitively, so no new dependency.
- **⚠️ A whitespace-only API key passed `--validate` with every check green** —
  each provider's presence test was a bare `if not api_key`, and Pydantic's
  `min_length=1` accepts `"   "` (a whitespace-only string is truthy), so the
  loader registered the provider and `validate_credentials` reported *every*
  check as passing. The failure was deferred to a 401 on the first real API
  call — exactly the outcome the pre-flight check exists to prevent. All seven
  key-checking providers now share `is_blank()` (mixins.py), which strips before
  testing, applying the same normalization `is_placeholder_api_key` already did.
- **`--validate` hard-failed on an uppercase-scheme URL the client accepts
  fine** — `require_https` (enforced at client construction) lowercased before
  comparing, while every provider's `validate_credentials` did a plain
  `startswith("https://")`. URL schemes are case-insensitive per RFC 3986 §3.1,
  so `HTTPS://api.deepseek.com/v1` built a working client that `--validate` then
  reported as a cleartext-endpoint failure. Two spellings of one predicate that
  had drifted; both now call `is_https_url()` (mixins.py), so the constructor
  and the pre-flight check cannot disagree about a URL in either direction. The
  inline `len(api_key) < 20` short-key warning, duplicated across five
  providers, moved to `is_short_api_key()` alongside it. Locked by
  `tests/test_provider_result_shape_contract.py`, including
  `test_every_url_checking_provider_uses_the_shared_https_predicate`, which
  scans the provider modules for an inline `startswith("https://")` so a new
  provider can't reintroduce the divergence.
- **The per-file token memo stopped paying off on large repositories** —
  `_TOKEN_CACHE_SIZE` was 4096, a plausible file count for a monorepo, and a
  single run estimates every scanned file at least twice (`--dry-run`'s table,
  then `create_batches`' packing loop). At a bound below the file count the
  first pass evicted its own earliest entries before the second pass reached
  them, so every file was re-encoded — the memoization silently degraded to
  nothing exactly in the repositories it was added for. Raised to 100,000
  entries (a run bound, not a "typical repo" bound); an entry is three ints plus
  a path string, so the added headroom costs tens of MB at most against a file
  list already held in memory. Locked by
  `tests/test_batcher.py::test_token_cache_bound_exceeds_a_large_repo_file_count`.
- **⚠️ `FileScanner` pruned directories a path-qualified exclude pattern never
  named, silently dropping files from the review** — `_get_excluded_dir_names`
  extracted the last literal segment of any pattern ending in a wildcard, so
  `docs/api/*` contributed the bare name `api` to the `os.walk` prune set.
  Pruning is by *bare name* and therefore matches at **any** depth: an
  unrelated `app/api/` elsewhere in the tree was skipped entirely, and the
  files were never scanned, never counted, and never reported as skipped —
  the run simply reviewed less code than it claimed to. `a/b/c/**` had the
  same effect on every directory named `c`. A prune name is now extracted only
  from an **unanchored** pattern (every segment before it a wildcard, or
  none), which is the only shape a bare-name prune can faithfully express;
  path-qualified patterns still exclude their files through `_is_excluded`,
  just walked rather than skipped. All 22 default patterns yield the same 14
  prune names as before, so the traversal optimization is unchanged for the
  default configuration. Locked by four tests in `tests/test_scanner.py`,
  including one that derives the expected names from
  `DEFAULT_EXCLUDE_PATTERNS` rather than hardcoding an example.
- **`--validate` could report *every* Bedrock model as available** — the
  model-access check tests `base_model_id in m or m in base_model_id` against
  each `modelSummaries` entry, and `m` came from `.get("modelId", "")`. A
  single summary without a `modelId` yields `""`, and `"" in base_model_id` is
  always true, so one malformed entry in the API response turned the check
  into an unconditional pass — the opposite of what a validation step is for.
  Id-less summaries are now dropped before the membership test, and the
  predicate is guarded on both operands. Locked by
  `tests/test_validation.py::TestBedrockModelAvailabilityMatching`.
- **Two `models.yaml` entries in the same provider claiming one name failed
  silently** — `ConfigLoader._register_model` warns on *cross-provider*
  collisions but returned quietly for intra-provider ones, leaving one entry
  unreachable with no diagnostic. Last-write-wins resolution is deliberate
  (it's how generation-neutral aliases like `opus` move to a newer entry), so
  the fix is the missing warning, not a resolution change: the log now names
  both entries, which one won, and which is now unreachable. Re-registering
  the *same* entry stays quiet. Locked by three tests in
  `tests/test_config.py`, one of which fails if the real registry ever
  develops an intra-provider duplicate.
- **Typing `yes` at the README-context prompt discarded the README** — the
  prompt reads `[Y/n/path]` and its third option is a file path, so anything
  unrecognized was treated as one: `yes` printed `File not found: yes` and the
  run continued with no project context, having just been told to use it. Both
  prompts now accept the spelled-out `yes`/`no` alongside `y`/`n`
  (case-insensitively, after stripping).
- **A literal `None` could appear in the streamed output** — content blocks
  arrive as dicts and `part.get("text", "")` returns `None` when the key is
  present but null (providers emit that for reasoning summaries and tool-call
  deltas; the default only covers a *missing* key), so `str()` spliced `"None"`
  into the live panel. `StreamingCallbackHandler._block_text` now checks the
  value instead of defaulting it.
- **Documented CI gates in `docs/examples.md` and `docs/usage.md` never
  failed** — the GitHub Actions, GitLab CI, and shell-parsing examples read
  `jq '.metrics.critical // 0'`, but the JSON field is `critical_issues`
  (likewise `high_issues`, `medium_issues`, `low_issues`, `info_issues`). The
  `// 0` fallback masked it perfectly: the expression always evaluated to `0`,
  so `if [ "$CRITICAL" -gt 0 ]` was never true and anyone who copied these
  snippets had a quality gate that passed unconditionally. The PR-comment
  script in the Actions example had the same bug, rendering an all-zero
  severity table. Fixed in all four places, and the examples now use
  `--fail-on` instead of hand-rolled `jq` parsing so the field name isn't a
  correctness dependency for a CI gate. The upload/artifact steps gained
  `if: always()` / `when: always` so the report survives a failing gate.
- **Two models listed their own `id` as an alias** — `gpt5.5-bedrock` and
  `deepseek-v4-flash-nvidia` each repeated their id in `aliases`, so
  `--list-models` printed the same name twice in adjacent columns. Now
  structurally impossible: `ModelConfig._check_alias_hygiene` (a Pydantic
  `mode="after"` validator) rejects a self-alias or a name repeated across
  `aliases`/`deprecated_aliases` at config-load time.
- **`--list-models` truncated long aliases into invalid spellings** — the
  Aliases column rendered `claude-opus-4.…` and `qwen-coder-nex…`, which fail
  as typed, so the table was advertising names that don't work. The column now
  sets `overflow="fold"` **and** a `min_width` computed from the longest alias
  in the registry; `fold` alone still splits a name across lines at the
  80-column default. Locked by
  `tests/test_cli.py::test_list_models_never_truncates_an_alias`.
- **Eight duplicated alias-registration blocks in `ConfigLoader`** collapsed
  into one `_register_all_names` helper. Each provider branch had its own
  hand-rolled id-then-aliases loop, so wiring the new `deprecated_aliases` key
  would have meant eight edits and one missed provider would have silently
  registered half a model's names.
- **Markdown export was corrupted by an unclosed code fence in any
  model-generated prose field** — a model that opened a ```` ``` ```` in
  `summary`, `description`, `rationale`, `system_design_insights`,
  `recommendations`, or `improvement_suggestions` without closing it swallowed
  every following section of the report into one code block, so the exported
  artifact silently lost its issues, metrics, and recommendations. New
  `balance_code_fences` (renderer.py) counts fence *lines* and appends the
  missing closer; nested/longer fences and inline triple-backtick spans are
  handled. `suggested_code` already had its own fence-widening and is
  unchanged. Locked by the fence tests in `tests/test_markdown_export.py`.
- **NVIDIA built a rate limiter and never used it** — `_build_rate_limiter`
  populated `self.rate_limiter` but `_create_model` omitted it from
  `model_params`, and an `InMemoryRateLimiter` only throttles the LangChain
  client it is attached to. Concurrent batches therefore hit NIM unthrottled
  (429s on the free tier). NVIDIA was the only provider with this gap; locked
  by `tests/test_nvidia_provider.py::test_rate_limiter_is_attached_to_the_client`.
- **Token-budget fallback was invisible without `--verbose`** — when the
  computed budget went non-positive the CLI silently dropped to count-only
  batching, which can overflow the model's context window mid-run. That
  changes what the review does, so the warning now prints on every run (with
  the computed value and a remediation hint) rather than only under
  `--verbose`. Locked by the two `test_token_budget_fallback_*` tests in
  `tests/test_cli.py`.
- **OpenAI-on-Bedrock accepted the placeholder key the README documents** —
  `validate_credentials` rejected only `your-bedrock-api-key-here`, but
  README.md's export line is
  `OPENAI_API_KEY="<your-amazon-bedrock-api-key>"`, so a copied-and-not-
  replaced placeholder passed `--validate` and failed later with a 401 —
  exactly the CLAUDE.md contract this check exists to satisfy. Both spellings
  (with and without the angle brackets) now hard-fail. A new drift guard,
  `test_every_readme_documented_placeholder_is_rejected`, scrapes every
  `export *_API_KEY="..."` line out of README.md and asserts the owning
  provider rejects it, so rewording an export line or adding a provider fails
  the suite until the placeholder set catches up.
- **`locals()`-based lookup for the Bedrock access-denied message** replaced
  with a normal binding — `model_display_name` is now seeded with the raw
  `--model` argument before the `try` block and upgraded once resolved, so the
  handler reads a real variable instead of probing the frame dict (which
  static analysis and refactoring tools cannot follow).
- **⚠️ Static analysis executed code from the reviewed repository** — mypy
  imports the module named by a `plugins =` entry, an `eslint.config.js` *is*
  JavaScript, and Prettier loads the modules named under `plugins`. Reviewing an
  untrusted tree ran all three with the user's privileges before any review
  output existed. Reproduced against the installed binaries (mypy 1.19.1,
  ESLint v10.8.0, Prettier 3.x): each ran attacker-supplied code, and mypy still
  reported `passed: True`. `run_tool` now consults `_find_executable_config`
  **before building the command** and returns a failed result naming the config,
  stating that the tool's findings are missing from the review. Detection is on
  *content*, not filename — an ordinary `pyproject.toml` with a `[tool.mypy]`
  section but no `plugins` entry still runs mypy — and fails closed on an
  unreadable or >512 KB config. New `--trust-repo-config` opts back in. Locked by
  the `_CONFIG_EXECUTION_RISK` tests in `tests/test_static_analysis.py`, which
  patch `subprocess.run` and assert it is never called (asserting on the *result*
  would pass even if the tool had already run), plus
  `test_this_repository_is_not_false_positived`.
- **⚠️ Static-analysis findings were attributed to the wrong file** —
  `_path_match_token` built a two-component `parent/basename` token and matched
  it as a substring, so a finding in `other/api/views.py` was reported against
  `app/api/views.py`: `api/`, `utils/`, `models/` and `tests/` are exactly the
  directory names that repeat. The substring test additionally matched
  mid-component (`foo.py` inside `foo.py.orig`). Both are now one mechanism:
  the token keeps the **whole** normalized path and `_line_mentions_any_path`
  compares component tuples from the right, at component granularity. Boundary-
  aware substring matching would have fixed only the second shape — the
  information the first needs was already discarded when the token was built.
  Tolerant of absolute↔relative, `./`, `//` and `\` spellings in both
  directions; locked by the four `condense_for_prompt` filter tests.
- **⚠️ The model's own recommendations were collected from no batch** — the
  aggregation loop gathered `issues`, `improvement_suggestions` and
  `system_design_insights` but dropped `recommendations`, and
  `_generate_recommendations` substituted severity/category counts. So a run
  that found an SQL injection at `views.py:42` recommended "🔒 Resolve 1
  security issue(s)": no file, no line, no title, and the same numbers the
  Metrics section already prints — while `SYSTEM_PROMPT` explicitly asks for
  recommendations "DERIVED FROM the issues you reported. Reference issue titles,
  not new ideas". The model's text now wins, deduplicated across concurrent
  batches (same normalization `_dedupe_design_insights` uses) and capped at 5;
  the count summary remains as the fallback for runs where no batch emitted any.
- **⚠️ A linter that couldn't run was reported as a clean pass** — exit code 2
  from ruff/black/mypy means "bad config / missing plugin / bad arguments", not
  "found problems", so a repository whose ruff config named a nonexistent rule
  contributed zero coverage while reporting a tidy issue count. New
  `_OPERATIONAL_FAILURE_EXIT_CODES` maps the distinguishable cases and yields
  `issues_count=0` plus an explicit error. Verified against the installed
  binaries rather than the docs: isort exits 1 for both an invalid config and a
  mis-sorted file and vulture exits 3 for findings, so neither is classifiable
  and neither is listed — an ambiguous exit keeps being treated as a finding.
- **⚠️ `npm audit` reported zero vulnerabilities whenever npm printed a
  warning** — `_count_npm_audit_issues` was fed `stdout + stderr`, and npm
  writes routine notices (`npm warn …`) to stderr, so one warning made the
  `--json` payload unparseable and the count silently fell to 0: a clean bill of
  health for a repo with real advisories. Parsing now reads `result.stdout`
  only; stderr still reaches the human-readable output.
- **⚠️ Retryable Bedrock throttling was given up on as a hard failure** —
  `_is_retryable_error` matched too narrow a set of spellings, so
  `TooManyRequestsException` / `ServiceUnavailable` / `ModelTimeout` and the
  equivalent `ClientError` codes ended the batch instead of backing off. Visible
  only as a lost batch in a partial-results run.
- **Tokens the provider billed but the parser rejected went unreported** —
  `_extract_token_usage` now runs on the raw `AIMessage` even when `parsed` is
  `None` (a schema violation, or a reasoning-only response). The provider
  charged for them either way, so omitting them under-reported real cost by
  exactly the retried batches — the expensive ones — while `--dry-run`
  estimates looked accurate.
- **`--validate` accepted a hostless `https://` endpoint** — `is_https_url`
  checked the scheme only, and `urlparse("https://")` yields
  `scheme == "https"` with an empty host, so a truncated or half-substituted
  `base_url` passed the cleartext check and surfaced as a connection error
  mid-run. A hostname is now required.
- **`files_analyzed` counted scanned files, not reviewed ones** — the batcher
  drops oversized files after the scan, so the metric (and the summary line)
  claimed coverage the review didn't have: "Analyzed 120 files" for a run that
  reviewed 118. `--dry-run` billed those bytes too, quoting ~1,000,350 input
  tokens for a run that sends 300 when one oversized file was present. Both now
  derive from batch membership, the only authority on what was sent.
- **`--temperature` validated too late to help** — `_resolve_temperature`
  enforced the 0.0–2.0 range, but it runs inside provider construction, i.e.
  after the scan, the line count, static analysis and batching. So
  `--temperature 99` spent all of that work (with `--static-analysis`, minutes
  of linters) before failing with a provider error and exit 1, on an argument
  that was invalid before anything started. The option is now
  `click.FloatRange(0.0, 2.0)`, which rejects it at parse time with a usage
  error; the deep check stays as the guard for callers that don't come through
  Click.
- **Line counts dropped the last line of any file not ending in a newline** —
  counting `"\n"` occurrences alone undercounts by one per such file, so
  `total_lines` drifted low across a repository.
- **A four-backtick snippet still swallowed the rest of the Markdown report** —
  `suggested_code` is wrapped in a *wider* fence, but a snippet already
  containing ```` ```` ```` made that wrapper no wider than its content.
  CommonMark closes a block only on a fence line of at least the opener's width
  *and* with no info string, so fence **parity** is the wrong model whenever
  widths vary: a numerically balanced document can still trap every following
  section. The wrapper now exceeds the widest fence in the content. The
  regression test previously asserted parity and passed under its own
  mutation — it now runs a real CommonMark state machine and asserts no section
  marker lands inside a code block, and that the document doesn't end inside
  one.

- **Corrected the documented reason for `supports_tool_use: false` on the Bedrock
  Claude entries** — `CLAUDE.md`, `models.yaml` (fable5 and opus5), and two tests
  all asserted "Anthropic allows only `tool_choice: auto/none` while thinking".
  That rule is real but **scoped to *manual* `thinking: {type: "enabled"}`**.
  Anthropic's thinking documentation states the opposite for the models this
  project actually ships: *"Adaptive thinking, including on models where thinking
  is on by default, supports forced tool use."* langchain-aws encodes the same
  scoping — `thinking_forced_tool_use_unsupported()` returns `False` for
  `claude-opus-4-8` outright and never listed Opus 5 / Sonnet 5 / Fable 5, and it
  engages only when a `thinking` key is present in the request.

  **No behavior change**: the observed failure is real and reproduced live on
  Opus 4.8 (`de5e2fc`) — a forced `tool_choice` returns the tool call as literal
  `<invoke name="issues">…` markup, so `CodeReviewReport.issues` fails Pydantic
  validation with a `list_type` error on think-heavy batches — and Opus 5 has
  independent vendor confirmation (its model card lists *Structured outputs: Not
  Supported*). `supports_tool_use: false` stays correct on all four entries. What
  changed is that the flag is now described as **empirical**, with the vendor rule
  it was misattributed to explicitly disclaimed at each site, so the next reader
  doesn't build on it. This nearly became a langchain-aws bug report against
  behavior that is upstream-correct by design.

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

### Changed
- **`bedrock_openai` and `moonshot` no longer duplicate the base class's
  structured-output routing** — both re-implemented the
  `supports_tool_use` → `with_structured_output(..., include_raw=True)` branch
  before falling through to `_apply_structured_output`, which does exactly
  that. CLAUDE.md states the routing "lives once in `base.py`"; two copies
  meant a future change to the tool-use path would need three edits. Both now
  end `_create_model` with a single `return self._apply_structured_output(...)`.
  Behavior is unchanged and was already covered on both paths by the existing
  provider tests.
- **Google GenAI `analyze_batch` docstring** said `max_retries` defaulted to 3
  while the code uses 5 (preview models throttle hard); the docstring now
  matches the code.
- **`run_review()` extracted from `main()`** (`cli.py`) — `main` was 704 lines
  doing Click parsing, three early-exit flags, and the entire review pipeline,
  so every test of pipeline behavior had to go through `CliRunner.invoke` and
  assert on rendered output. `main` now keeps only argument parsing and the
  flags that exit before any review work (`--list-models`, `--validate`,
  no-directory help); `run_review(directory, *, console, ...)` owns scanning
  through the `--fail-on` gate. The 650-line pipeline body moved **verbatim** —
  no logic changed — and the parameters are keyword-only with the same defaults
  as the Click options, so `main` is a thin pass-through.
  - `--fail-on` remains the last statement in the run, after export.
    `test_run_review_applies_the_gate_after_writing_the_report` now asserts
    that by *observed ordering* (export event, then the `SystemExit`) rather
    than inferring it from a file existing afterwards.
  - `test_run_review_defaults_match_the_click_option_defaults` compares what a
    bare CLI invocation actually forwards against `run_review`'s signature, and
    asserts every reviewable option is forwarded — so a new Click option
    `main()` forgets to pass through fails the suite.
- **Per-file token counts are memoized on `(path, size, mtime_ns)`**
  (`batcher.py`) — `FileBatcher.estimate_file_tokens` ran a full tiktoken
  encode over each file's text, and a single run estimates every file at least
  twice (`--dry-run` builds its per-file table, then `create_batches` packs the
  same list). On this repo's own 26 files that second pass cost ~348 ms of pure
  re-encoding; it is now ~0.15 ms. A changed file's `(size, mtime_ns)`
  invalidates its entry, so no stale count can survive an edit. Deliberately
  caches only the **count**, never the file's text: batches run concurrently in
  a `ThreadPoolExecutor`, and holding every scanned file's contents alive for a
  whole run would trade a bounded number of re-reads for unbounded memory.
  `clear_token_cache()` is the escape hatch for tests that rewrite a file
  in place.
- **⚠️ Retry counts now come from the provider, not the analyzer — a real
  behavior change for six providers.** `CodeAnalyzer.analyze_batch` defaulted
  `max_retries` to a hardcoded `3` and forwarded it *unconditionally*, so the
  `None` sentinel every provider was written to honour never arrived: each
  provider's own default was dead code and `NVIDIAConfig.max_retries` was
  unreachable config that `--verbose` would never reflect. A Bedrock-tuned
  count (throttling clears in a couple of attempts) was silently applied to
  NVIDIA NIM's frequent gateway 504s, Azure's quota windows, and the
  OpenAI-compatible endpoints. `max_retries=None` is now the "provider decides"
  sentinel end to end.
  - **Effective retries per batch change from 3 → 5** for Azure OpenAI, NVIDIA
    NIM, Google GenAI, DeepSeek, Z.AI, Moonshot, and OpenAI-on-Bedrock. Bedrock
    stays at 3. Backoff base waits are unchanged, so a fully-exhausted batch on
    a rate-limited endpoint now takes longer before failing — which is the
    intent: those endpoints were being given up on early.
  - New `ModelProvider._resolve_max_retries(override, provider_config, default)`
    mirrors the existing `_resolve_temperature` precedence
    (`override > provider_config.max_retries > provider default`) and rejects a
    negative override. NVIDIA's and Google's hand-rolled `if max_retries is
    None` blocks were replaced by it.
  - Passing an explicit `max_retries` still overrides everything, so the
    existing error-handling tests that pin a count are unaffected.
  - Five guards in `tests/test_provider_result_shape_contract.py`:
    `test_none_max_retries_uses_the_providers_own_default` and
    `test_explicit_max_retries_overrides_the_providers_default` (both
    parametrized over all eight providers),
    `test_nvidia_config_max_retries_is_live_config` (a non-default `9` must
    reach `RetryConfig`), `test_analyzer_defers_the_retry_decision_to_the_provider`,
    and `test_every_provider_analyze_batch_defaults_max_retries_to_none`, which
    reflects over every provider class so a new one that hardcodes a signature
    default — the exact shape of this bug — fails the suite.

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
- **Registry cleanup: 11 model entries pruned from `models.yaml`** (registry
  goes 41 → 30 models). Every surviving and candidate entry was probed against
  its **live provider endpoint** rather than judged from release notes, and only
  entries that were dead, unreachable from the configured region, or strictly
  superseded at equal-or-worse price/context were dropped. Each removal leaves a
  dated comment in `models.yaml` explaining the verification result.
  Version-neutral aliases were **migrated to a live successor**; the
  version-explicit ones were **deleted** so a stale `--model` spelling errors
  instead of silently running a different model (see Changed and the alias
  deletion entry below).
  - **AWS Bedrock — Claude Opus 4.7** (`opus4.7`) and **Claude Opus 4.6**
    (`opus4.6`): both still ACTIVE on Bedrock, but superseded by Opus 4.8/Opus 5
    at identical $5/$25 pricing (4.6 additionally capped at a 200K context).
  - **AWS Bedrock — Qwen3 Coder 480B** (`qwen-bedrock`,
    `qwen.qwen3-coder-480b-a35b-v1:0`): not offered in this provider's
    `us-east-1` region — verified against the live `ListFoundationModels`
    catalog, where it exists only in `us-west-2` (in which
    `qwen-next-bedrock` is in turn absent). `qwen-next-bedrock` is the
    supported Qwen coding model here.
  - **Azure OpenAI — Kimi K2.5** (`kimi-k2.5-azure`) and **DeepSeek-V4-Pro**
    (`deepseek-v4-pro-azure`): both return `DeploymentNotFound` on the
    configured Azure resource (verified live). Unlike Bedrock/NVIDIA, an Azure
    entry only works if someone has explicitly created a deployment with that
    exact name, and such failures are invisible to `--list-models`. Both models
    remain reachable via their canonical direct providers.
  - **NVIDIA NIM — MiniMax M2.7** (`minimax-m2.7-nvidia`): endpoint still live,
    but superseded by MiniMax M3 on the same free endpoint (1M context vs 204K,
    multimodal).
  - **NVIDIA NIM — Qwen3 Coder 480B** (`qwen-nvidia`,
    `qwen/qwen3-coder-480b-a35b-instruct`): **gone from the live NIM catalog**
    (verified against `GET /v1/models`) — a dead endpoint, not merely
    superseded. Qwen3.5 397B is the current Qwen on NIM.
  - **NVIDIA NIM — Step 3.5 Flash** (`step-3.5-flash`): endpoint still live,
    superseded by Step 3.7 Flash at the same 256K context and free tier, with
    multimodal input and reasoning levels.
  - **Google GenAI — Gemini 3 Flash Preview** (`gemini-3-flash`,
    `gemini-3-flash-preview`): deprecated upstream (announced 2025-12-17) with
    `gemini-3.6-flash` named as the replacement. Still answers on the API, so
    re-add from git history if the cheaper $0.50/$3.00 rate is worth the
    deprecation risk.
  - **Z.AI — GLM-5.1** (`zhipuai/glm-5.1`): superseded by GLM-5.2 at the *same*
    $1.40/$4.40 list price with a 1M context instead of 203K — no reason to pick
    5.1. Still live on Z.AI.
  - **OpenAI-on-Bedrock — GPT-5.4** (`gpt5.4-bedrock`): two newer OpenAI
    generations ride the same `bedrock-mantle` endpoint (GPT-5.5 at identical
    $2.50/$15, and GPT-5.6 Sol), so the 5.4 entry added no capability. Still
    responds on the endpoint (probed HTTP 200). **GPT-5.4 on *Azure* is a
    separate entry and stays** — that's a user-created deployment and the newest
    GPT on that resource.
  - Guards added/updated in `tests/test_config.py`: `DEAD_UPSTREAM_FULL_IDS`
    (dead or region-unreachable wire ids), the new
    `DEAD_AZURE_DEPLOYMENT_NAMES` / `test_no_model_points_at_missing_azure_deployment`
    (Azure failures only surface at invocation time), the expanded
    retired-alias redirect test, and the new
    `test_no_historical_model_id_is_orphaned` git-history sweep. Superseded-but-live endpoints are deliberately
    *not* in `DEAD_UPSTREAM_FULL_IDS` — re-adding those is a judgement call, not
    a bug.
- **40 model aliases deleted from `models.yaml`** (registry goes 143 → 101
  resolvable names: 84 advertised + 17 deprecated). Two separate problems:
  version-explicit aliases of removed models that had been redirected to a
  *successor*, and cryptic short forms of live models that were pure noise in
  `--list-models`.
  - **Version-explicit redirects, now failing fast** — `opus4.7`, `opus-4.7`,
    `claude-opus-4.7`, `claude-opus-47`, `opus4.6`, `opus-4.6`,
    `claude-opus-4.6`; `minimax-m2.7`, `minimax-m2.7-nvidia`, `mm2.7-nvidia`,
    `mm27`, `minimax-m2.5-nvidia`, `mm2.5-nvidia`, `minimax-m2.5`, `mm25`;
    `kimi-k2.5-nvidia`, `kimi-k2.5`, `kimi25`; `glm51`, `glm51-nvidia`,
    `glm-5.1`, `glm5.1`, `glm5.1-zai`, `zhipuai/glm-5.1`; `step35`,
    `step-3.5-flash`; `gpt5.4-bedrock`. **Breaking** for anyone scripting these
    — `--model opus4.6` now errors instead of silently running Opus 5.
  - **Redundant short forms of live models** — `gpt54p` (use `gpt54-pro`),
    `glm5b` (`glm5-bedrock`), `gpt5.6-sol` and `sol` (`gpt5.6`), `dsv4f`
    (`dsv4-flash`), `dsv4pro` (`dsv4-pro`), `dsv4-azure` (`deepseek-v4-pro`),
    `g31pro`/`g3pro` (`gemini31-pro`), `g36flash` (`gemini36-flash`),
    `mm35`/`mmed` (`mistral-medium`), `kimi-moonshot` (`kimi`).
  - Full replacement table: **Migrating Deleted Aliases** in `README.md`.
    Every deleted name is listed in `RETIRED_ALIASES_DELETED_NOT_REDIRECTED`
    (`tests/test_config.py`) with a reason, and
    `test_deleted_aliases_do_not_resolve` asserts each one raises.
  - `test_documented_model_names_all_resolve` scrapes every `--model X` out of
    `README.md`, `docs/usage.md` and `docs/examples.md` and resolves it through
    the loader. Docs had been advertising deleted aliases as "route here"; now
    prose that names a dead spelling in a runnable command fails CI.
  - Dated notes were left in `models.yaml` at each removal site recording what
    was deleted and what to use instead, so re-adding from git history stays a
    judgement call with the evidence attached.
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
- **GLM-5.1 (NVIDIA) entry** `glm51` (`z-ai/glm-5.1`) — NVIDIA deprecated the
  free `bedrock-mantle`/NIM endpoint (~2026-07) and superseded it with
  `z-ai/glm-5.2`. The new GLM-5.2 (NVIDIA) entry absorbs the version-neutral
  `glm5`/`glm-5`/`glm5-nvidia` spellings; the version-explicit `glm51`,
  `glm-5.1`, `glm5.1` were deleted (see Removed). The Z.AI-direct
  `zhipuai/glm-5.1` entry was removed in the same cycle.

### Documentation
- Trimmed overall documentation footprint from ~5,600 to ~4,500 lines
- Updated `CLAUDE.md` and `README.md` to v0.3.1-current state: correct
  default model (now Opus 5), correct test count, new model tables
- Added `.ruff_cache/` to `.gitignore` to match existing cache ignores

### Quality
- Test suite: **1156 passing** (up from 319; +17 code-review-triage regressions
  — markdown code-fence balancing, NVIDIA rate-limiter wiring, the
  token-budget-fallback warning, and the README-placeholder drift guard;
  +8 cross-provider cleartext-endpoint contract, incl. a self-checking registry
  guard that fails when a provider calls `require_https` without appearing in
  it; +2 Provider-Setup-table vs `models.yaml` env-var drift; +5 token-count
  memoization; +6 `run_review` seam incl. gate-after-export ordering and the
  `main`-is-a-pass-through default/forwarding guard;
  +19 `max_retries` resolution contract, incl. the reflective guard against a
  provider hardcoding a signature default;
  +23 second-round code-review triage — scanner bare-name over-pruning (4,
  one deriving its expectations from `DEFAULT_EXCLUDE_PATTERNS`), the Bedrock
  empty-`modelId` substring match that confirmed every model (2), the
  intra-provider name-conflict warning incl. a guard on the real registry (3),
  the README prompt's spelled-out `yes`/`no` asserting on the bogus
  "File not found" line rather than just the return value (11), and
  `{"text": None}` content blocks (3);
  +48 shared `validate_credentials` validator contract — the whitespace-only-key
  false-positive across five providers (15), the uppercase-scheme URL the
  constructor accepts but `--validate` rejected (4), a reflective guard against a
  provider reintroducing an inline `startswith("https://")` (1), and the
  `is_blank` / `is_https_url` / `is_short_api_key` helper units incl. the
  `is_https_url` ≡ `require_https` agreement check (28);
  +1 token-memo bound guarding the two-pass estimate;
  +10 alias-hygiene / deleted-alias /
  deprecated-alias-display / documented-model-name guards, +14 `Correctness`
  category and
  drift-counter guards, +8 `--fail-on` gate incl. the
  `--severity`-must-not-weaken-the-gate regression test, +2 scanner `exclude_hidden`,
  +3 supply-chain `_resolve_tool_binary`, +5 ruff/mypy/bandit counters,
  +2 Azure `supports_tool_use=false`, +9 Z.AI provider, +11 DeepSeek
  provider, +10 Moonshot provider, +5 truncation/timeout/batcher resets,
  plus OpenAI-on-Bedrock provider, Sonnet 5, GLM-5.2, Grok 4.3, GPT-5.6 Sol,
  Opus 5, Gemini 3.6 Flash, and the retired-alias / deleted-alias /
  orphaned-historical-id / dead-upstream / missing-Azure-deployment registry
  guards);
  +33 third-round code-review triage — the repo-config code-execution gate
  (18, incl. a `subprocess.run`-never-called assertion and a
  this-repository-is-not-false-positived guard), component-granular path
  matching for prompt condensation (10), model-recommendation aggregation (4),
  and `--trust-repo-config` CLI forwarding (1);
  +109 cross-provider retry contract (`tests/test_retry_contract.py`) — an
  HTTP-status × provider matrix, throttling-is-always-retryable and
  4xx-is-never-retryable sweeps, transport-failure cases, backoff bounds and
  monotonicity, a reflective guard so a new provider can't skip retry coverage,
  and the meta-guard that found both dead classifiers by feeding each provider
  the 429 its *own installed client* builds. Each error is constructed the way
  its real client constructs it — the NIM builder routes a genuine
  `requests.Response` through `_NVIDIASyncClient._try_raise` — because the
  previous hand-built exceptions of a type the SDK no longer raises are exactly
  what let the dead code pass for so long;
  +52 table-driven credential validation over the axes the per-provider
  placeholder tests don't cover (`tests/test_placeholder_keys.py`) — blank and
  whitespace-only keys × 7 providers × 5 spellings, cleartext and hostless URLs
  × every URL-taking provider (fail-closed at construction, before a client
  exists), padded-but-real keys that must be *accepted* after normalization, and
  a reflective guard so a new `api_key`-taking provider inherits all three axes;
  +91 fourth-round code-review triage — recursive-`**` exclusion matching, the
  markdown-export `RuntimeError` handler, `models.yaml` parse errors naming the
  offending entry, the five legacy `codereview.config` constants going lazy,
  short-key warnings on Google/NVIDIA, NVIDIA's doubled-slash probe URL, the
  `ProviderFactory` registry table, and the `--temperature` range moving to
  parse time;
  +24 token-usage contract (`tests/test_token_usage_contract.py`) — an
  extractor × provider matrix in which every `AIMessage` is built by the **real
  vendor client** from a recorded wire payload (Chat Completions and Responses
  API through `BaseChatOpenAI`, `_parse_response` for Bedrock Converse, a
  genuine `requests.Response` for NIM, `_response_to_result` for Google), an
  end-to-end assertion that the counts survive `_execute_with_retry` into the
  provider's totals, two reflective guards (every provider in the matrix; every
  `use_responses_api`-capable provider exercised on that path), and the
  `Retry-After`-on-5xx case. Same reasoning as the retry contract: the
  pre-existing hand-built `AIMessage` tests invented the one field the broken
  extractor read, so they passed for exactly as long as it was wrong;
  two pre-existing fixtures fixed — they passed kwargs that Pydantic
  silently dropped
- **The `slow` pytest marker is now registered** (`[tool.pytest.ini_options]
  markers`). An unregistered mark only warns, and that warning was the *only*
  signal distinguishing correct usage from a typo — `@pytest.mark.slwo` marks
  nothing and looks identical. Registering it also clears the suite's one
  self-inflicted warning; the remaining one is upstream
  (`google.genai.types` / `_UnionGenericAlias` under Python 3.14).
- `ruff check`, `ruff format --check`, `mypy`: clean
- New runtime dependencies: `langchain-deepseek>=1.0.1`,
  `langchain-moonshot>=0.1.0` (both small single-purpose packages, not
  the heavy `langchain-community`)
- **Dependency floors raised to the current latest release** and the lockfile
  upgraded (`uv lock --upgrade` + `uv sync`): langchain-core 1.4.9 → 1.5.1,
  langchain-openai 1.3.5 → 1.4.1, langchain-aws 1.6.2 → 1.6.3,
  langchain-google-genai 4.2.7 → 4.3.1, langchain-deepseek 1.0.1 → 1.1.0,
  google-api-core 2.32.0 → 2.33.0, boto3 1.43.51 → 1.43.56,
  pydantic 2.13.2 → 2.13.4, pyyaml 6.0 → 6.0.3, tiktoken 0.12.0 → 0.13.0;
  static-analysis black 26.3.1 → 26.5.1, ruff 0.15.22 → 0.16.0,
  types-PyYAML → 6.0.12.20260724; dev pytest 9.0.3 → 9.1.1, pytest-mock
  3.15.0 → 3.15.1. Transitive upgrades come with the lockfile (openai
  2.46.0 → 2.48.0, google-genai 2.11.0 → 2.14.0, langsmith, aiohttp, …).
  `<major` caps and the exact `langchain-moonshot==0.1.0` pin are unchanged.
  Raising the `ruff` floor to 0.16.0 is safe *because* the rule set is now
  pinned in `[tool.ruff.lint]` — the floor no longer decides the gate's verdict.
  Verified on the upgraded stack: all 1156 tests plus all five gates clean,
  `--list-models` and `--dry-run` smoke-tested. `pydantic-core` and `websockets`
  stay behind their latest on purpose — pinned by `pydantic` and `google-genai`
  respectively.

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
