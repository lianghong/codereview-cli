# Provider internals

Background for the provider rules in `CLAUDE.md`. Read the section that covers what you're
about to change. See also `docs/structured-output.md` for the tool-use-vs-prompt-parsing
routing and `docs/validation-contract.md` for `--validate`.

## Contract: public API vs internal hooks

When implementing a provider, know which methods callers invoke versus which the base class
calls into.

| Method | Role | Notes |
|---|---|---|
| `analyze_batch` | **public** | The single entry point `CodeAnalyzer` calls. Build `chain_input`, then delegate to `_execute_with_retry` (don't reimplement the retry loop). Keep `max_retries: int \| None = None` and resolve it with `_resolve_max_retries` — a concrete default here overrides the provider layer for every caller that doesn't pass one. |
| `validate_credentials` | **public** | Called by `--validate`. Follow the hard-fail vs warning contract in `docs/validation-contract.md`. |
| `get_pricing` / `get_model_display_name` | **public** | Used by cost reporting and the renderer; `get_pricing` is mandatory for every provider. |
| `supports_token_streaming` | **hook (classmethod, optional)** | Defaults to `True`. Override to `False` — with the reason in the docstring — if the client never delivers a token to a callback, or `--stream` will serialize the run for nothing. Must stay answerable from the class (no `self`, no credentials): the CLI calls it before constructing the provider. `tests/test_streaming_contract.py::test_the_non_streaming_provider_set_is_exactly_the_documented_one` fails until the new provider is classified either way. |
| `_create_model` | **hook (required)** | Build the LangChain client. Enforce HTTPS here via `require_https` (`mixins.py`), **not** in `validate_credentials`: `_create_model` runs from `__init__`, so a caller that never calls `--validate` still can't ship an API key to `http://` (CWE-319). Pydantic's `HttpUrl` accepts `http://`, so `require_https` is the only thing enforcing it. Call it before the client is constructed — fail *closed*, with the credential never reaching a client instance. End with `return self._apply_structured_output(base_model)`. |
| `_create_chain` | **base-provided** | Default pipes the prompt template into the model, appending the `PydanticOutputParser` on the prompt-parsing path. Override only for genuinely custom chains. |
| `_extract_token_usage` | **hook (required)** | OpenAI-compatible providers should delegate to `extract_openai_token_usage` (mixins.py). |
| `_is_retryable_error` / `_calculate_backoff` | **hook (required)** | OpenAI-compatible providers should use `is_openai_retryable_error` + `parse_retry_after` (mixins.py); keep any provider-specific base-wait local (see Azure). |
| `_execute_with_retry`, `_prepare_batch_context`, `_build_batch_system_prompt`, `_resolve_temperature`, `_resolve_max_retries`, `_build_rate_limiter` | **base-provided** | Inherited from `ModelProvider`; call them, don't override unless you have a specific reason. `_build_rate_limiter` only *builds* the limiter — an `InMemoryRateLimiter` throttles nothing unless it is passed to the client, so `_create_model` must also put `"rate_limiter": self.rate_limiter` in `model_params`. NVIDIA built one and dropped it, which left concurrent batches hammering NIM until 429s. |

## Classify retryability on the HTTP *status*, not on an exception class

An `isinstance` check against a type the *installed* client never raises is dead code, and dead
retry logic is invisible: a misclassified throttle looks exactly like a lost batch, and the only
symptom is a review that silently covered fewer files.

Two classifiers were dead this way, both found by `tests/test_retry_contract.py` (109 tests)
and both fixed by reading the status:

- **NVIDIA** tested `httpx.HTTPStatusError`, but `langchain-nvidia-ai-endpoints` runs on
  `requests` and `_NVIDIASyncClient._try_raise` *discards* the typed error, re-raising a bare
  `Exception("[504] Gateway Timeout\n…")` (its own source carries a
  `# todo: raise as an HTTPError`) — so every NIM gateway 504, the exact failure NVIDIA raises
  `max_retries` for, aborted on attempt 1.
- **Google** tested `google.api_core.exceptions.ResourceExhausted`/`ServiceUnavailable`, but
  `langchain-google-genai` 4.x raises `google.genai.errors.ClientError`/`ServerError` — every
  429 and 503 aborted on attempt 1.

Both stayed invisible because the legacy package is still installed transitively (the import
succeeded) *and* because the tests hand-constructed exceptions of the legacy type, which passed
for exactly as long as the classifier was dead.

The rule those tests encode: **build the error the way the real client builds it** — `_nim_error`
routes a real `requests.Response` through the client's own `_try_raise`.

Each provider now reads the status off `.code`/`.response.status_code` with a text fallback for
the wrappers that keep it only in the message (`[504] …` for NIM, a leading `429` for
`ChatGoogleGenerativeAIError`), and transport failures with no status yet (DNS blip, read
timeout) come from the shared `TRANSPORT_TRANSIENT_ERRORS` tuple in `mixins.py`.

Provider status sets differ **on purpose** and the differences are recorded in `_RETRY_MATRIX`
rather than normalized: NIM's is exactly `{429, 502, 503, 504}` (a bare 500 there is usually a
request the gateway rejected, and `test_non_rate_limit_error_not_retried` pins it
non-retryable), Google's adds 500/504, Bedrock takes 5xx wholesale.

**Retryability has to be tested from the *outside***: `_is_retryable_error` classifies on the
exception's rendered text, and Bedrock's throttling arrives under several spellings
(`ThrottlingException`, `TooManyRequestsException`, `ServiceUnavailable`, `ModelTimeout`, plus
`ClientError` codes). A narrow substring list silently converted a retryable throttle into a
hard batch failure — visible only as a lost batch. Add the spelling *and* a case for it when a
new one is observed.

## Retry and backoff

Per-provider, exponential, capped at 60s. Azure honours `Retry-After`. NVIDIA uses 4s base for
504. Google uses 10s base for HTTP 429 on preview models.

`parse_retry_after` (`mixins.py`) reads the header off **any** `APIStatusError`, not just
`RateLimitError`: it's defined for 503 (RFC 9110 §10.2.3), every OpenAI-client provider already
retries 5xx, and the openai SDK's own `_calculate_retry_timeout` honours it on every retryable
status — narrowing it to 429 meant a server saying "come back in 30s" during a capacity window
got blind exponential backoff instead. `max_wait` still bounds the value, so a hostile header
can't stall a run; the log line says "backoff", not "rate limit", because a 503 reaches it.

### A too-short client timeout looks exactly like a retryable outage

Retry policy is only as good as the deadline each attempt runs against. If the client gives up
before the model can plausibly answer, every attempt fails identically, `_is_retryable_error` says
"transient" because `ReadTimeout` genuinely is, and the run burns its whole budget converging on the
same wall. The symptom reads as an overloaded provider; the cause is local.

Both non-streaming providers have hit this, for the same underlying reason — the response arrives in
one piece, so nothing is received until generation completes, and a thinking model generates for a
long time before emitting its first output token:

- **Bedrock** — `read_timeout` defaults to 300s; any model whose thinking is on without being asked
  for needs `read_timeout: 1800` in `models.yaml`.
- **NVIDIA** — `polling_timeout` is forwarded to `ChatNVIDIA` as `timeout`, and
  `_NVIDIAClient.timeout` is the `session.post` **read** timeout in addition to the 202-poll budget
  its docstring advertises. The package default is **60s**. Every NIM entry is now an
  always-reasoning model, and at 60s none of them can finish a batch: `glm53-flash-nvidia` on a
  7-line file failed every attempt with `ReadTimeout: read timeout=60`, taking 7m05s to fail, and
  completed in ~3m once the timeout was wired through. **Wiring it then exposed that the configured
  value was itself too low:** `glm53-nvidia` needs 19m0s for the same 7-line file, so the YAML's 900
  sat under the flagship's floor and that run survived only on a retry. It is now `1800`, Bedrock's
  number for Bedrock's reason. Set this from the *slowest* entry the provider serves, not the
  typical one, and treat a run that exceeds the ceiling but still exits 0 as a finding — a
  successful retry hides the misconfiguration while doubling the work.

Two things generalise. **A config knob that reaches the provider's `<Name>Config` but not the client
is invisible** — the same hazard as `ConfigLoader`'s forwarding rule, one layer lower, and a
plausible default is what hides it; `polling_timeout` sat disconnected through an upstream change
that made forwarding safe again, and nothing failed. And **assert against the real client, not a
mock**, when a kwarg's destination is the thing at issue: `timeout` on `ChatNVIDIA` has historically
been both a request-body parameter (HTTP 400) and a client transport option, and a mock cannot tell
those apart.

### Our retry loop must be the only one

**Every client is constructed with its own retries disabled** — `max_retries=CLIENT_RETRIES_DISABLED`
(`= 0`, `mixins.py`) for the six OpenAI-SDK-shaped clients, `BotocoreConfig(retries={"max_attempts": 0})`
for Bedrock. NVIDIA needs nothing: `ChatNVIDIA` runs on a plain `requests.Session` with no retry
adapter and takes no retry knob.

Nested loops **multiply**, they don't add. The bug shipped for six providers at once because the
parameter was simply *absent* and each SDK filled in its own default:

| Client | Its default | Our budget of 5 (6 attempts) became |
|---|---|---|
| `ChatOpenAI` / `AzureChatOpenAI` / `ChatDeepSeek` / `ChatMoonshot` (openai SDK `DEFAULT_MAX_RETRIES = 2`) | 2 | 6 × 3 = **18** requests |
| `ChatGoogleGenerativeAI` (declares it) | 6 | 6 × 7 = **42** requests |

Two consequences beyond the request count, both worse than the waste:

- **`_is_retryable_error` and `_RETRY_MATRIX` are bypassed.** The inner loop retries on the SDK's
  policy, not ours, so a status this project deliberately classifies as fatal (NVIDIA's bare 500,
  a 400) still gets retried, and a status it classifies as retryable never reaches the classifier
  at all. The whole matrix becomes advisory.
- **The inner backoff pre-empts the outer one.** Azure's `Retry-After` handling and Google's 10s
  base for 429 exist because a short wait burns the entire budget inside one rate-limit window —
  which is exactly what the SDK's own sub-second backoff does first.

Don't reintroduce a nonzero value "as a safety net": that net is what made the doubled attempts
invisible. `tests/test_retry_contract.py::test_provider_client_does_not_retry_underneath_our_retry_loop`
asserts on the kwargs each provider hands its client (not on source text — the bug was an *absent*
parameter), and `test_every_provider_is_classified_for_client_level_retries` reflects over
`codereview/providers/*.py` so provider #9 can't skip the classification.

**`max_retries=None` means "the provider decides".** Every `analyze_batch` in the chain
(`CodeAnalyzer` → provider) defaults it to `None` and providers resolve it via
`_resolve_max_retries(override, provider_config, provider_default)`, precedence
`override > provider_config.max_retries > provider default` (5 everywhere except Bedrock's 3,
whose throttling clears in a couple of attempts).

Do **not** give `analyze_batch` a concrete signature default: `CodeAnalyzer` used to default to
`3` and forward it unconditionally, which made every provider default dead code and
`NVIDIAConfig.max_retries` unreachable — NIM's frequent gateway 504s and Azure's quota windows
were being given up on at 3 attempts. Locked by the `max_retries` block in
`tests/test_provider_result_shape_contract.py`, including
`test_every_provider_analyze_batch_defaults_max_retries_to_none`, which reflects over every
provider class.

**Output-parsing failures are retried under `enable_output_fixing`** via a dedicated `except` in
`_execute_with_retry` that names three shapes: `ValidationError` (tool-use schema violation),
`OutputParsingRetryError` (include_raw `parsed` is None), and `OutputParserException` (prompt-
parsing path got malformed JSON — a `ValueError` subclass but NOT a `ValidationError`, so it
must be named explicitly or it falls into the generic non-retryable `except`). Reasoning models
on the prompt-parsing path (e.g. GPT-5.6 Sol on Bedrock) intermittently emit invalid JSON on
think-heavy batches; the retry is what makes those runs complete.

## Token accounting counts what was billed, not what parsed

`base.py`'s `_extract_token_usage` runs on the raw `AIMessage` even when `parsed` is `None`
(a tool-use schema violation, or a reasoning-only response), because the provider charged for
those tokens regardless. Omitting them made `--dry-run` estimates look accurate while real runs
under-reported cost by exactly the retried batches — the expensive ones.

**The prompt-parsing path needs its own hook for the same reason**: an `OutputParserException`
raises from the *parser*, past the `AIMessage`, so there is no usage metadata left to read —
`_track_usage_from_parse_failure` estimates from the prompt text plus the rejected output the
parser attaches as `llm_output`, called from the retry `except` in `_execute_with_retry`.
Estimating isn't a shortcut there: a `CodeReviewReport` carries no metadata either, so the
*success* branch of that path is already estimated, and every `supports_tool_use: false`
reasoning model (Opus 5, GPT-5.6 Sol, GLM-5.3, K3, …) is exactly the kind that
burns several billed attempts on a think-heavy batch. Swallow accounting failures to
`logging.debug` — this runs on the way to a retry or a raise and must never mask the parse
error.

### `AIMessage` carries usage in two independent places

Only `usage_metadata` is filled on every path. `usage_metadata` is LangChain's normalized
`input_tokens`/`output_tokens`, set by *both* the Chat Completions and the Responses API
converters; `response_metadata["token_usage"]` is the vendor's raw
`prompt_tokens`/`completion_tokens`, and **only** the Chat Completions converter copies it
through (it reaches `response_metadata` at all only because langchain-core merges `llm_output`
into the message — so it's absent from anything that bypasses `_create_chat_result`, which the
Responses API path does).

`extract_openai_token_usage` read the raw dict only, so it returned `(0, 0)` for every
`use_responses_api: true` entry, `.get(..., 0)` made that indistinguishable from "no usage
reported", and `base.py` silently substituted its byte-heuristic *estimate* — which cannot see
reasoning tokens at all. Azure `gpt-5.4` under-reported ~13x on a think-heavy batch
(40,000/9,000 billed, 6,211/145 recorded; $0.2350 printed as $0.0177), i.e. the failure was
largest on the priciest models. **Read `usage_metadata` first everywhere; keep the raw dict as
the fallback.**

**The test rule is the same one the retry classifiers taught: build the response the way the
real client builds it.** `tests/test_token_usage_contract.py` drives each provider's extractor
with an `AIMessage` the *vendor's own client* produced from a recorded wire payload
(`BaseChatOpenAI` for both OpenAI paths, `_parse_response` for Bedrock Converse — note it
**mutates** its argument, so deep-copy the payload; a real `requests.Response` for NIM;
`_response_to_result` for Google). The pre-existing hand-built-`AIMessage` tests invented the
one field the broken extractor read, so they passed for exactly as long as it was wrong. Two
reflective guards keep it from lapsing: every `ModelProvider` subclass must appear in
`_USAGE_MATRIX`, and every provider module mentioning `use_responses_api` must be covered on
that path.

### Tiered pricing is a property of one request, not of a run

GPT-6 Astra is the first entry whose rate depends on how big a single call is: In-Region it
bills $11/$55 per million at 272,000 input tokens **or fewer** and $22/$82.50 above that.
Three optional `PricingConfig` fields carry it — `long_context_threshold_tokens`,
`long_input_per_million`, `long_output_per_million` — and a `model_validator` rejects a
*partial* tier, because two of the three would fall back to the flat pair with no error and no
warning, which is exactly the "YAML key that looks like configuration" failure that shipped
sixteen times on `pricing`/`inference_params`.

**The threshold must never be applied to an accumulated total.** `estimate_cost` used to
multiply `_total_input_tokens` by one rate, and a run of five 100K batches has a 500K total
while every one of its requests billed at the cheap tier — pricing the total would invent a
long-context charge nothing incurred, and *doubling* a cost figure is the same class of wrong
as halving it. So the tier is selected in `TokenTrackingMixin._track_tokens`, the last place a
single request's input size is still visible: it calls `PricingConfig.rates_for_request`,
accrues `_accrued_input_cost`/`_accrued_output_cost` under the existing token lock, and counts
`_long_context_requests`. `estimate_cost` then only *reports* the accrual — it must not
recompute, or the per-request sizes are gone again. For a flat-priced entry the accumulator is
arithmetically identical to the old single multiplication (`sum(t_i) * r == sum(t_i * r)`), so
the untiered entries are unaffected to the cent.

`get_pricing()` adds the three tier keys **only** when the entry has a tier, so consumers must
read them with `.get()` — a flat model reports no tier rather than a null one. `--dry-run`
mirrors the same rule in `cli.py`'s `_estimate_tiered_cost`, which prices each `FileBatch` as
its own request (files plus the per-batch overhead, since the re-sent system prompt and README
are billed input) and names how many batches crossed the break, because a smaller
`--batch-size` can drop the run back to the cheap tier. Guards in
`tests/test_tiered_pricing.py`.

**Reporting the accrual is half the job, and it was the half that shipped broken.** Making
`--dry-run` tier-aware left `estimate_cost()` with **no caller at all**: the completed run's
summary (`cli.py`) and the Markdown export (`renderer.py`) each independently recomputed
`tokens / 1_000_000 * rate` from the run *totals*, and the only rate in `get_pricing()` for them
to use was the short-context one. Unclamping Astra's `context_window` made batches over 272K
reachable, so both reported roughly **half** the billed cost — the very failure the clamp had
existed to prevent. `ReviewMetrics` therefore carries `input_cost`, `output_cost` and
`long_context_requests`; `run_review` fills them from `analyzer.estimate_cost()` and both
consumers *report* them, plus the count of requests billed at the long tier (a doubled rate a
user cannot see is indistinguishable from a bug). The export drops its `($11.00/M tokens)`
annotation whenever that count is nonzero, because the run spans two rates and naming one would
be a lie. Its flat arithmetic survives only as a fallback for a metrics dict with no accrual —
a hand-built or legacy report, where no provider ran and the product is exact anyway.

The rule this leaves: **a cost consumer reads the accrual, it does not price tokens.**
`test_no_new_consumer_recomputes_cost_from_a_rate` scans `codereview/` for per-million
arithmetic and fails on any file outside `_ALLOWED_COST_ARITHMETIC`, whose three entries each
carry a reason — `mixins.py` *produces* the accrual, `cli.py`'s `_estimate_tiered_cost` prices a
dry run that has no requests yet, and `renderer.py` holds the documented fallback above. A
companion test rejects an allowlist entry that no longer does the arithmetic.

## Streaming: `streaming=bool(callbacks)` was wrong twice, and `--stream` was wrong a third time

Three coupled defects, all in `tests/test_streaming_contract.py`:

1. **`ProgressCallbackHandler` (the `--verbose` handler) does not override
   `on_llm_new_token`** — it cannot observe a streamed token. So `--verbose` alone moved all
   five OpenAI-compatible providers onto the streaming wire path to feed a handler that ignores
   it. Use `wants_token_streaming(callbacks)` (`mixins.py`), which compares each handler's
   `on_llm_new_token` against `BaseCallbackHandler`'s — an **override check, not class
   identity**, both so third-party handlers work and so `mixins.py` (imported by every
   provider) needn't import `codereview.callbacks` and thus Rich.
2. **Streaming without `stream_usage` silently loses the billed counts.** `_stream` is the only
   place langchain-openai turns `stream_usage` into `stream_options={"include_usage": True}`,
   and it auto-enables that only when *no* `base_url` is configured — all five of these
   providers configure one. Without it a real server sends no usage chunk →
   `usage_metadata is None` → `extract_openai_token_usage` returns `(0,0)` → the byte-heuristic
   estimate, i.e. the under-reporting bug above, reintroduced by the flag meant to show more
   detail. `openai_stream_params(callbacks)` returns them as one unit; the flag is inert off the
   streaming path, so it's set only with it.
3. **`--stream` forced `max_workers=1` even where no token ever arrives.** Bedrock passes
   `disable_streaming=True` (and its `read_timeout: 1800` overrides depend on the
   non-streaming Converse path), `ChatNVIDIA` has no `streaming` field at all, and Google's is
   off because `method="json_schema"` structured output through the streaming wire path is
   unproven live — so on three providers, including the default `opus5`, the flag bought a
   3-5x slowdown for output that cannot appear.

`ModelProvider.supports_token_streaming()` is a **classmethod** (default `True`; those three
override to `False`) and `ProviderFactory.supports_token_streaming(model_name)` resolves it
through `_PROVIDER_REGISTRY` **without constructing anything**. It must stay answerable from
the class: worker count and which handler to attach are *one* decision — a
`StreamingCallbackHandler` under `max_workers > 1` is precisely the concurrent-`Live` overlap
`docs/architecture.md` documents — and both feed the provider constructor, so neither can wait
for an instance. `run_review` downgrades the flag with an explicit notice (silently ignoring it
is worse than the slowdown), keeps the parallelism, and attaches the concurrency-safe spinner
handler instead. An unresolvable model returns `True` so this never becomes the thing that fails
a run; `create_provider` reports the real error.

## Sampling params

**Reasoning models** (Claude Opus 5, Claude Sonnet 5, Claude Fable 5, GPT-5.4 / 5.4 Pro on Azure,
GPT-5.6 Sol and GPT-6 Astra on Bedrock, DeepSeek-V4-Pro) don't accept `temperature`/`top_p`. Bedrock and Azure
providers both pass `allow_none=True` to `_resolve_temperature`; omit `default_temperature` from
`inference_params` for new reasoning models.

Being a reasoning model does not by itself mean the sampling params are refused — xAI's Grok 4.3
was reasoning-first *and* accepted `temperature`/`top_p` (card defaults 0.7/0.95), which is why its
entry rode Chat Completions rather than the Responses API. Its entry was removed 2026-08-29, so
every remaining reasoning entry refuses them; read the model card rather than generalizing from
that.

**Gemini sampling params are deprecated from 3.6 Flash onward** — Google's API ignores
`temperature`/`top_p`/`top_k` on Gemini 3.6 Flash and documents an HTTP 400 for future model
generations. Omit all three (`default_temperature`/`default_top_p`/`default_top_k`) from
`inference_params` for every new Gemini entry; the Google provider already passes
`allow_none=True` to `_resolve_temperature` and drops `top_p`/`top_k` when unset, so no code
change is needed. The older Gemini 3.1 Pro entry keeps theirs — that generation still honors
them. Locked by `test_gemini38_flash_matches_the_published_model_card` for the current Flash entry
(the pinned-entry test has followed the roster twice: the 3.6 entry it originally covered went on
2026-08-29, the 3.7 one on 2026-09-19) and by
`test_every_modern_gemini_entry_omits_sampling_params`, which parses the version out of every
`google_genai` entry's `id` and fails when a *new* one at ≥3.6 reintroduces a sampler — the pinned
single-entry test can't catch that.

## Per-provider quirks

### OpenAI-on-Bedrock is NOT the `bedrock` provider

GPT-5.6 Sol and GPT-6 Astra on Bedrock go through Bedrock's *OpenAI-compatible* endpoint, which
authenticates with an Amazon Bedrock **API key (bearer token)** via `ChatOpenAI` + `base_url` —
not the SigV4 `ChatBedrockConverse` path. It lives in the separate `bedrock_openai` provider.
Underlying transport is the `openai` SDK (already pulled by `langchain-openai`; no new dep).

The `bedrock_openai` model entries' `full_id` is a **literal**, not
`${BEDROCK_OPENAI_MODEL_ID}` — an unset env var expands to `""` and fails `full_id`'s
`min_length=1`, breaking `--list-models`; paste the wire id from the console instead.

The GPT entries are reasoning models (Responses API via `use_responses_api: true`, no
temperature/top_p) and use `supports_tool_use: false` — **verified against the live endpoint on
GPT-5.5**: GPT-5.x here engages adaptive server-side thinking per request, and on think-heavy
batches returns a reasoning-only response (`tool_calls=[]`, no `parsed` field → "Structured Output
response does not have a 'parsed' field"), which breaks the forced `tool_choice` that
`.with_structured_output()` sets. Intermittent (only the batches where it thinks). Same failure
mode as Opus 4.8 on Bedrock, so they route through prompt-based JSON parsing. **The GPT-5.5 entry
was removed 2026-08-29 but that observation is the whole basis for Sol's flag**, so it is recorded
here and in the YAML rather than left in git history — `openai.gpt-5.5` is still live on
`bedrock-mantle`, and re-adding the entry should restore `supports_tool_use: false` with it.

The GPT-5.4-on-Bedrock entry was removed 2026-07-25 (two newer generations on the same
endpoint; its `gpt5.4-bedrock` alias was deleted rather than pointed at GPT-5.5 — see the
version-explicit rule in `docs/model-registry.md`). Note GPT-5.4 on *Azure* is a separate entry
that stays and keeps `supports_tool_use: true` — that deployment doesn't exhibit this; the
Bedrock OpenAI-compatible endpoint does.

**GPT-5.6 Sol** (`openai.gpt-5.6-sol`, id `gpt5.6-sol-bedrock`, aliases
`gpt5.6`/`gpt-5.6`/`gpt5.6-bedrock` plus the inherited `gpt-bedrock`; flagship of the
Sol/Terra/Luna family) is Responses-API-only, rejects
`temperature`/`top_p`, and its `full_id` is a real published wire id rather than a console
literal. It's OpenAI's best coding model, so it's the code-review pick of the family; In-Region
only us-east-1 / us-east-2. It is also, at $5/$30 per million, **twice the price of the GPT-5.5
entry it replaced** ($2.50/$15) and narrower (272K vs 400K) — which is why `gpt-bedrock` sits in
`deprecated_aliases`, resolvable but unadvertised, rather than in `aliases`.

**GPT-6 Astra** (`openai.gpt-6-astra`, id `gpt6-astra-bedrock`, aliases
`gpt6`/`gpt-6`/`gpt6-bedrock`) joined 2026-09-13, ending Sol's spell as the provider's only entry.
GA on Bedrock 2026-09-08, OpenAI's most capable model, text + image input, 128K max output,
knowledge cutoff 2026-04-30. Two properties are unlike anything else in the registry:

- **Its Region is mutually exclusive with Sol's.** `base_url` is *provider*-level
  (`${OPENAI_BASE_URL}`), and `bedrock-mantle` serves Astra from **us-west-2 only** while Sol is
  In-Region us-east-1 / us-east-2 only. One base URL cannot reach both; the wrong one 404s the
  model id with no fallback, on *every* batch of the run — the observed shape on 2026-09-13 was
  four batches and four identical `The model 'openai.gpt-6-astra' does not exist` 404s against a
  us-east-2 endpoint. This was an operator switch (re-export `OPENAI_BASE_URL` to change models)
  until both entries gained **`region:`**, which `BedrockOpenAIProvider._resolve_base_url` reads
  to rewrite the Region label of the configured URL per model, so one export serves both. Two
  deliberate properties of that helper: it **derives from the configured URL** rather than a
  hardcoded `bedrock-mantle` host template, so `OPENAI_BASE_URL` stays authoritative for scheme,
  host and path and a custom gateway keeps working; and a host with **no Region label to rewrite
  passes through unchanged** (debug-logged) rather than raising — failing closed there would
  break a working self-hosted endpoint over a cosmetic mismatch. The HTTPS gate runs on the
  *resolved* URL, so a Region override can't smuggle the bearer key onto `http://`. Don't widen
  Astra's Region from memory — the model card's `bedrock-mantle` availability table had exactly
  one row on 2026-09-13. Note `--validate` cannot catch a wrong Region: it has no model in scope
  and checks only the provider-level URL, so the Region is first exercised on invoke.
- **`supports_tool_use: false` is live-verified here, not assumed** (2026-09-13). A/B on
  `codereview/providers/` — 2 batches, ~98K input tokens, on the us-west-2 mantle base. With
  `true`, batch 1 of 2 died on `ResponseError(code='server_error', message='The server had an
  error while processing your request.')`, surfaced through `base.py`'s
  `raise ValueError(response.error)` after retries consumed the budget: 3m06s wall clock for a
  half-finished review. With `false`, both batches completed in 37.4s and found 4 issues. Same
  target, key and Region. **The symptom differs from GPT-5.5's** — 5.5 returned a reasoning-only
  response (`tool_calls=[]`, no `parsed`), Astra 500s — but the cause and the fix are the same.
  Two traps: the error *reads* as transient, so don't respond to a recurrence by widening the
  retry classifier; and a trivial batch passes forced `tool_choice` cleanly (a 12-line file did),
  so only a think-heavy target reproduces it.
- **It is the first entry here with *tiered* pricing**, and its `context_window` was clamped to
  the price break until the code could handle that. In-Region it bills $11/$55 per million up to
  272K input tokens and $22/$82.50 above; with one flat input/output pair per entry, a batch
  crossing 272K would have cost 2x what `--dry-run` and the cost line reported — the same
  under-reporting class as the Azure `gpt-5.4` `usage_metadata` bug, and the worst kind of wrong
  because the next reader trusts a pricing number. Clamping the window to 272000 meant the batcher
  could not pack past it. The window is now `1000000` and both tiers are configured; the
  mechanism, and the per-request-not-per-run rule that makes it correct, are under
  [Tiered pricing is a property of one request](#tiered-pricing-is-a-property-of-one-request-not-of-a-run).
  What the clamp cost while it stood was review *quality*, not money: ~4x more batches, and each
  batch only ever sees its own files, so cross-file findings were lost.

**The `bedrock_openai` provider is not OpenAI-only, and the code still proves it.** xAI's
**Grok 4.3** rode the same `bedrock-mantle` OpenAI-compatible endpoint (model id `xai.grok-4.3`,
base_url `https://bedrock-mantle.{region}.api.aws/openai/v1`) through this provider until its
entry was cut on 2026-08-29 — the endpoint is live, only the registry row is gone. Nothing in the
provider is OpenAI-specific, so a non-OpenAI `bedrock-mantle` model needs a YAML entry and no
code. Two things that entry taught, worth keeping for the next one: a `bedrock-mantle` model may
**accept `temperature`/`top_p`** and therefore omit `use_responses_api` and ride Chat Completions
(Grok's card defaulted 0.7/0.95); and In-Region support is per-model, not per-endpoint (Grok:
us-west-2 / us-east-1 / us-east-2; Sol: us-east-1 / us-east-2 only), so give the entry a
**`region:`** naming a Region that serves that specific model rather than relying on whatever
`OPENAI_BASE_URL` happens to point at.

### Moonshot has two platforms

`platform.moonshot.cn` (Chinese, default in our YAML) and `platform.moonshot.ai`
(international). Keys are NOT interchangeable. `KIMI_API_KEY` typically maps to `.cn`; users
with `.ai` keys must override `base_url` in the moonshot section.

### DeepSeek-V4-Pro on Azure / SGLang null-model bug

The Foundry endpoint validates `body.model` strictly. langchain-openai's `AzureChatOpenAI`
defaults `model_name=None` and serializes `"model": null`, which real Azure-OpenAI ignores but
SGLang rejects with HTTP 400. The Azure provider explicitly sets `model=deployment_name` to
satisfy both backends.
