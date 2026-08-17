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
on the prompt-parsing path (e.g. GPT-5.5 on Bedrock) intermittently emit invalid JSON on
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
reasoning model (Opus 5, GPT-5.5/5.6 Sol, Grok 4.3, GLM-5.2, K2.6, …) is exactly the kind that
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

**Reasoning models** (Claude Opus 5, Claude Opus 4.8, Claude Sonnet 5, Claude Fable 5,
GPT-5.4 / 5.4 Pro on Azure, GPT-5.5 / GPT-5.6 Sol on Bedrock, DeepSeek-V4-Pro) don't accept
`temperature`/`top_p`. Bedrock and Azure providers both pass `allow_none=True` to
`_resolve_temperature`; omit `default_temperature` from `inference_params` for new reasoning
models.

**Gemini sampling params are deprecated from 3.6 Flash onward** — Google's API ignores
`temperature`/`top_p`/`top_k` on Gemini 3.6 Flash and documents an HTTP 400 for future model
generations. Omit all three (`default_temperature`/`default_top_p`/`default_top_k`) from
`inference_params` for every new Gemini entry; the Google provider already passes
`allow_none=True` to `_resolve_temperature` and drops `top_p`/`top_k` when unset, so no code
change is needed. The older Gemini 3.1 Pro entry keeps theirs — that generation still honors
them. Locked by `test_gemini36_flash_omits_sampling_params`.

## Per-provider quirks

### OpenAI-on-Bedrock is NOT the `bedrock` provider

GPT-5.5 / GPT-5.6 Sol on Bedrock go through Bedrock's *OpenAI-compatible* endpoint, which
authenticates with an Amazon Bedrock **API key (bearer token)** via `ChatOpenAI` + `base_url` —
not the SigV4 `ChatBedrockConverse` path. It lives in the separate `bedrock_openai` provider.
Underlying transport is the `openai` SDK (already pulled by `langchain-openai`; no new dep).

The `bedrock_openai` model entries' `full_id` is a **literal**, not
`${BEDROCK_OPENAI_MODEL_ID}` — an unset env var expands to `""` and fails `full_id`'s
`min_length=1`, breaking `--list-models`; paste the wire id from the console instead.

The GPT entries are reasoning models (Responses API via `use_responses_api: true`, no
temperature/top_p) and use `supports_tool_use: false` — **verified against the live endpoint**:
GPT-5.x here engages adaptive server-side thinking per request, and on think-heavy batches
returns a reasoning-only response (`tool_calls=[]`, no `parsed` field → "Structured Output
response does not have a 'parsed' field"), which breaks the forced `tool_choice` that
`.with_structured_output()` sets. Intermittent (only the batches where it thinks). Same failure
mode as Opus 4.8 on Bedrock, so they route through prompt-based JSON parsing.

The GPT-5.4-on-Bedrock entry was removed 2026-07-25 (two newer generations on the same
endpoint; its `gpt5.4-bedrock` alias was deleted rather than pointed at GPT-5.5 — see the
version-explicit rule in `docs/model-registry.md`). Note GPT-5.4 on *Azure* is a separate entry
that stays and keeps `supports_tool_use: true` — that deployment doesn't exhibit this; the
Bedrock OpenAI-compatible endpoint does.

**The `bedrock_openai` provider is not OpenAI-only.** xAI's **Grok 4.3** rides the same
`bedrock-mantle` OpenAI-compatible endpoint (model id `xai.grok-4.3`; base_url
`https://bedrock-mantle.{region}.api.aws/openai/v1`) and lives in this provider too. Grok 4.3
differs from the GPT-5.x entries in two ways: it is **not** Responses-API-only — it **accepts
`temperature`/`top_p`** (card defaults 0.7/0.95), so its entry omits `use_responses_api` and
passes a low temperature over Chat Completions; and its `full_id` (`xai.grok-4.3`) is a real
published wire id, not a console-specific literal. It still uses `supports_tool_use: false`
because its always-on reasoning is the highest-risk forced-`tool_choice`-while-thinking profile
(assume-prompt-parsing rule). Grok 4.3 is **In-Region only** (us-west-2 / us-east-1 /
us-east-2; no Geo/Global) — pin `OPENAI_BASE_URL` to a supported Region.

**GPT-5.6 Sol** (`openai.gpt-5.6-sol`, id `gpt5.6-sol-bedrock`, aliases
`gpt5.6`/`gpt-5.6`/`gpt5.6-bedrock`; flagship of the Sol/Terra/Luna family) is also here: like
the GPT-5.x entries it is Responses-API-only and rejects `temperature`/`top_p`, but like Grok
its `full_id` is a real published wire id (not a console literal). It's OpenAI's best coding
model, so it's the code-review pick of the family; In-Region only us-east-1 / us-east-2
(narrower than Grok — no us-west-2).

### Moonshot has two platforms

`platform.moonshot.cn` (Chinese, default in our YAML) and `platform.moonshot.ai`
(international). Keys are NOT interchangeable. `KIMI_API_KEY` typically maps to `.cn`; users
with `.ai` keys must override `base_url` in the moonshot section.

### DeepSeek-V4-Pro on Azure / SGLang null-model bug

The Foundry endpoint validates `body.model` strictly. langchain-openai's `AzureChatOpenAI`
defaults `model_name=None` and serializes `"model": null`, which real Azure-OpenAI ignores but
SGLang rejects with HTTP 400. The Azure provider explicitly sets `model=deployment_name` to
satisfy both backends.
