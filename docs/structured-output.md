# Structured output: which path each model takes, and why

Background for the one-line rule in `CLAUDE.md` ("Structured output" under Key patterns).
Read this before adding a model, changing a `supports_tool_use` value, or replacing the
prompt-parsing path.

## The two paths

Default is `.with_structured_output(CodeReviewReport, include_raw=True)`. `include_raw` is
required to read real token counts from the raw `AIMessage`.

Models with `supports_tool_use: false` in `models.yaml` use `PydanticOutputParser` instead.
**The routing lives once in `base.py`**: providers call `self._apply_structured_output(base_model)`
from `_create_model` (extra kwargs forwarded, e.g. Google's `method="json_schema"`), and the
base class owns `_use_prompt_parsing`, `_output_parser` (cached property), the default
`_create_chain`, and the format-instructions injection — so every provider, including Google,
honors the YAML flag automatically.

## The rule for new models

**When adding a reasoning/thinking model, assume the prompt-parsing path until a live run
proves tool-use works.** The failure is often intermittent — only on batches where the model
thinks. Set `supports_tool_use: false` in `models.yaml` to opt into prompt parsing; the
provider appends a `PydanticOutputParser` and injects format instructions.

## Do not replace this with `method="prompt_prefill"`

`ChatBedrockConverse` gained `method="prompt_prefill"` in langchain-aws ≥1.6.3. Investigated
and rejected 2026-07-26. It is mechanically our path plus a forced ` ```json ` assistant
prefill and a stop sequence, and both additions are hazards here:

1. The stop sequence is a bare triple-backtick unioned into the request unconditionally
   (`_PROMPT_PREFILL_STOP`; the prefill method has no way to opt out), and Bedrock stop
   sequences match generated text *inside* JSON string values — so any finding whose
   `suggested_code`, `description` or `rationale` contains a code fence truncates generation
   mid-JSON. That is deterministic on the content, not intermittent on the thinking, and this
   repo already knows models put fences in prose fields (that's why `balance_code_fences`
   exists, and `tests/test_markdown_export.py` has a case for a fenced `suggested_code`).
   Reviewing Markdown or fenced docstrings would trigger it.
2. Anthropic documents *"You can't pre-fill the assistant response while thinking is on"* —
   and every Bedrock entry we'd use it for has thinking on by default.
3. It is `ChatBedrockConverse`-only, while 11 of our 18 prompt-path entries are NVIDIA /
   Moonshot / Z.AI / `bedrock_openai`, so adopting it means two structured-output paths to
   maintain.

Upstream's own docstring names its target as non-thinking models ("notably Amazon Nova"),
which is consistent with all of the above.

## The path matrix

| Model (provider) | Thinking | `supports_tool_use` | Path | Why prompt-parsing (if so) |
|---|---|---|---|---|
| Claude Fable 5 (Bedrock) | adaptive (always on, can't disable) | `false` | prompt | Same forced-`tool_choice`-while-thinking conflict as Opus 4.8 below, but **constant** rather than intermittent — thinking can't be disabled. Also rejects `temperature`/`top_p`/`top_k`; requires one-time `provider_data_share` data-retention opt-in |
| **Claude Opus 5 (Bedrock)** | **on by default** (effort-controlled) | `false` | prompt | **Documented, not assumed**: the Bedrock model card lists *Structured outputs: Not Supported* on both `bedrock-runtime` and `bedrock-mantle`. Thinking-on-by-default also reproduces the Opus 4.8 forced-`tool_choice` conflict. Current CLI default; also needs `read_timeout: 1800` (Fable 5's non-streaming-Converse problem) |
| Claude Opus 4.8 (Bedrock) | adaptive (server-side) | `false` | prompt | Forced `tool_choice` while thinking → tool call returned as **literal text** → `list_type` error (intermittent). The Opus 4.7 / 4.6 entries were removed 2026-07-25; their version-explicit aliases were deleted, so use `opus`/`opus5` |
| Claude Sonnet 5 (Bedrock) | adaptive (on by default, server-side) | `false` | prompt | Same forced-`tool_choice`-while-thinking conflict as Opus 4.8 — first Sonnet tier with adaptive thinking on by default. Also rejects `temperature`/`top_p`/`top_k`. No `provider_data_share` opt-in (unlike Fable 5); geo-US routes from the us-west-2 default |
| GPT-5.5 (**Bedrock** OpenAI-compat) | adaptive (server-side) | `false` | prompt | Think-heavy batches return reasoning-only (`tool_calls=[]`, no `parsed`) → "no 'parsed' field" (intermittent). The GPT-5.4-on-Bedrock entry was removed 2026-07-25; its version-explicit `gpt5.4-bedrock` alias was deleted, not migrated |
| GPT-5.6 Sol (**Bedrock** `bedrock-mantle` OpenAI-compat) | adaptive (server-side) | `false` | prompt | Same `bedrock-mantle` endpoint and reasoning-only failure mode as GPT-5.5-on-Bedrock. **Responses API only** (Chat Completions not supported → `use_responses_api: true` required), no `temperature`/`top_p`. Sol tier = OpenAI's best coding model; In-Region us-east-1/us-east-2 only |
| Grok 4.3 (**Bedrock** `bedrock-mantle` OpenAI-compat) | reasoning-first (always-on, effort configurable) | `false` | prompt | Same `bedrock-mantle` endpoint as GPT-5.5-on-Bedrock; always-on reasoning is the highest-risk forced-`tool_choice`-while-thinking profile → assume-prompt-parsing until proven. Unlike GPT-5.x here it **accepts** `temperature`/`top_p`, so it uses Chat Completions (no `use_responses_api`) |
| GPT-5.4 / 5.4 Pro (**Azure**) | reasoning | `true` | tool-use | Azure deployment tolerates forced `tool_choice`; Bedrock's endpoint does not |
| Kimi K2.6 (Moonshot) | enabled (server-side) | `false` | prompt | Moonshot rejects `tool_choice='specified'` (HTTP 400) while thinking |
| Kimi K2.6 (NVIDIA) | on by default | `false` | prompt | Same model/behavior as Kimi K2.6 on Moonshot; kept consistent (thinking on → forced `tool_choice` rejected) |
| Kimi K2.5 (Bedrock) | server-side think toggle | `false` | prompt | Bedrock Converse leaks Moonshot tool-call markers (`<\|tool_call_begin\|>…`) into text instead of parsing as `tool_use` — literal-text failure like Opus |
| DeepSeek V4 family (**DeepSeek direct**) | on by default (both V4-Pro and V4-Flash) | `true` | tool-use | Thinking is on by default and rejects a forced `tool_choice` (HTTP 400), but **the provider explicitly sends `thinking: disabled`** so tool calling works — tool-use is a property of us disabling thinking, not of the model. **`inference_params.thinking: enabled` flips this entry to the prompt path at runtime** (see `deepseek._create_model`) |
| MiniMax M2.5 (Bedrock) | — | `false` | prompt | No usable tool-based structured output |
| MiniMax M3 (NVIDIA) | enabled (interleaved) | `false` | prompt | New reasoning/thinking model — assume prompt-parsing until a live run proves tool-use (forced `tool_choice` while thinking is unproven on this endpoint). Live-verified working on the prompt path. Owns the whole MiniMax-on-NVIDIA alias lineage after M2.7 was removed 2026-07-25 |
| Qwen3.5 397B (NVIDIA) | on by default | `false` | prompt | With thinking on, tool calls emitted as XML inside the `<think>` block instead of structured `tool_use` — literal-text failure. The only Qwen on NIM (Qwen3 Coder 480B's endpoint is gone; `qwen-nvidia`/`qwen3-nvidia`/`qwen-coder-nvidia` resolve here) |
| GLM 5 (Bedrock) | on by default (reasoning_effort=max) | `false` | prompt | Thinking model → forced `tool_choice` auto-downgraded/returned as text; assume-prompt-parsing until proven (positive Converse report was for GLM-4.7, not GLM-5) |
| GLM-5.2 (NVIDIA) | on by default (effort levels) | `false` | prompt | 753B MoE, 1M context. NIM re-host emits malformed/truncated tool-call JSON (as the deprecated GLM-5.1-on-NVIDIA endpoint did), and it's a thinking model → assume-prompt-parsing rule. **Absorbed the version-neutral `glm5`/`glm-5` aliases** of the retired GLM-5.1 entry (NVIDIA deprecated the free z-ai/glm-5.1 endpoint ~2026-07); the version-explicit `glm51`/`glm5.1` names were deleted. Unverified live; flip to `true` only if a live run proves tool-use |
| Step 3.7 Flash (NVIDIA) | on by default (reasoning_effort=medium) | `false` | prompt | Always-thinking backbone; forced `tool_choice` while thinking unproven; assume-prompt-parsing rule. Owns `step-flash` after Step 3.5 Flash was removed 2026-07-25 (the version-explicit `step35`/`step-3.5-flash` names were deleted) |
| Mistral Small 4 119B (NVIDIA) | off by default | `false` | prompt | NVIDIA NIM endpoint observed not to deliver usable tool-based structured output (per config note) — non-thinking, but empirically prompt-path |
| GLM-5.2 (Z.AI) | enabled (server-side) | `false` | prompt | Z.AI's endpoint ignores `json_schema` response_format and returns markdown-fenced JSON (`PydanticOutputParser` strips the fences) **and** it's a thinking model → assume-prompt-parsing rule. 1M context, only Z.AI entry (GLM-5.1 removed 2026-07-25; its aliases resolve here). Unverified live; flip to `true` only if a live run proves tool-use |
| Gemini 3.6 Flash (Google) | on by default (level medium) | `true` (default) | tool-use | **Exception to the assume-prompt-parsing rule, earned by a live run**: a thinking model that still tool-calls fine. Google documents both structured outputs and function calling, and a real review run returned a valid `CodeReviewReport` on the tool-use path. Keep new Gemini entries on prompt-parsing until you likewise prove it. Owns `gemini-flash` plus the version-neutral `gemini-3-flash`/`gemini3-flash`/`g3flash` of the removed 3 Flash Preview (2026-07-25) |
| Everything else (Claude Sonnet, GPT-OSS, Qwen, other Gemini, …) | — | `true` (default) | tool-use | Standard `.with_structured_output()` |

## The two failure shapes

Two distinct shapes drive the `false` cases:

**"Can't tool-call at all / mangles the tool call"** — MiniMax family, Kimi-K2.5-on-Bedrock
marker leakage, Qwen3.5/GLM-5.2-on-NVIDIA malformed output, GLM-5.2-on-Z.AI fenced JSON,
Mistral Small.

**"Can tool-call but not *while thinking*"** — Opus 5, Opus 4.8, Sonnet 5, Fable 5, GLM 5,
GPT-5.5/5.6-Sol-on-Bedrock, Grok 4.3-on-Bedrock, K2.6, Step 3.7 Flash. These are
intermittent, except the always-on-thinking models (Fable 5, Grok 4.3), which are constant.

Opus 5 belongs to **both** shapes: its model card denies structured-output support outright
*and* thinking is on by default.

Many NVIDIA-NIM and Bedrock re-host `false` values are set under the **assume-prompt-parsing
rule** (thinking model, forced `tool_choice` unproven live), not a confirmed live failure —
flip to `true` only if a live run proves tool-use.

## The forced-`tool_choice`-while-thinking failure is an observation, not a documented API restriction

**Don't cite Anthropic's docs for it.** `CLAUDE.md`, `models.yaml` and two tests all used to
assert "Anthropic allows only `tool_choice: auto/none` while thinking". That rule is real but
**scoped to *manual* `thinking: {type: "enabled"}`**; Anthropic's thinking page states the
opposite for the models we actually ship: *"Adaptive thinking, including on models where
thinking is on by default, supports forced tool use."*

langchain-aws encodes the same scoping — `thinking_forced_tool_use_unsupported()` (`utils.py`)
explicitly returns `False` for `claude-opus-4-8` and never listed Opus 5 / Sonnet 5 / Fable 5,
and it only engages when a `thinking` key is actually present in the request. So upstream
forces a `tool_choice` on our Bedrock Claude entries **by design**, and there is no upstream
bug to file (I nearly filed one).

What survives is the empirical failure, reproduced live on Opus 4.8 in `de5e2fc`: markup as
text, `list_type` on `issues`, only on think-heavy batches. Opus 5 has an independent reason
anyway (its model card denies structured-output support). Keep `supports_tool_use: false`;
describe it as observed behavior, and don't attach a vendor-rule explanation that the vendor
contradicts.

## Per-model detail

**MiniMax M2.5 on Bedrock, MiniMax M3 on NVIDIA, Kimi K2.6 on Moonshot, Claude Opus 5, Opus
4.8 and Sonnet 5 on Bedrock, and GLM-5.2 on Z.AI** lack usable tool-based structured output.

- **Opus 5** is the one case with vendor confirmation rather than inference: its Bedrock model
  card lists *Structured outputs: Not Supported* for both `bedrock-runtime` and
  `bedrock-mantle`, and thinking is on by default (a breaking change from Opus 4.8, where it
  was off unless requested) so it also hits the forced-`tool_choice`-while-thinking conflict.
  Opus 5 additionally rejects sampling params and needs `read_timeout: 1800`.
- **MiniMax M3** is a new reasoning/thinking model: per the assume-prompt-parsing rule it ships
  `false` (live-verified working on the prompt path against the NVIDIA NIM endpoint).
- **Sonnet 5** is the first Sonnet-tier model with adaptive thinking on by default, so it
  inherits the exact Opus 4.8 conflict (unverified live; ships `false` under the rule).
- **GLM-5.2 (Z.AI)**: Z.AI's OpenAI-compat endpoint ignores OpenAI's `json_schema`
  response_format that `.with_structured_output()` sets and returns markdown-fenced JSON
  (` ```json … ``` `), which the json_schema parser rejects with "Invalid JSON: expected value
  at line 1 column 1"; `PydanticOutputParser` strips the fences. It is the current
  `zai_default` (1M context, the only Z.AI entry since GLM-5.1 was removed 2026-07-25) and
  additionally a thinking model, so it stays on the prompt path under the rule (unverified
  live).
- **K2.6** — Moonshot's server rejects `tool_choice='specified'` (HTTP 400) when thinking is
  enabled.
- **Opus 4.8 and Sonnet 5** support only `thinking.type: "adaptive"` and engage thinking
  server-side per request, and a forced `tool_choice` returns the tool call as **literal text**
  (`<invoke name="issues">…`) → `CodeReviewReport.issues` fails with a Pydantic `list_type`
  error on the batches where the model thinks (intermittent). `.with_structured_output()` sets
  exactly that forced `tool_choice`, so we route around it.

**Azure Foundry deployments of open-weight models (SGLang/vLLM) reject a forced `tool_choice`**
— they need the backend started with `--enable-auto-tool-choice`. The Kimi K2.5 and
DeepSeek-V4-Pro Azure entries that documented this were removed 2026-07-25
(`DeploymentNotFound` on the configured resource), but the pattern still applies to any
tool-use-less Foundry deployment: set `supports_tool_use: false`.
`tests/test_azure_provider.py::test_supports_tool_use_false_uses_prompt_parsing` keeps the
shape as the reference case with a synthetic config.

**`use_responses_api: true`** for GPT-5.x in `models.yaml` — the ChatCompletion API does not
support reasoning summaries for these.
