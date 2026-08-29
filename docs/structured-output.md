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
3. It is `ChatBedrockConverse`-only, while 5 of our 9 prompt-path entries are NVIDIA /
   Moonshot / Z.AI / `bedrock_openai`, so adopting it means two structured-output paths to
   maintain.

Upstream's own docstring names its target as non-thinking models ("notably Amazon Nova"),
which is consistent with all of the above.

## The path matrix

| Model (provider) | Thinking | `supports_tool_use` | Path | Why prompt-parsing (if so) |
|---|---|---|---|---|
| Claude Fable 5 (Bedrock) | adaptive (always on, can't disable) | `false` | prompt | Same forced-`tool_choice`-while-thinking conflict reproduced on Opus 4.8 (see below), but **constant** rather than intermittent — thinking can't be disabled. Also rejects `temperature`/`top_p`/`top_k`; requires one-time `provider_data_share` data-retention opt-in |
| **Claude Opus 5 (Bedrock)** | **on by default** (effort-controlled) | `false` | prompt | **Documented, not assumed**: the Bedrock model card lists *Structured outputs: Not Supported* on both `bedrock-runtime` and `bedrock-mantle`. Thinking-on-by-default also reproduces the Opus 4.8 forced-`tool_choice` conflict. Current CLI default; also needs `read_timeout: 1800` (Fable 5's non-streaming-Converse problem) |
| Claude Sonnet 5 (Bedrock) | adaptive (on by default, server-side) | `false` | prompt | Same forced-`tool_choice`-while-thinking conflict as Opus 4.8 — first Sonnet tier with adaptive thinking on by default. Also rejects `temperature`/`top_p`/`top_k`. No `provider_data_share` opt-in (unlike Fable 5); geo-US routes from the us-west-2 default. Owns the generation-neutral `sonnet`/`claude-sonnet` since the Sonnet 4.6 entry was removed 2026-08-29 |
| Claude Haiku 4.5 (Bedrock) | opt-in, and we never ask | `true` | tool-use | Thinking is off unless requested and this entry doesn't request it, so there is no forced-`tool_choice` conflict to route around — and since the 2026-08-29 curation pass this is the **only** Bedrock entry on the tool-use path. Keep it that way when trimming: it is the case that proves the Bedrock tool-use path still works at all. Takes `temperature` (0.1) |
| GPT-5.6 Sol (**Bedrock** `bedrock-mantle` OpenAI-compat) | adaptive (server-side) | `false` | prompt | The reasoning-only failure mode **live-verified on GPT-5.5** at this same endpoint (entry removed 2026-08-29): think-heavy batches came back `tool_calls=[]` with no `parsed` → "no 'parsed' field", intermittently. **Responses API only** (Chat Completions not supported → `use_responses_api: true` required), no `temperature`/`top_p`. Sol tier = OpenAI's best coding model; In-Region us-east-1/us-east-2 only |
| GPT-5.4 / 5.4 Pro (**Azure**) | reasoning | `true` | tool-use | Azure deployment tolerates forced `tool_choice`; Bedrock's endpoint does not |
| **Kimi K3 (NVIDIA)** | **always on, no off switch** | `false` | prompt | 2.8T/104B MoE, 1M context, native multimodal. Model card: *"Thinking is always enabled"* — so this is the **constant** forced-`tool_choice`-while-thinking profile (like Fable 5), not the intermittent one. Tool-use unverified: NIM's free tier 429'd every forced-`tool_choice` probe, so the assume-prompt-parsing rule decides it. K2.6 on Moonshot is prompt-path for the same reason. Effort levels are low/high/max, but **no `reasoning_effort` is set** — `InferenceParams` only permits up to `high` and the wire spelling couldn't be verified |
| Kimi K2.6 (Moonshot) | enabled (server-side) | `false` | prompt | Moonshot rejects `tool_choice='specified'` (HTTP 400) while thinking. Sole Kimi entry outside NVIDIA since the K2.5-on-Bedrock and K2.6-on-NVIDIA entries were removed 2026-08-29 |
| DeepSeek V4 family (**DeepSeek direct**) | on by default (both V4-Pro and V4-Flash) | `true` | tool-use | Thinking is on by default and rejects a forced `tool_choice` (HTTP 400), but **the provider explicitly sends `thinking: disabled`** so tool calling works — tool-use is a property of us disabling thinking, not of the model. **`inference_params.thinking: enabled` flips this entry to the prompt path at runtime** (see `deepseek._create_model`) |
| MiniMax M3 (NVIDIA) | enabled (interleaved) | `false` | prompt | New reasoning/thinking model — assume prompt-parsing until a live run proves tool-use (forced `tool_choice` while thinking is unproven on this endpoint). Live-verified working on the prompt path. Owns the whole MiniMax alias lineage after M2.7 was removed 2026-07-25 and M2.5-on-Bedrock 2026-08-29 |
| GLM 5 (Bedrock) | on by default (reasoning_effort=max) | `false` | prompt | Thinking model → forced `tool_choice` auto-downgraded/returned as text; assume-prompt-parsing until proven (positive Converse report was for GLM-4.7, not GLM-5). **Absorbed the version-neutral `glm5`/`glm-5` aliases** 2026-08-29 when the GLM-5.2-on-NVIDIA entry was removed — they name GLM *5*, and this is the live GLM 5 |
| GLM-5.2 (Z.AI) | enabled (server-side) | `false` | prompt | Z.AI's endpoint ignores `json_schema` response_format and returns markdown-fenced JSON (`PydanticOutputParser` strips the fences) **and** it's a thinking model → assume-prompt-parsing rule. 1M context, only Z.AI entry (GLM-5.1 removed 2026-07-25; its aliases resolve here). Unverified live; flip to `true` only if a live run proves tool-use |
| Gemini 3.7 Flash (Google) | supported low/medium/high (`minimal` errors) | `true` (default) | tool-use | **The Gemini exception, and the only live-proven one left** (2026-08-17): card lists Structured outputs *and* Function calling as Supported, and three live runs each returned a valid `CodeReviewReport` with `parsing_error` None and `output_token_details.reasoning > 0` — tool-use held *while thinking*, which is the condition the rule exists for. Owns the generation-neutral `gemini-flash` plus 3.6's `gemini-3-flash`/`gemini3-flash`/`g3flash`; sampling params omitted (3.6-onward rule) |
| Everything else (Gemini 3.1 Pro, DeepSeek V4 family on NVIDIA) | — | `true` (default) | tool-use | Standard `.with_structured_output()` |

**Two separate 2026-08-29 passes removed rows from this matrix; don't conflate them.**

*Endpoint EOL* took four prompt-path rows for a reason that has nothing to do with structured
output: NVIDIA end-of-lifed the endpoints. Qwen3.5 397B (XML tool calls inside the `<think>`
block), GLM-5.2-on-NVIDIA (malformed/truncated tool-call JSON), Step 3.7 Flash (always-thinking,
unproven) and Mistral Small 4 119B (non-thinking but empirically prompt-path) all answer HTTP 410
now. If NVIDIA re-publishes any of them, restore the row rather than re-deriving the path.

*Curation* then took nine more — Opus 4.8, Sonnet 4.6, Kimi K2.5-on-Bedrock,
Qwen3-Coder-Next-on-Bedrock, MiniMax M2.5-on-Bedrock, Kimi K2.6-on-NVIDIA, Gemini 3.6 Flash,
GPT-5.5-on-Bedrock and Grok 4.3-on-Bedrock. **Every one of those endpoints is still live**, so
these rows are absent by choice, not by upstream removal, and re-adding an entry means restoring
its row rather than re-deriving the path. Three observations from that set are load-bearing for
rows that remain and are preserved below rather than only in `git log`: Opus 4.8's literal-text
reproduction (Opus 5 / Sonnet 5 / Fable 5 rest on it), GPT-5.5's reasoning-only failure (GPT-5.6
Sol rests on it), and Kimi K2.5-on-Bedrock's tool-call-marker leakage (the "mangles the tool
call" shape's clearest case). Gemini 3.6 Flash was the *first* model to earn `true` back with a
live run; 3.7 Flash's row now carries that precedent.

## The two failure shapes

Two distinct shapes drive the `false` cases:

**"Can't tool-call at all / mangles the tool call"** — MiniMax M3, GLM-5.2-on-Z.AI fenced JSON.
The clearest case was Kimi-K2.5-on-Bedrock's marker leakage (entry removed 2026-08-29), which is
why that observation is kept above.

**"Can tool-call but not *while thinking*"** — Opus 5, Sonnet 5, Fable 5, GLM 5,
GPT-5.6-Sol-on-Bedrock, K2.6, K3. These are intermittent, except the always-on-thinking models
(Fable 5, K3), which are constant.

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

**MiniMax M3 on NVIDIA, Kimi K3 on NVIDIA, Kimi K2.6 on Moonshot, Claude Opus 5, Sonnet 5 and
Fable 5 on Bedrock, GPT-5.6 Sol on `bedrock-mantle`, GLM 5 on Bedrock and GLM-5.2 on Z.AI** lack
usable tool-based structured output.

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
- **Sonnet 5 (and Opus 4.8, whose entry was removed 2026-08-29)** support only
  `thinking.type: "adaptive"` and engage thinking server-side per request, and a forced
  `tool_choice` returns the tool call as **literal text** (`<invoke name="issues">…`) →
  `CodeReviewReport.issues` fails with a Pydantic `list_type` error on the batches where the
  model thinks (intermittent). `.with_structured_output()` sets exactly that forced
  `tool_choice`, so we route around it. Opus 4.8 is where this was actually reproduced
  (`de5e2fc`), which is why it is still named here — the model is live on Bedrock
  (`us.anthropic.claude-opus-4-8`), only our entry is gone.

**Azure Foundry deployments of open-weight models (SGLang/vLLM) reject a forced `tool_choice`**
— they need the backend started with `--enable-auto-tool-choice`. The Kimi K2.5 and
DeepSeek-V4-Pro Azure entries that documented this were removed 2026-07-25
(`DeploymentNotFound` on the configured resource), but the pattern still applies to any
tool-use-less Foundry deployment: set `supports_tool_use: false`.
`tests/test_azure_provider.py::test_supports_tool_use_false_uses_prompt_parsing` keeps the
shape as the reference case with a synthetic config.

**`use_responses_api: true`** for GPT-5.x in `models.yaml` — the ChatCompletion API does not
support reasoning summaries for these.
