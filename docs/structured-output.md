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
3. It is `ChatBedrockConverse`-only, while 9 of our 13 prompt-path entries are NVIDIA /
   Moonshot / Z.AI / Google / `bedrock_openai`, so adopting it means two structured-output
   paths to maintain.

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
| GPT-6 Astra (**Bedrock** `bedrock-mantle` OpenAI-compat) | adaptive (server-side) | `false` | prompt | **Live-verified 2026-09-13 by A/B on `codereview/providers/`** (2 batches, ~98K input): with `true`, batch 1 died on `ResponseError(code='server_error')` from the Responses API after burning the retry budget — 3m06s for a half-finished review; with `false`, both batches finished in 37.4s with 4 issues. **A different symptom from GPT-5.5's** reasoning-only `tool_calls=[]`/no-`parsed` response — this endpoint 500s instead — but the same root cause and fix. A trivial batch passes forced `tool_choice` cleanly, so only a think-heavy target reproduces it. Keep Responses for reasoning summaries; no `temperature`/`top_p`. us-west-2 only |
| GPT-5.4 / 5.4 Pro (**Azure**) | reasoning | `true` | tool-use | Azure deployment tolerates forced `tool_choice`; Bedrock's endpoint does not |
| **Kimi K3 (NVIDIA)** | **always on, no off switch** | `false` | prompt | 2.8T/104B MoE, 1M context, native multimodal. Model card: *"Thinking is always enabled"* — so this is the **constant** forced-`tool_choice`-while-thinking profile (like Fable 5), not the intermittent one. Tool-use unverified: NIM's free tier 429'd every forced-`tool_choice` probe, so the assume-prompt-parsing rule decides it. K3 on Moonshot-direct is prompt-path too, and there it is **live-verified** rather than assumed — see the row below, which is the best available evidence for this one. Effort levels are low/high/max, but **no `reasoning_effort` is set** — `InferenceParams` only permits up to `high` and the wire spelling couldn't be verified |
| **Kimi K3 (Moonshot)** | **always on (server-side), no off switch** | `false` | prompt | **Live-verified 2026-09-19, not assumed**: a forced `tool_choice='specified'` returns HTTP 400 *"tool_choice 'specified' is incompatible with thinking enabled"* — the vendor names the conflict in the error string. Byte-identical to the K2.6 failure this entry replaced, and K3 cannot turn thinking off, so it is **constant** rather than intermittent. Sole Kimi entry outside NVIDIA. 2.8T/104B MoE, 1M context; `reasoning_effort: high` is pinned down from the card's `max` default and forwarded by `moonshot.py` |
| DeepSeek V4 family (**DeepSeek direct**) | on by default (both V4-Pro and V4-Flash) | `true` | tool-use | Thinking is on by default and rejects a forced `tool_choice` (HTTP 400), but **the provider explicitly sends `thinking: disabled`** so tool calling works — tool-use is a property of us disabling thinking, not of the model. **`inference_params.thinking: enabled` flips this entry to the prompt path at runtime** (see `deepseek._create_model`) |
| GLM-5.3 / GLM-5.3-Flash (Z.AI) | always on, low/high/max (default max) | `false` | prompt | Both are new always-thinking models. Z.AI advertises Function Calling and Structured Output, but no live review has proved forced tool use while thinking; the assume-prompt-parsing rule applies. GLM-5.2's endpoint also returned fenced JSON on this provider path, reinforcing the conservative default. |
| Gemini 3.8 Flash (Google) | supported low/medium/high (`minimal` errors) | `false` | prompt | New thinking model: its card advertises Structured outputs and Function calling, but policy requires a live review proving forced tool use while thinking. No such run has been recorded yet, and since the Gemini 3.7 Flash entry was curated away on 2026-09-19 there is **no longer a live-proven counter-example anywhere in the registry** — 3.7 was it. Owns `gemini-flash` plus 3.7's `gemini-3-flash`/`gemini3-flash`/`g3flash`; sampling params omitted. |
| **GLM-5.3 (NVIDIA)** | **always on, low/high/max (default max)** | `false` | prompt | 753B/40B MoE, 1M context, text-only, NVFP4 on GB300. Same always-thinking profile as its Z.AI-direct twin above, and the NIM card's *"tool calls are emitted in OpenAI-compatible form"* says nothing about a **forced** `tool_choice` surviving thinking — which is what `.with_structured_output` sets — so the assume-prompt-parsing rule decides it. Unlike Kimi K3 this row **does** set `reasoning_effort: high`: the card's default is `max`, NIM bills reasoning inside `completion_tokens`, and `InferenceParams` caps at `high` anyway. **Latency is the operational catch, not the path**: ~4m50s to return 8 tokens on a probe and 19m0s for a real one-file review, so a forced-tool-use A/B here is expensive to run |
| **GLM-5.3-Flash (NVIDIA)** | **always on, low/high/max (default max)** | `false` | prompt | 320B/18B MoE, 1M context, natively multimodal (up to 8 images/request), native FP8 on 8×H100 and fast. Same reasoning as the row above, same `reasoning_effort: high`. This is the NIM default (`nvidia_default`) and the cheapest place to attempt the live forced-tool-use run that would flip either GLM-5.3 row to `true` |
| Everything else (Gemini 3.1 Pro) | — | `true` (default) | tool-use | Standard `.with_structured_output()` |

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
live run, and 3.7 Flash carried that precedent until its own entry was curated away on
2026-09-19 — **so the precedent now survives only in this paragraph.** No live `true` remains in
the matrix for a thinking model. That matters for the rule: 3.7's three clean runs
(`parsing_error` None, `output_token_details.reasoning > 0`) are what proved
assume-prompt-parsing is a *default*, not a law, and the bar for flipping Gemini 3.8 Flash or
either GLM-5.3 row is still to reproduce exactly that shape. Restore the 3.7 entry from git
history if you want the comparison back.

*A third pass, 2026-09-19*, was NVIDIA again — three rows out, two in. MiniMax M3 (prompt path,
EOL 2026-09-09) and DeepSeek-V4-Pro-0813 (tool-use path, EOL 2026-09-14) both answer HTTP 410, and
DeepSeek-V4-Flash-0731 (tool-use path) went with them on a **scheduled sunset advertised in a
`deprecation: 2026-09-21T08:00:00Z` response header** while still returning HTTP 200. The two
GLM-5.3-on-NVIDIA rows above replaced them.

Three things worth keeping. MiniMax M3 was the last *current* entry in the "mangles the tool call"
failure shape, so that shape now rests entirely on retired evidence. V4-Pro-0813 was the control in
the DeepSeek-on-NIM thinking comparison — it did **not** reason unless asked, which is how we knew
`thinking: false` on V4-Flash-0731 was load-bearing rather than decorative.

And the structural consequence: **every NVIDIA NIM entry is now on the prompt path.** V4-Flash-0731
was the only NIM row ever to hold the tool-use path, and it held it only because `thinking: false`
switched its reasoning off — a property of our config, not of the endpoint. Every model NIM now
serves us reasons unconditionally. So there is no longer a NIM row proving the tool-use path works
on this provider at all, which is the same gap the Haiku 4.5 row exists to close on Bedrock. If a
non-thinking model appears on NIM, prefer keeping it on the tool-use path for exactly that reason.

*A fourth pass, also 2026-09-19*, was curation at explicit request rather than breakage: **GLM 5
(Bedrock)** and **Gemini 3.7 Flash** out, and **Kimi K2.6 (Moonshot)** replaced in place by
**Kimi K3**. Both removed endpoints are believed live — neither was probed as dead, and the GLM
one could not be re-verified at all, so it is deliberately **not** in `DEAD_UPSTREAM_FULL_IDS`.
Two consequences for this document. Losing 3.7 Flash costs the matrix its only live `true` on a
thinking model, discussed above. Losing GLM 5 means **Bedrock now serves Claude only**, so the
Claude reasoning tiers and Haiku 4.5 are the whole Bedrock picture: the "assume prompt parsing on
a third-party Bedrock re-host" case has no current instance. In exchange, the K3 row is the first
`false` in the matrix backed by the vendor *naming the conflict in an error string*, which is
stronger evidence than anything the retired rows carried.

## The two failure shapes

Two distinct shapes drive the `false` cases:

**"Can't tool-call at all / mangles the tool call"** — the retired MiniMax M3 and
GLM-5.2-on-Z.AI entries' fenced JSON. The clearest case was Kimi-K2.5-on-Bedrock's marker
leakage (entry removed 2026-08-29), which is why that observation is kept above. No *current*
entry sits in this shape; every live `false` below is the thinking conflict.

**"Can tool-call but not *while thinking*"** — Opus 5, Sonnet 5, Fable 5,
GPT-5.6-Sol-on-Bedrock, GPT-6-Astra-on-Bedrock, K3 on both providers. These are intermittent,
except the always-on-thinking models (Fable 5, K3), which are constant. **K3-on-Moonshot is the
cheapest live reproduction in the registry**: one forced-`tool_choice` call returns an HTTP 400
naming the conflict, no think-heavy batch required.

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

**Kimi K3 on NVIDIA and on Moonshot, Claude Opus 5, Sonnet 5 and
Fable 5 on Bedrock, GPT-5.6 Sol and GPT-6 Astra on `bedrock-mantle`, Gemini 3.8 Flash, and the
GLM-5.3 family on both Z.AI and NVIDIA** lack usable tool-based structured output — 12 of the 18
entries.

- **Opus 5** is the one case with vendor confirmation rather than inference: its Bedrock model
  card lists *Structured outputs: Not Supported* for both `bedrock-runtime` and
  `bedrock-mantle`, and thinking is on by default (a breaking change from Opus 4.8, where it
  was off unless requested) so it also hits the forced-`tool_choice`-while-thinking conflict.
  Opus 5 additionally rejects sampling params and needs `read_timeout: 1800`.
- **Sonnet 5** is the first Sonnet-tier model with adaptive thinking on by default, so it
  inherits the exact Opus 4.8 conflict (unverified live; ships `false` under the rule).
- **GLM-5.3 and GLM-5.3-Flash (Z.AI)** are always-thinking models and ship on
  prompt parsing under the assume-prompt-parsing rule. Z.AI advertises Function Calling and
  Structured Output for both, but that does not prove LangChain's forced `tool_choice` works
  while reasoning is active. Flip either entry only after a live review demonstrates that
  condition. The removed GLM-5.2 entry supplied additional provider-path evidence: Z.AI's
  OpenAI-compatible endpoint ignored the `json_schema` response format and returned fenced JSON,
  which `PydanticOutputParser` handled.
- **K3 (Moonshot)** — Moonshot's server rejects `tool_choice='specified'` with HTTP 400
  *"tool_choice 'specified' is incompatible with thinking enabled"*. Re-verified on K3 when this
  entry replaced K2.6 on 2026-09-19, so it is not inherited on faith; and because K3's thinking
  has no off switch, the 400 is unconditional rather than batch-dependent. This is the registry's
  one *positive* confirmation of a `false` value from the vendor's own error message — cite it
  when someone asks whether the assume-prompt-parsing rule ever actually catches anything.
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
