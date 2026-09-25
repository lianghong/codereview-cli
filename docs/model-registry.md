# `models.yaml` conventions

Background for the Configuration rules in `CLAUDE.md`. Read this before adding, renaming or
removing a model entry, or before adding a config key.

## `ConfigLoader` must forward every key it parses

A key present in `models.yaml` but absent from the `<Name>Config` construction is invisible: no
error, no warning, and the setting appears to work because the field has a default.
`NVIDIAConfig.max_retries` was unreachable this way. When you add a field to a provider config
class, add it to the parsing branch in `loader.py` in the same commit.

**The same hazard exists one level down, on a model entry's `pricing` and `inference_params`**,
and there it shipped sixteen times: six `cache_write_per_million`/`cache_read_per_million` pairs
plus four `cached_input_per_million` keys, none of which `PricingConfig` declares and none of
which `_parse_model_config` copies — so they loaded silently and could never reach a cost figure,
while reading to a human as a rate the tool honors. An unread *pricing* number is the worst
version of this, because the next reader trusts it.

`test_every_pricing_and_inference_key_in_the_yaml_is_actually_read` (`tests/test_config.py`) now
scrapes the keys the loader reads and rejects any YAML key outside that set; it deliberately
covers only `pricing`/`inference_params`, since `capabilities`/`architecture`/`notes` are
documented doc-only. If you want a new knob, wire it through `_parse_model_config` **and** the
Pydantic model — a comment in the YAML is not configuration.

The scrape matches `pricing_data["key"]` / `pricing_data.get("key")` on **one line**, so keep the
key spelling unwrapped — `long_context_threshold_tokens` is bound to a local in `loader.py` for
exactly that reason. A call `ruff format` wrapped across lines passes the assertion vacuously.

## Tiered pricing: three keys or none

An entry whose vendor charges by request size sets all three of
`long_context_threshold_tokens`, `long_input_per_million`, `long_output_per_million`;
`PricingConfig` raises on a partial set, because two of the three silently fall back to the flat
pair. GPT-6 Astra is the reference case ($11/$55 per M at or below 272K input tokens,
$22/$82.50 above). Two things to know before adding one:

- **The threshold is per API call, not per run.** Tier selection happens in
  `TokenTrackingMixin._track_tokens`; never in `estimate_cost`, which sees only totals. →
  `docs/providers.md`
- **A tier is what makes a wide `context_window` safe.** Astra's window was clamped to the
  272K price break until the code could price both tiers, since a flat pair reports half the
  billed cost on any batch past the break. `context_window` itself stays a conservative round
  number under the card's figure (Astra 1,000,000 against 1,050,000; Gemini 1,000,000 against
  1,048,576).

## Doc-only YAML

The `defaults:` block (`zai_default`, `bedrock_default`, …) and a model's
`capabilities`/`architecture`/`notes` keys are **informational only** — no Pydantic class reads
them and `ModelConfig` isn't `extra="forbid"`. The CLI's real default `--model` is hardcoded
(`opus5.5`) in `cli.py`; changing a `*_default` won't change runtime behavior.

## Canonical-owner convention

When the same model is exposed by both a vendor's direct API and a re-hoster
(Bedrock/NVIDIA/Azure), the **direct API owns the canonical aliases**. E.g. `deepseek-v4-pro`
routes to DeepSeek direct — it held that name against NVIDIA's free re-host `dsv4-nvidia` until
NIM end-of-lifed that endpoint on 2026-09-14; `kimi` and `kimi-k3`
route to Moonshot direct, while NVIDIA's Kimi keeps `kimi-nvidia-3`/`kimi3-nvidia`. Re-host
entries keep provider-suffixed aliases only.

The convention also decides which entry survives a curation pass: on 2026-08-29 the K2.5 and
MiniMax re-hosts on Bedrock were dropped in favour of the canonical owner (Moonshot direct) and
the newer generation (MiniMax M3 on NVIDIA) respectively, because a re-host of something the
registry already carries from its owner earns its place only on price, context or availability.

That second case also shows the risk in keeping only the re-host: NIM end-of-lifed MiniMax M3 three
weeks later (2026-09-09), so cutting the still-live Bedrock entry in favour of it left the registry
with **no** MiniMax at all. When curation picks a free NIM entry over a billed one elsewhere, note
that NIM is the registry's shortest-lived block and the choice is a bet on the endpoint.

## Generation-neutral aliases track the current generation

Bare family names (`opus`, `claude-opus`) belong to the newest entry in that family — they moved
to `opus5` when Opus 5 shipped and on to `opus5.5` on 2026-09-23, and a superseded entry keeps
version-explicit names only until it's retired. The CLI default is a separate decision: it names
an entry `id` (`opus5.5` since 2026-09-23), not an alias, so moving `opus` doesn't move the default.

Two traps when doing this:

1. `ConfigLoader._register_model` is **last-write-wins within a provider** — it warns only on
   *cross-provider* collisions — so if the old entry keeps the bare name as its `id`, it silently
   shadows the new entry's alias depending on YAML order. Rename the old entry's `id`, don't just
   add the alias.
2. An `id` rename is a breaking change for anyone scripting `--model <old-id>`; note it under
   Changed in the CHANGELOG.

`gemini-flash` followed the same move to `gemini-3.6-flash` (only an alias there, so no `id`
rename was needed) when Google deprecated `gemini-3-flash-preview`, then to
`gemini-3.7-flash` on 2026-08-17, and to `gemini-3.8-flash` in September 2026. The 3.7 move
showed the other half of the rule while both entries were live: 3.6 **kept** its
version-explicit `gemini-3-flash`/`gemini3-flash`/`g3flash` back-compat names, because a name
that says "3" must not jump between live minor versions. Only the generation-neutral name
travels. When the 3.6 entry was retired on 2026-08-29, all three names moved onto 3.7 as
`deprecated_aliases`. They moved once more when 3.7 itself was curated away on 2026-09-19, so all
three now sit on 3.8 alongside `gemini-flash` — two hops, each following the rule rather than
short-cutting it. 3.7's own version-explicit spellings (`gemini-3.7-flash`, `gemini37-flash`,
`gemini3.7-flash`) were **deleted**, not migrated.

`sonnet` was the deliberate exception for a while — it stayed on Sonnet 4.6 when Sonnet 5 shipped,
because 4.6 was the cheaper daily driver and holding the bare name there was a pricing choice, not
an oversight. The 2026-08-29 curation pass ended the exception by removing the 4.6 entry, so
`sonnet`/`claude-sonnet` now sit on `sonnet5` as plain `aliases` (not deprecated: this *is* the
current Sonnet, so the name is truthful and worth advertising).

## Removing a model must not break a `--model` invocation *silently*

The 2026-07-25 cleanup dropped 11 entries (41 → 30 models); the rule it established: verify
against the **live provider endpoint** (Bedrock `ListFoundationModels`, NIM `GET /v1/models`, an
actual Azure call, an HTTP probe) rather than release notes, and remove only entries that are
dead, unreachable from the configured region, or strictly superseded at equal-or-worse
price/context. Leave a dated comment in `models.yaml` at the removal site recording what the probe
showed and whether the endpoint is still live, so re-adding from git history is a judgement call
with the evidence attached.

**Catalog visibility is not invocability — probe both.** The 2026-08-29 pass re-ran the same
procedure and found **five of ten NVIDIA NIM entries dead** (32 → 27 models): Mistral Small 4,
Mistral Medium 3.5, Qwen3.5 397B, GLM-5.2 and Step 3.7 Flash all answer **HTTP 410 Gone** with
NVIDIA's own EOL date in the body. `GET /v1/models` no longer lists them either, which is why
neither `--list-models` (credential-free, reads the YAML) nor `--validate` (catalog visibility
only, and a miss is a warning by design) could surface it — the entries looked healthy right up
to the invocation. NIM retires endpoints on a rolling basis, so treat the NVIDIA roster as the
shortest-lived block in the registry and re-probe it whenever you touch this file. A 410 is
*evidence*, not a judgement call: record the date it names. (This pass concluded NIM gives "no
in-band deprecation signal"; that turned out to be wrong — see the header rule below.)

**A dated GA id is not a stable target either.** The 2026-09-19 pass re-probed the four surviving
NVIDIA entries and found **two more dead** (20 → 19 models): `minimaxai/minimax-m3` (410, EOL
2026-09-09) and `deepseek-ai/deepseek-v4-pro-0813` (410, EOL 2026-09-14). The second is the
instructive one — that dated id was adopted on 2026-08-29 precisely *because* the undated preview
had been end-of-lifed, and NIM then retired the GA release three weeks later, faster than the
preview it replaced. So re-pointing a NIM entry at a dated id buys correctness today, not
stability; the only durable response is to re-probe the block on every visit. Both entries'
aliases were deleted rather than migrated, including the ones that read as version-neutral:
`dsv4-nvidia` looked like it could follow the then-live `dsv4-flash-nvidia`, but Pro and Flash
shipped as **separate concurrent entries**, so that name meant "the Pro one" in explicit opposition
to the Flash spelling. When two tiers of one generation coexist as entries, each tier's names are
version-explicit in effect even when they don't spell a version.

**Read the response headers, not just the status — a 200 can carry its own sunset.** Later the same
day, `deepseek-ai/deepseek-v4-flash-0731` was removed too, and it is the first entry in this
registry retired on a *scheduled* sunset rather than a 410. It answered **HTTP 200** and produced a
valid completion; what condemned it was a header:

```
deprecation: 2026-09-21T08:00:00Z
```

Every check this project had would have called that endpoint healthy — `--list-models` reads the
YAML, `--validate` checks catalog visibility, and the removal procedure above says "probe a real
completion", which succeeded. So the procedure now has a third step: **probe the completion with
`curl -D -` and read the `deprecation` header.** Two consequences worth internalising. First, it
means NIM *does* warn in-band, contradicting the 2026-08-29 conclusion above — the earlier pass had
simply never looked at the headers, having only ever caught endpoints after they died. Second, a
dated sunset in the future is still a dead entry: two days of life is not worth a registry entry,
documented examples and a `nvidia_default` pointing at it, so it was removed on the strength of the
header alone rather than waiting for the 410 to confirm what NVIDIA already told us.

Its three aliases were deleted rather than migrated. They read as version-neutral and the model
survives at its canonical owner (`dsv4-flash` on DeepSeek direct), but every spelling carries
`-nvidia` and that target is **billed** where NIM was free — the Qwen-on-NVIDIA case, where a
silent free-to-billed provider switch is worse than an error a human reads and fixes. That
removal left no DeepSeek entry on NIM until V4.1-Flash was added on 2026-09-25.

**DeepSeek-V4.1-Flash is a new release, not a revival of the V4 endpoints.**
Its [NVIDIA page](https://build.nvidia.com/deepseek-ai/deepseek-v4.1-flash)
and live catalog name `deepseek-ai/deepseek-v4.1-flash`. It registers as
`deepseek-v4.1-flash-nvidia` with `dsv41-flash-nvidia` / `dsv4.1-flash-nvidia`.
The retired V4 aliases and dead wire ids stay retired. The guard in
`tests/test_config.py` now rejects those known-dead endpoints rather than banning
the entire `deepseek-ai/` prefix. The model uses prompt parsing under the
reasoning-model rule; neither the old `thinking: false` switch nor another
model's `reasoning_effort: high` is carried forward without a documented request
contract.

On 2026-09-25 a live CLI review of a three-line Python function completed in
3m55s, identified its empty-list division bug, and exported valid JSON. The
short completion/header probes timed out, so no conclusion about a
`deprecation` header was recorded for this endpoint.

**A dead prefix can come back, so don't write "never" into a removal note.** The same 2026-09-19
pass *added* two entries to the block it had just cut down: `z-ai/glm-5.3` and `z-ai/glm-5.3-flash`
(19 → 18 with DeepSeek-V4-Flash gone, then → 20). The GLM-5.2 removal note from three weeks
earlier had recorded that "NIM now
serves no `z-ai/*` model at all" — true when written, false three weeks later. That note was
correct to delete the 5.2 aliases and is still correct to keep them deleted, because each of them
spells a version 5.3 is not; what needed amending was only the claim about the prefix. Removal
notes should record what a probe showed on a date, which stays true, rather than what a vendor will
do, which does not.

**Curation is the other reason to remove an entry, and it reads differently at every step.** A
second 2026-08-29 pass cut nine more entries (27 → 18) at the user's direction: Opus 4.8, Sonnet
4.6, Kimi K2.5-on-Bedrock, Qwen3-Coder-Next-on-Bedrock, MiniMax M2.5-on-Bedrock,
Kimi K2.6-on-NVIDIA, Gemini 3.6 Flash, GPT-5.5-on-Bedrock and Grok 4.3-on-Bedrock. **All nine
endpoints are live** — the probes were run, and they answer. Three consequences:

- **None of their `full_id`s may enter `DEAD_UPSTREAM_FULL_IDS`.** That set means "pointing an
  entry here is a bug"; a live-but-unwanted id in it would block a future re-add on false
  grounds. `tests/test_config.py` carries a comment saying so, listing the nine wire ids.
- **The removal comment must say "this was CURATION" and name what the registry loses.** Cutting
  Qwen3-Coder-Next took Bedrock's cheapest entry ($0.50/$1.20 → `haiku` at $1.00/$5.00) *and* one
  of only three Bedrock entries on the tool-use path; cutting Grok 4.3 took `bedrock_openai`'s
  cheapest ($1.25 → $5.00), its widest context, and its only entry accepting `temperature`/`top_p`.
  An entry removed as "superseded" needs the successor's price and window in the comment when
  either is worse.
- **Evidence outlives the entry.** A removed model's observed behavior can be load-bearing for
  entries that stay (Opus 4.8's literal-text reproduction underpins Opus 5 / Sonnet 5 / Fable 5;
  GPT-5.5's reasoning-only failure underpins GPT-5.6 Sol). Move it into the surviving entry's
  comment or into `docs/`, not into git history alone.

What happens to the removed entry's identifiers depends on **whether the name states a version**
(narrowed 2026-07-25 from a blanket "migrate everything"):

- **Version-neutral name** (`glm5`, `kimi-azure`, `kimi-bedrock`, `gemini-3-flash`, `sonnet`,
  `gpt-bedrock`) → **migrate onto the live successor**, as `deprecated_aliases`. The user asked
  for "the GLM one"; giving them the current GLM one is what they meant. Canonical-owner and
  generation-neutral conventions decide which successor. `sonnet` is the one that landed in plain
  `aliases` instead, because its successor is the *same family's current generation* rather than a
  sideways move — nothing about the name became untrue.
- **Version-explicit name** (`opus4.6`, `opus4.8`, `glm51`, `mm25`, `mm2.5-bedrock`, `kimi25`,
  `step35`, `gpt5.4-bedrock`, `gpt5.5-bedrock`, `grok-4.3`, `gemini-3.6-flash`) → **delete it**. A
  name that says "4.6" resolving to a two-generations-newer model with different pricing,
  different sampling-param support and a different structured-output path is worse than an error a
  human reads and fixes. Add every deleted name to `RETIRED_ALIASES_DELETED_NOT_REDIRECTED` in
  `tests/test_config.py` with a one-line reason, and give it a row in README's *Migrating Deleted
  Aliases* table.

**"Migrate onto the live successor" needs a successor that is honestly the same thing.** Three
version-neutral families were *deleted* across the two 2026-08-29 passes rather than migrated, and
each reason generalizes:

- `step-flash` / `step-nvidia` — no StepFun model remains anywhere in the registry. There is
  nothing to migrate onto, so the whole family fails fast.
- `qwen-nvidia` / `qwen3-nvidia` / `qwen-coder-nvidia` — deleted in the EOL pass because the only
  live Qwen was then `qwen-next-bedrock`, and resolving a `-nvidia`-suffixed name to Bedrock would
  move the user across **both a provider and a billing boundary** (free NIM tier → billed
  Bedrock), silently. The provider suffix is part of what the name states; treat it like a
  version. The curation pass then removed that Bedrock entry too, which took the *bare* `qwen` and
  `qwen-coder` with it — reason one, applied a second time to the same family.
- `grok` / `grok-bedrock` — no xAI model remains after Grok 4.3 was cut. `bedrock_openai` still
  *supports* Grok (the provider is not OpenAI-only), so this is the case where a live endpoint,
  working code and zero registry entries coexist: the names must still fail, because there is no
  entry for them to name.

`glm5` / `glm-5` are the counter-example that shows the line: they migrated to `glm5-bedrock`,
also a provider change, but the target *is* GLM 5 — the thing the name says. Being a
`deprecated_alias` (resolvable, unadvertised) is the right home for a name that stays truthful
about the model while changing where it comes from. (When the Bedrock re-host was curated away on
2026-09-19 they moved again, onto Z.AI's `zhipuai/glm-5.3`. That one stretches the rule and the
YAML says so: Z.AI serves a *distinct real* `glm-5`, so pointing `glm5` at 5.3 is a deliberate
choice to track the family's current release, not an identity claim. The provider-explicit
`glm5-bedrock`/`glm-5-bedrock`/`glm5b` were deleted, since the suffix names a provider that no
longer serves it.) `kimi-bedrock` → Moonshot's `kimi-k3` was the same shape until 2026-09-23,
when Kimi K3 reached Bedrock and the name moved back onto `kimi-k3-bedrock` as a plain alias — a
parked version-neutral name goes home once its suffix is true again. `gpt-bedrock` → `gpt5.6-sol-bedrock` are the same shape.

## `aliases` vs `deprecated_aliases` is purely a display split

`ConfigLoader._register_all_names` registers both identically, so resolution never differs.
`deprecated_aliases` holds the back-compat-only names inherited from removed entries;
`--list-models` renders them as `+N deprecated` and only spells them out under `--verbose`,
because advertising a name that resolves to a *different* model than it says is actively
misleading. Keep genuinely current alternative spellings in `aliases`.

## The six guards in `tests/test_config.py`

- `test_retired_model_aliases_redirect_to_live_successors` — every *migrated* identifier still
  resolves.
- `test_deleted_aliases_do_not_resolve` — every entry in the allowlist raises.
- `test_no_historical_model_id_is_orphaned` — replays the last 8 revisions of `models.yaml`; a
  name is either resolvable or explicitly allowlisted.
- `DEAD_UPSTREAM_FULL_IDS` — no entry points at a dead/region-unreachable wire id;
  superseded-but-live ids deliberately stay out of it.
- `test_no_model_points_at_missing_azure_deployment` — Azure `DeploymentNotFound` is invisible to
  `--list-models`, so it needs its own check.
- `test_documented_model_names_all_resolve` — scrapes every `--model X` out of
  `README.md`/`docs/usage.md`/`docs/examples.md` and resolves it, so renaming or deleting an alias
  fails until the prose catches up. It deliberately matches only runnable `--model` commands, so
  the migration table can keep naming dead spellings.

`ModelConfig` additionally rejects self-aliases and cross-list duplicates at load time
(`_check_alias_hygiene`) — two entries used to list their own `id` as an alias.

## Cross-checked against the partner packages' model profiles, never overwritten from them

`tests/test_model_profile_drift.py`. Each LangChain partner package ships a `_MODEL_PROFILES`
table in `<package>/data/_profiles.py`, read via the private
`_get_default_model_profile(name)` — a plain dict lookup, so no credentials, no client, no
network. 9 of 22 entries resolve one (Bedrock 4/6, Azure 2/2, DeepSeek 2/2, Google 1/2); the
misses are re-hosts and direct vendor APIs whose wire ids the tables don't carry (all of NVIDIA,
Z.AI, Moonshot and `bedrock_openai`, plus `kimi-k3-bedrock`) plus anything newer than the installed package —
`gemini-3.8-flash` and `opus5.5` (`global.anthropic.claude-opus-5-5`) are currently that last group.

**Neither side is authoritative**: the tables are generated from the community-curated
[models.dev](https://github.com/sst/models.dev), and our `supports_tool_use` is *empirical* — the
whole structured-output matrix (`docs/structured-output.md`) exists because models advertising
`structured_output: true` fail on the forced `tool_choice` anyway. So a disagreement is a prompt
to check, and the four deliberate ones are allowlisted with a reason each (there were eight before
the 2026-08-29 curation pass removed the entries behind four of them — a stale allowlist row is
itself a test failure, see rule 2).

Three design rules:

1. Assertions are **one-directional** where a direction exists (a conservative
   `max_output_tokens`/`context_window` is a valid cost choice; only *exceeding* the cap is a
   bug).
2. A separate test fails when an allowlist entry stops diverging, so it can't accumulate
   permission for problems already fixed.
3. Re-host ids are **not** mapped onto the direct API's profile (`openai.gpt-5.6-sol` →
   `gpt-5.6-sol` would compare our 272K `bedrock-mantle` window against the direct API's 1.05M —
   the exact over-claim the file exists to catch). Only `strip_cross_region_prefix` is
   applied, because a Bedrock inference-profile prefix names the *same* endpoint and langchain-aws's
   table carries both spellings with identical limits.

What makes the profiles worth checking at all: langchain-aws **acts** on its own table at
runtime — `_default_params` drops `temperature`/`top_p` whenever `profile["temperature"] is
False`. Two meta-guards prevent a vacuous pass (the failure the retry and token-usage contracts
were written against): one test asserts the private lookup still resolves *something*, and a
per-provider coverage pin fails when a provider drops from some coverage to none.
