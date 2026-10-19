# `models.yaml` conventions

Background for the Configuration rules in `CLAUDE.md`. Read this before adding, renaming or
removing a model entry, or before adding a config key.

The registry contains **22 models across eight providers as of 2026-10-19**.
`codereview --list-models` is authoritative; the current provider counts are:

| Provider | Models |
| --- | ---: |
| AWS Bedrock (native Converse) | 7 |
| Azure OpenAI | 2 |
| NVIDIA NIM | 2 |
| Google GenAI | 2 |
| DeepSeek direct | 2 |
| Z.AI | 2 |
| Moonshot direct | 1 |
| OpenAI-on-Bedrock (Mantle) | 4 |

The default is **Opus 5.5**. `sonnet`/`claude-sonnet` select **Sonnet 5.5**;
`gpt6.1-sol` selects **GPT-6.1 Sol**. Opus 5, Sonnet 5, and
GPT-5.6 Sol version-specific names were removed on September 30 as curation,
after confirming that their upstream endpoints were still live.

All 14 remaining deprecated aliases were deleted on October 19 at user
request. The bundled registry has no `deprecated_aliases`; current model IDs
and advertised aliases keep their existing targets. README's *Migrating
Deleted Aliases* table lists replacements.

The [October 19 NVIDIA audit](nvidia-model-review-2026-10.md) removed
`moonshotai/kimi-k3` (HTTP 410, EOL October 15) and `z-ai/glm-5.3`
(still HTTP 200, but `deprecation: 2026-10-28T08:00:00Z`). The latter is a
scheduled retirement, following the V4-Flash precedent below. All six IDs and
aliases were deleted; none silently redirect to billed providers or Flash.
DeepSeek-V4.1-Flash and GLM-5.3-Flash remain, with Flash still the NIM default.

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
route to Moonshot direct. NVIDIA's K3 re-host was retired October 15;
`kimi-nvidia-3`/`kimi3-nvidia` were deleted on October 19. Re-host entries keep
provider-suffixed aliases only.

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
to `opus5` when Opus 5 shipped and on to `opus5.5` on 2026-09-23.
Opus 5 was removed as curation on 2026-09-30;
a superseded entry keeps
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
`deprecated_aliases`. They moved once more when 3.7 itself was curated away on
2026-09-19, then were deleted in the 2026-10-19 deprecated-alias cleanup.
`gemini-flash` still selects 3.8. 3.7's own version-explicit spellings
(`gemini-3.7-flash`, `gemini37-flash`, `gemini3.7-flash`) were also deleted.

`sonnet` was the deliberate exception for a while — it stayed on Sonnet 4.6 when Sonnet 5 shipped,
because 4.6 was the cheaper daily driver and holding the bare name there was a pricing choice, not
an oversight. The 2026-08-29 curation pass ended the exception by removing the 4.6 entry, so
`sonnet`/`claude-sonnet` moved to `sonnet5` as plain `aliases`. On 2026-09-30,
Sonnet 5.5 replaced that entry at user request, so the names now select `sonnet5.5`.
The version-explicit `sonnet5`, `claude-sonnet-5`, `sonnet-5`, and `claude-sonnet5`
were deleted. Sonnet 5 remains live upstream; this was curation.

Sonnet 5.5 preserves the 1M context and 128K output cap, but AWS serves it through
`global.anthropic.claude-sonnet-5-5` only. The replacement changes US residency to
Global routing and lowers the configured rates from $2.20/$11 to $2/$10 per M.
The AWS catalog, inference profile, and Converse invocation were checked from
us-east-1 on 2026-09-30. Its
[model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5-5.html)
lists structured outputs as unsupported; prompt parsing and a 1800s read timeout
follow the reasoning-model rules. The installed LangChain package has no Sonnet
5.5 profile, so the removed Sonnet 5 divergence needs no replacement allowlist row.

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
neither `--list-models` (credential-free, reads the YAML) nor NVIDIA's
`--validate` (endpoint connectivity, without model membership or invocation
checks) could detect their retirement. NIM retires endpoints on a rolling basis, so treat the NVIDIA roster as the
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
YAML, NVIDIA's `--validate` checks endpoint connectivity, and the removal procedure above says "probe a real
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
- **Evidence outlives the entry.** A removed model's observed behavior can
  inform later entries: Opus 4.8's literal-text reproduction informed Opus 5
  and Sonnet 5, and remains relevant to Fable 5 and the 5.5 releases.
  GPT-5.5's reasoning-only failure informed GPT-5.6 Sol and the later GPT-6
  family. Keep the evidence in comments or `docs/`, with observed failures
  distinguished from conservative defaults on successors.

The bundled registry's removal policy, updated 2026-10-19:

- **Current family aliases** (`opus`, `sonnet`, `gemini-flash`, `kimi`) may
  track the current model as advertised `aliases`, provided the family,
  tier, and any provider suffix remain accurate.
- **Deprecated compatibility aliases** are deleted. Do not park removed
  names under `deprecated_aliases` in the bundled registry. A removed name
  should fail explicitly so the caller can choose its replacement.
- **Version-specific or provider-specific names** are deleted when the
  version or provider they name is removed. A silent redirect can change
  model behavior, credentials, or billing. Add every deleted name to
  `RETIRED_ALIASES_DELETED_NOT_REDIRECTED` in `tests/test_config.py` with a
  reason, and give it a row in README's *Migrating Deleted Aliases* table.

The same rule applies to `CodeAnalyzer`'s deprecated `model_id` argument.
`global.anthropic.claude-sonnet-4-6` no longer redirects to the floating
`sonnet` alias. Select `model_name="sonnet"` explicitly to use the current
generation, or `model_name="sonnet5.5"` to pin the configured version.

Earlier compatibility redirects included `glm5`/`glm-5` from NVIDIA to
Bedrock and then Z.AI, Azure Kimi/DeepSeek names to their direct APIs, and
`gpt-bedrock` through successive GPT releases. Those redirects ended on
2026-10-19. Select `glm`, `kimi`, `deepseek-v4-pro`, or `gpt6.1-sol`
explicitly. Z.AI serves a distinct `glm-5`, so redirecting that name to 5.3
was especially ambiguous.

Current provider-specific names remain valid where their suffix is true:
`kimi-bedrock` selects Bedrock K3 and `grok-bedrock` selects native Bedrock
Grok 4.7. Retired version names remain deleted. The `qwen*`, `step*`,
`mistral*`, and `minimax*` families have no configured successor.

## `aliases` vs `deprecated_aliases` is purely a display split

The schema retains `deprecated_aliases` for custom configurations.
`ConfigLoader._register_all_names` registers both lists identically.
`--list-models` renders custom deprecated names as `+N deprecated` and spells
them out under `--verbose`. The bundled registry uses only `aliases` and
contains no deprecated names.

## Registry guards in `tests/test_config.py`

- `test_current_family_aliases_resolve_to_expected_models` — current names
  retain their expected model and provider.
- `test_bundled_registry_has_no_deprecated_aliases` — bundled entries use
  current names only.
- `test_custom_deprecated_aliases_still_resolve` — custom configurations
  retain opt-in compatibility aliases.
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
network. Profile coverage depends on the installed package versions; run that
test module to check the current registry. Every NVIDIA, Z.AI, Moonshot, and
Mantle entry currently lacks a profile.
The other missing entries are Opus 5.5, Sonnet 5.5, Kimi K3 on Bedrock,
Grok 4.7, GLM-5.3 on Bedrock (langchain-aws profiles only `zai.glm-5`), DeepSeek-V4-Flash, and Gemini 3.8 Flash. A missing profile means
there is nothing to cross-check in the installed partner package.

**Neither side is authoritative**: the tables are generated from the community-curated
[models.dev](https://github.com/sst/models.dev), and our `supports_tool_use` is *empirical* — the
whole structured-output matrix (`docs/structured-output.md`) exists because models advertising
`structured_output: true` fail on the forced `tool_choice` anyway. So a disagreement is a prompt
to check, and deliberate differences are allowlisted with a reason each. Only the Azure
GPT-5.4 Pro difference remains after Opus 5 and Sonnet 5 were removed; a stale
allowlist row is itself a test failure, see rule 2.

Three design rules:

1. Assertions are **one-directional** where a direction exists (a conservative
   `max_output_tokens`/`context_window` is a valid cost choice; only *exceeding* the cap is a
   bug).
2. A separate test fails when an allowlist entry stops diverging, so it can't accumulate
   permission for problems already fixed.
3. Re-host ids are **not** mapped onto the direct API's profile. The retired
   GPT-5.6 Sol entry illustrated the hazard: stripping `openai.` would have
   compared its 272K Mantle window against the direct API's 1.05M. The guard
   now uses GPT-6 Astra and checks every remaining Mantle entry.
   Only `strip_cross_region_prefix` is applied, because a Bedrock
   inference-profile prefix names the *same* endpoint and langchain-aws's
   table carries both spellings with identical limits.

What makes the profiles worth checking at all: langchain-aws **acts** on its own table at
runtime — `_default_params` drops `temperature`/`top_p` whenever `profile["temperature"] is
False`. Two meta-guards prevent a vacuous pass (the failure the retry and token-usage contracts
were written against): one test asserts the private lookup still resolves *something*, and a
per-provider coverage pin fails when a provider drops from some coverage to none.
