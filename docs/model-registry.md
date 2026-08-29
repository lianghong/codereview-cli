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

## Doc-only YAML

The `defaults:` block (`zai_default`, `bedrock_default`, …) and a model's
`capabilities`/`architecture`/`notes` keys are **informational only** — no Pydantic class reads
them and `ModelConfig` isn't `extra="forbid"`. The CLI's real default `--model` is hardcoded
(`opus5`) in `cli.py`; changing a `*_default` won't change runtime behavior.

## Canonical-owner convention

When the same model is exposed by both a vendor's direct API and a re-hoster
(Bedrock/NVIDIA/Azure), the **direct API owns the canonical aliases**. E.g. `deepseek-v4-pro`
routes to DeepSeek direct, not NVIDIA's free re-host (`dsv4-nvidia`); `kimi` and `kimi-k2.6`
route to Moonshot direct, while NVIDIA's Kimi keeps `kimi-nvidia-3`/`kimi3-nvidia`. Re-host
entries keep provider-suffixed aliases only.

The convention also decides which entry survives a curation pass: on 2026-08-29 the K2.5 and
MiniMax re-hosts on Bedrock were dropped in favour of the canonical owner (Moonshot direct) and
the newer generation (MiniMax M3 on NVIDIA) respectively, because a re-host of something the
registry already carries from its owner earns its place only on price, context or availability.

## Generation-neutral aliases track the current generation

Bare family names (`opus`, `claude-opus`) belong to the newest entry in that family — they moved
to `opus5` when Opus 5 shipped, and a superseded entry keeps version-explicit names only until
it's retired.

Two traps when doing this:

1. `ConfigLoader._register_model` is **last-write-wins within a provider** — it warns only on
   *cross-provider* collisions — so if the old entry keeps the bare name as its `id`, it silently
   shadows the new entry's alias depending on YAML order. Rename the old entry's `id`, don't just
   add the alias.
2. An `id` rename is a breaking change for anyone scripting `--model <old-id>`; note it under
   Changed in the CHANGELOG.

`gemini-flash` followed the same move to `gemini-3.6-flash` (only an alias there, so no `id`
rename was needed) when Google deprecated `gemini-3-flash-preview`, and again to
`gemini-3.7-flash` on 2026-08-17. That second move showed the other half of the rule while both
entries were live: 3.6 **kept** its version-explicit `gemini-3-flash`/`gemini3-flash`/`g3flash`
back-compat names, because a name that says "3" must not jump two generations to different
pricing and capabilities. Only the generation-neutral name travels — until the older entry is
retired, at which point those names have nowhere else to go: the 3.6 entry was removed on
2026-08-29 and all three moved onto `gemini-3.7-flash` as `deprecated_aliases`.

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
to the invocation. NIM retires endpoints on a rolling basis with no in-band deprecation signal,
so treat the NVIDIA roster as the shortest-lived block in the registry and re-probe it whenever
you touch this file. A 410 is *evidence*, not a judgement call: record the date it names.

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
about the model while changing where it comes from. `kimi-bedrock` → Moonshot's `kimi-k2.6` and
`gpt-bedrock` → `gpt5.6-sol-bedrock` are the same shape.

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
network. 9 of 18 entries resolve one (Bedrock 4/5, Azure 2/2, DeepSeek 2/2, Google 1/2); the
misses are re-hosts and direct vendor APIs whose wire ids the tables don't carry (all of NVIDIA,
Z.AI, Moonshot and `bedrock_openai`) plus anything newer than the installed package —
`gemini-3.7-flash` is currently in that last group.

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
