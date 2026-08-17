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
route to Moonshot direct, not Bedrock's K2.5 (`kimi-bedrock`) or NVIDIA's K2.6
(`kimi-nvidia-26`). Re-host entries keep provider-suffixed aliases only.

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
rename was needed) when Google deprecated `gemini-3-flash-preview`. `sonnet` is the deliberate
exception — it stayed on Sonnet 4.6 when Sonnet 5 shipped.

## Removing a model must not break a `--model` invocation *silently*

The 2026-07-25 cleanup dropped 11 entries (41 → 30 models); the rule it established: verify
against the **live provider endpoint** (Bedrock `ListFoundationModels`, NIM `GET /v1/models`, an
actual Azure call, an HTTP probe) rather than release notes, and remove only entries that are
dead, unreachable from the configured region, or strictly superseded at equal-or-worse
price/context. Leave a dated comment in `models.yaml` at the removal site recording what the probe
showed and whether the endpoint is still live, so re-adding from git history is a judgement call
with the evidence attached.

What happens to the removed entry's identifiers depends on **whether the name states a version**
(narrowed 2026-07-25 from a blanket "migrate everything"):

- **Version-neutral name** (`glm5`, `kimi-azure`, `gemini-3-flash`, `qwen-bedrock`, `step-flash`)
  → **migrate onto the live successor**, as `deprecated_aliases`. The user asked for "the GLM
  one"; giving them the current GLM one is what they meant. Canonical-owner and
  generation-neutral conventions decide which successor.
- **Version-explicit name** (`opus4.6`, `glm51`, `mm25`, `kimi25`, `step35`, `gpt5.4-bedrock`) →
  **delete it**. A name that says "4.6" resolving to a two-generations-newer model with different
  pricing, different sampling-param support and a different structured-output path is worse than
  an error a human reads and fixes. Add every deleted name to
  `RETIRED_ALIASES_DELETED_NOT_REDIRECTED` in `tests/test_config.py` with a one-line reason, and
  give it a row in README's *Migrating Deleted Aliases* table.

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
network. 15 of 30 entries resolve one; the misses are re-hosts whose wire ids the tables don't
carry (all of NVIDIA, Z.AI, `bedrock_openai`) plus anything newer than the installed package.

**Neither side is authoritative**: the tables are generated from the community-curated
[models.dev](https://github.com/sst/models.dev), and our `supports_tool_use` is *empirical* — the
whole structured-output matrix (`docs/structured-output.md`) exists because models advertising
`structured_output: true` fail on the forced `tool_choice` anyway. So a disagreement is a prompt
to check, and the eight deliberate ones are allowlisted with a reason each.

Three design rules:

1. Assertions are **one-directional** where a direction exists (a conservative
   `max_output_tokens`/`context_window` is a valid cost choice; only *exceeding* the cap is a
   bug).
2. A separate test fails when an allowlist entry stops diverging, so it can't accumulate
   permission for problems already fixed.
3. Re-host ids are **not** mapped onto the direct API's profile (`openai.gpt-5.5` → `gpt-5.5`
   would light up three more rows and compare our 400K Bedrock window against the direct API's
   1.05M — the exact over-claim the file exists to catch). Only `strip_cross_region_prefix` is
   applied, because a Bedrock inference-profile prefix names the *same* endpoint and langchain-aws's
   table carries both spellings with identical limits.

What makes the profiles worth checking at all: langchain-aws **acts** on its own table at
runtime — `_default_params` drops `temperature`/`top_p` whenever `profile["temperature"] is
False`. Two meta-guards prevent a vacuous pass (the failure the retry and token-usage contracts
were written against): one test asserts the private lookup still resolves *something*, and a
per-provider coverage pin fails when a provider drops from some coverage to none.
