# `validate_credentials` semantics (`--validate`)

Background for the `--validate` rules in `CLAUDE.md`. Read this before writing a new provider's
`validate_credentials` or changing what an existing one reports.

Every provider's `validate_credentials` returns a `ValidationResult`. Keep the hard-failure
(`valid=False`) vs warning distinction **consistent across providers** — an inconsistency here
is what let a bad Azure key report success once.

## The contract

- **Hard failure** (`result.valid = False`, via `add_check(..., False, msg)`) — a problem that
  *will* break the run: missing/placeholder API key, non-HTTPS `base_url`, unparseable
  endpoint, and an **explicit auth rejection from the connection test (HTTP 401/403)**. Bedrock
  additionally fails on AWS identity/credential-chain errors.
- **Warning** (`add_warning(msg)`) — non-fatal or inconclusive: unusually short key,
  missing/defaulted API version, and **inconclusive connection tests** (timeout, DNS/TLS/
  connection refused, or a non-200/401/403 status). These don't flip `valid` because the run may
  still succeed.

The connection test is best-effort and skippable via `CODEREVIEW_SKIP_CONNECTION_TEST=1`. The
401/403→hard-fail rule applies to every provider that runs a connection test (Azure, NVIDIA);
providers without one (DeepSeek, Moonshot, Z.AI, OpenAI-on-Bedrock) validate key presence +
HTTPS only and defer auth verification to the first call. When adding a provider with a
connection test, follow this same mapping.

## Placeholder keys

**The placeholder set must include the exact strings the README tells users to export** (e.g.
`your-deepseek-key`, `your-moonshot-key`) — not just the generic `placeholder` /
`your-…-api-key-here` — and is matched case-insensitively after `.strip()`, so a copied-and-not-
replaced placeholder fails fast at `--validate` instead of 401'ing on the first real call.

Use `is_placeholder_api_key(key, extra)` from `mixins.py`: the generic set lives there once; pass
the provider's README string(s) as `extra`.

Locked by `tests/test_placeholder_keys.py` — including
`test_every_readme_documented_placeholder_is_rejected`, which scrapes every
`export *_API_KEY="..."` line out of `README.md` and asserts the owning provider rejects that
literal value, so rewording an export line (or adding a provider) fails the suite until the
placeholder set catches up.

Copy the README string **verbatim**, punctuation included: OpenAI-on-Bedrock documents
`<your-amazon-bedrock-api-key>` with angle brackets, and only rejecting the bracket-less
spelling let the documented placeholder pass `--validate` and 401 later. New providers must add
their env var to `_README_KEY_EXTRAS` in that test.

## Model-access validation checks *catalog visibility only*

Never invocation permission, never inference-profile routing. Only Bedrock has such a check;
the contract, so a new provider's version doesn't quietly promise more:

| Question | What the check does | Why |
|---|---|---|
| Does the account have permission to *invoke* this model? | **Not checked.** | The only way to know is to call it, and `--validate` must not spend tokens. `bedrock:ListFoundationModels` and `bedrock:InvokeModel` are separate IAM actions, so the one we can read says nothing about the one the run needs. |
| Does an inference profile / cross-region route resolve? | **Not checked.** | `ListFoundationModels` returns *base* foundation-model ids, so the configured `full_id` is compared after `strip_cross_region_prefix` (`us.`/`eu.`/`apac.`). Whether the profile itself exists and is entitled is invisible here. |
| Is the model in this region's catalog? | **Checked, exact-matched** on the prefix-stripped base id against the `modelId` set. | Substring matching made a *version* difference read as a match — with only `minimax.minimax-m2`/`m2.1` in the account, `minimax.minimax-m2.5` reported a green "Model Access" check (verified against the live us-west-2 catalog) and then failed with `AccessDeniedException` on the first real call. `zai.glm-5` against a catalog holding only `zai.glm-5.2` did the same. |

Consequences of that scope, all deliberate:

- A catalog hit is **necessary but not sufficient** for a working run, so it is the *only* branch
  that reports success.
- A miss is a **warning, never `valid = False`** — the catalog lists what the region offers, not
  what the account was granted, and the model may well be enabled.
- An `AccessDeniedException` on `ListFoundationModels` itself is also a warning, since a
  principal can hold `InvokeModel` without `ListFoundationModels`.

A false green is the one outcome worse than "could not confirm": it sends the user to run a
review that dies on batch 1. `bedrock.py`'s check carries this reasoning inline at the
`model_found` predicate.

## The four shared predicates live in `mixins.py` — never re-spell one inline

Each existed as a per-provider one-liner first, and each drifted:

| Helper | Use it for | What the duplicate got wrong |
|---|---|---|
| `is_blank(value)` | presence of an api_key / endpoint | A bare `if not api_key` passes `"   "`: Pydantic's `min_length=1` accepts whitespace and a whitespace-only string is truthy, so the loader registered the provider *and* `--validate` reported **every** check green, deferring the failure to a 401 on the first call. Strips before testing. Accepts `None` for optional fields (NVIDIA's `base_url`). |
| `is_https_url(url)` | the cleartext-endpoint check | Providers wrote `startswith("https://")` while `require_https` lowercased first. URL schemes are case-insensitive (RFC 3986 §3.1), so `HTTPS://api.deepseek.com/v1` built a working client that `--validate` then hard-failed. It also requires a **hostname**, not just the scheme: a bare `"https://"` (or `https:///v1`) parses with `scheme == "https"` and an empty host, so a scheme-only check passed a URL no client can reach — the config error surfaced as a connection failure mid-run instead of at `--validate`. |
| `require_https(url, label)` | fail-closed enforcement in `_create_model` | Nothing — but it must stay defined *in terms of* `is_https_url`, so the constructor and the pre-flight check can't disagree in either direction (a URL validation rejects but the client accepts is a lying check; the reverse ships a key over HTTP). |
| `is_short_api_key(key)` | the "unusually short" **warning** | Five copies of `len(api_key) < 20`. Strips first, so padding can't push a short key past the bar. Stays a warning: it's a heuristic and no provider documents a minimum. |

Two tests keep this from lapsing:

- `test_every_url_checking_provider_uses_the_shared_https_predicate` (in
  `tests/test_provider_result_shape_contract.py`) regex-scans `codereview/providers/*.py` for an
  inline `startswith("https://")` — `mixins.py` is the only permitted hit, since it's the
  definition site.
- `test_validate_accepts_any_url_the_constructor_accepted` builds each URL-taking provider with
  an uppercase-scheme URL and asserts `--validate` agrees with `__init__`. Azure is excluded
  from the latter only because `HttpUrl` normalizes the scheme before the provider ever sees it.

If a new provider takes a URL, add a `_cleartext_<name>` builder to `_CLEARTEXT_BUILDERS` in
`tests/test_provider_result_shape_contract.py`;
`test_every_url_taking_provider_is_covered_by_the_cleartext_contract` scans the provider modules
for `require_https` call sites and fails if one isn't in the registry, so the coverage can't
silently lapse.
