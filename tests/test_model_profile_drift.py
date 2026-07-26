"""Cross-check ``models.yaml`` against the partner packages' model profiles.

Every LangChain partner package now ships a ``_MODEL_PROFILES`` table (in
``<package>/data/_profiles.py``): a per-model record of context window, output
cap, and capability flags. Our ``models.yaml`` records the same facts by hand,
and the two can disagree in a way nothing else notices — a stale
``context_window`` sizes every token budget wrong for the whole run, and a
``max_output_tokens`` above the model's real cap fails the request outright.

**Neither side is authoritative.** The profile tables are generated from
`models.dev <https://github.com/sst/models.dev>`_, which is community-curated,
not vendor-published — so a disagreement is a *prompt to check*, never grounds
to overwrite the YAML from the profile. And langchain-aws already consumes its
own table at runtime (``_default_params`` drops ``temperature``/``top_p``
whenever ``profile["temperature"] is False``), which is the concrete reason to
care what it says: the installed client acts on it.

**So this is a warn-with-allowlist check.** Our values are deliberately
different in several places, for reasons the profiles cannot know:

* ``supports_tool_use`` is an *empirical* value here — several models advertise
  ``structured_output: true`` and still fail on the forced ``tool_choice`` that
  ``.with_structured_output()`` sets, which is precisely why CLAUDE.md's
  structured-output matrix exists. The profile is the vendor's claim; ours is
  what a live run did.
* A conservative ``max_output_tokens`` is a cost and latency choice, not an
  error. Only a value *above* the profile's cap is a bug.
* A conservative ``context_window`` is safe (smaller batches); one *above* the
  profile's ``max_input_tokens`` overflows.

So the assertions are one-directional where a direction exists, and every
genuine divergence is allowlisted with its reason. The check's value is the
*new* disagreement — the one that appears when a vendor revises a limit or when
someone adds a model with a typo'd context window.

The lookup deliberately uses each package's ``_get_default_model_profile``
(a plain dict lookup) rather than a client's ``.profile`` property: no
credentials, no network, no client construction. It is private API, so
``test_the_profile_lookup_api_still_exists`` fails loudly if it moves rather
than letting every check below silently degrade to "no profile found".
"""

from codereview.config import get_config_loader
from codereview.providers.bedrock import strip_cross_region_prefix

# ---------------------------------------------------------------------------
# Deliberate divergences, keyed (provider, model id, profile field).
#
# Each entry means "we know, and ours is right". Adding one should require the
# same evidence CLAUDE.md asks for elsewhere: a live run, a model card, or a
# documented failure mode.
# ---------------------------------------------------------------------------
_ALLOWED_DIVERGENCES: dict[tuple[str, str, str], str] = {
    # supports_tool_use is empirical here; see CLAUDE.md's structured-output
    # matrix. A forced tool_choice comes back as literal text on these — an
    # observed failure, not a documented API restriction: Anthropic limits
    # tool_choice to auto/none only under manual thinking.type: "enabled", and
    # documents forced tool use as supported with adaptive thinking.
    ("bedrock", "sonnet5", "structured_output"): (
        "first Sonnet tier with adaptive thinking on by default — inherits the "
        "Opus 4.8 literal-text failure under a forced tool_choice"
    ),
    ("bedrock", "kimi-k2.5-bedrock", "structured_output"): (
        "Bedrock Converse leaks Moonshot tool-call markers "
        "(<|tool_call_begin|>…) into text instead of parsing as tool_use"
    ),
    ("bedrock", "glm5-bedrock", "structured_output"): (
        "thinking on by default (reasoning_effort=max); forced tool_choice "
        "auto-downgraded or returned as text"
    ),
    ("azure_openai", "gpt-5.4-pro", "structured_output"): (
        "the profile is conservative and the Azure deployment does tolerate a "
        "forced tool_choice — live-verified on the tool-use path, unlike the "
        "Bedrock OpenAI-compatible endpoint for the same family"
    ),
    # Output/context caps above the profile's, all on Bedrock's third-party
    # re-hosts. Kept as-is rather than lowered from the profile: models.dev is
    # community-curated, and three of these four profile numbers are exactly
    # half the input window (196608/2 = 98304, 202752/2 = 101376, 131072*2 =
    # 262144), which is a derivation rather than an observation — nine of the
    # 110 Bedrock profiles have that exact in/out ratio. Lowering
    # max_output_tokens on a hunch truncates long reviews mid-report, which is
    # silent; being too high fails loudly on the first batch and is then easy
    # to fix. Revisit each one when a live run says otherwise.
    ("bedrock", "kimi-k2.5-bedrock", "max_output_tokens"): (
        "profile says 16000, model card says 65536; unverified live, and "
        "lowering it would silently truncate long reviews"
    ),
    ("bedrock", "minimax-m2.5-bedrock", "max_output_tokens"): (
        "profile says 98304, exactly half its 196608 input window; the YAML's "
        "128000 comes from MiniMax's own card. Unverified live"
    ),
    ("bedrock", "glm5-bedrock", "max_output_tokens"): (
        "profile says 101376, exactly half its 202752 input window; the YAML's "
        "128000 comes from Zhipu's card. Unverified live"
    ),
    ("bedrock", "qwen-next-bedrock", "max_input_tokens"): (
        "profile says 131072, Alibaba's card says 262144 for Qwen3-Coder-Next; "
        "unverified live. Over-claiming context only over-packs a batch, which "
        "the retry path reports as a validation error naming the real limit"
    ),
}


def _profile_lookups():
    """Map provider name → that package's profile lookup function.

    Imported inside the function: these pull in every vendor's LangChain
    client, which is exactly what ``providers/factory.py`` keeps lazy.
    """
    from langchain_aws.chat_models.bedrock_converse import (
        _get_default_model_profile as aws,
    )
    from langchain_deepseek.chat_models import _get_default_model_profile as deepseek
    from langchain_google_genai.chat_models import _get_default_model_profile as google
    from langchain_moonshot.chat_models.base import (
        _get_default_model_profile as moonshot,
    )
    from langchain_nvidia_ai_endpoints.chat_models import (
        _get_default_model_profile as nvidia,
    )
    from langchain_openai.chat_models.base import _get_default_model_profile as openai

    # zai and bedrock_openai are ChatOpenAI-based, so they read OpenAI's table
    # — which is why neither of them ever hits (their wire ids are `glm-5.2`
    # and `openai.gpt-5.6-sol`, not names OpenAI's table carries).
    return {
        "bedrock": aws,
        "azure_openai": openai,
        "nvidia": nvidia,
        "google_genai": google,
        "deepseek": deepseek,
        "zai": openai,
        "moonshot": moonshot,
        "bedrock_openai": openai,
    }


def _registry_rows():
    """Yield ``(provider, model_config, profile)`` for every configured model.

    ``profile`` is ``{}`` when the package has no entry for that wire id, which
    is the common case for re-hosted models (NVIDIA's whole catalog, Z.AI,
    Bedrock's OpenAI-compatible endpoint). A missing profile is not a failure —
    there is simply nothing to cross-check.
    """
    lookups = _profile_lookups()
    loader = get_config_loader()
    for provider, models in loader.list_models().items():
        lookup = lookups[provider]
        for model in models:
            wire_id = model.full_id or model.id
            profile = (
                lookup(strip_cross_region_prefix(wire_id)) or lookup(wire_id) or {}
            )
            yield provider, model, profile


def _rows_with_profiles():
    return [(p, m, prof) for p, m, prof in _registry_rows() if prof]


def test_the_profile_lookup_api_still_exists():
    """Every partner package must still expose the lookup this file uses.

    ``_get_default_model_profile`` is private. If it is renamed or moved, every
    other test here would find no profiles and pass vacuously — the exact
    failure mode the retry-classifier and token-usage contracts were written to
    prevent. Assert the API *and* that it actually resolves something.
    """
    lookups = _profile_lookups()
    assert set(lookups) == set(get_config_loader().list_models()), (
        "a provider was added or removed without updating _profile_lookups"
    )
    for provider, lookup in lookups.items():
        assert callable(lookup), f"{provider}: profile lookup is not callable"

    rows = _rows_with_profiles()
    assert rows, (
        "no model in models.yaml resolved to a partner-package profile. The "
        "lookup API or the wire-id normalization changed; this file's checks "
        "are all vacuous until it's fixed."
    )


def test_every_provider_is_covered_by_a_profile_lookup():
    """A new provider must be wired in here, not silently skipped."""
    from codereview.providers.factory import _PROVIDER_REGISTRY

    assert set(_profile_lookups()) == set(_PROVIDER_REGISTRY), (
        "_profile_lookups and _PROVIDER_REGISTRY disagree — a new provider "
        "needs its package's _get_default_model_profile added above"
    )


def test_no_context_window_exceeds_the_profiles_input_limit():
    """A too-large ``context_window`` over-packs every batch for the whole run.

    One-directional on purpose: a *smaller* context_window than the profile is
    a safe, deliberate choice (and several entries round 1048576 down to
    1000000). Only exceeding the real limit is a bug.
    """
    offenders = []
    for provider, model, profile in _rows_with_profiles():
        limit = profile.get("max_input_tokens")
        ours = model.context_window
        if not limit or not ours or ours <= limit:
            continue
        if (provider, model.id, "max_input_tokens") in _ALLOWED_DIVERGENCES:
            continue
        offenders.append(f"{provider}/{model.id}: ours={ours:,} profile={limit:,}")

    assert not offenders, (
        "context_window above the profile's max_input_tokens:\n  "
        + "\n  ".join(offenders)
        + "\nEither lower it or allowlist it in _ALLOWED_DIVERGENCES with the "
        "evidence that ours is right."
    )


def test_no_max_output_tokens_exceeds_the_profiles_output_limit():
    """A ``max_output_tokens`` above the real cap is an HTTP 400 on batch 1."""
    offenders = []
    for provider, model, profile in _rows_with_profiles():
        limit = profile.get("max_output_tokens")
        params = model.inference_params
        ours = params.max_output_tokens if params else None
        if not limit or not ours or ours <= limit:
            continue
        if (provider, model.id, "max_output_tokens") in _ALLOWED_DIVERGENCES:
            continue
        offenders.append(f"{provider}/{model.id}: ours={ours:,} profile={limit:,}")

    assert not offenders, (
        "max_output_tokens above the profile's cap:\n  "
        + "\n  ".join(offenders)
        + "\nEither lower it or allowlist it in _ALLOWED_DIVERGENCES."
    )


def test_supports_tool_use_disagreements_are_all_deliberate():
    """Our ``supports_tool_use`` is empirical; every disagreement needs a reason.

    Both directions matter here, and they fail differently:

    * ours ``False`` / profile ``True`` — we route around a *live* failure the
      vendor's table doesn't record (thinking + forced ``tool_choice``, leaked
      tool-call markers, fenced JSON). Cheap: prompt parsing works either way.
    * ours ``True`` / profile ``False`` — we're claiming a capability the vendor
      doesn't. That one breaks batches, so it needs a live run behind it.
    """
    undocumented = []
    for provider, model, profile in _rows_with_profiles():
        claimed = profile.get("structured_output")
        if claimed is None or claimed == model.supports_tool_use:
            continue
        if (provider, model.id, "structured_output") in _ALLOWED_DIVERGENCES:
            continue
        undocumented.append(
            f"{provider}/{model.id}: supports_tool_use={model.supports_tool_use} "
            f"profile structured_output={claimed}"
        )

    assert not undocumented, (
        "undocumented supports_tool_use disagreement(s):\n  "
        + "\n  ".join(undocumented)
        + "\nIf ours is right, allowlist it in _ALLOWED_DIVERGENCES with the "
        "failure mode (and add a row to CLAUDE.md's structured-output matrix). "
        "If the profile is right, flip the YAML."
    )


def test_models_whose_profile_rejects_sampling_params_ship_none():
    """``temperature: false`` in a profile means the model rejects the param.

    CLAUDE.md's rule is to omit ``default_temperature``/``default_top_p``/
    ``default_top_k`` for a reasoning model; the profiles are an independent
    source for which models those are, and langchain-aws *acts* on this one —
    ``_default_params`` drops ``temperature`` and ``top_p`` with a warning when
    the profile says they're unsupported, so setting them is at best noise.
    ``top_k`` is not dropped for us: on Bedrock it rides
    ``additional_model_request_fields`` and reaches the API unfiltered, so it
    fails the request.
    """
    offenders = []
    for provider, model, profile in _rows_with_profiles():
        if profile.get("temperature") is not False:
            continue
        params = model.inference_params
        if params is None:
            continue
        for field in ("temperature", "top_p", "top_k"):
            value = getattr(params, field)
            if value is None:
                continue
            if (provider, model.id, field) in _ALLOWED_DIVERGENCES:
                continue
            offenders.append(f"{provider}/{model.id}: default_{field}={value}")

    assert not offenders, (
        "sampling param set on a model whose profile says sampling params are "
        "unsupported:\n  " + "\n  ".join(offenders)
    )


def test_allowlist_has_no_stale_entries():
    """An allowlisted divergence that no longer exists must be removed.

    Otherwise the allowlist accumulates permission for problems that were
    already fixed, and the next real one hides behind a familiar-looking entry.
    """
    live: set[tuple[str, str, str]] = set()
    for provider, model, profile in _rows_with_profiles():
        params = model.inference_params
        ours_out = params.max_output_tokens if params else None
        limit_in, limit_out = (
            profile.get("max_input_tokens"),
            profile.get("max_output_tokens"),
        )
        if limit_in and model.context_window and model.context_window > limit_in:
            live.add((provider, model.id, "max_input_tokens"))
        if limit_out and ours_out and ours_out > limit_out:
            live.add((provider, model.id, "max_output_tokens"))
        claimed = profile.get("structured_output")
        if claimed is not None and claimed != model.supports_tool_use:
            live.add((provider, model.id, "structured_output"))
        if profile.get("temperature") is False and params:
            for field in ("temperature", "top_p", "top_k"):
                if getattr(params, field) is not None:
                    live.add((provider, model.id, field))

    stale = sorted(set(_ALLOWED_DIVERGENCES) - live)
    assert not stale, (
        "_ALLOWED_DIVERGENCES entries that no longer diverge (the YAML, the "
        f"profile, or the model was changed): {stale}"
    )


# Providers where at least one model resolves a profile today. Pinned per
# provider rather than as a total count: the total moves whenever a vendor ships
# a model (which must not fail the suite), but a provider dropping from "some
# coverage" to "none" means its lookup or its wire-id spelling broke, and every
# check above then skips it in silence.
#
# The empty ones are empty for a reason, not by oversight:
#   nvidia         — NIM's re-host ids (`z-ai/glm-5.2`, `moonshotai/kimi-k2.6`)
#                    aren't in the table; it carries older NIM models only
#   moonshot       — the table has kimi-k2.5, not our kimi-k2.6
#   zai            — GLM isn't in langchain-openai's table at all
#   bedrock_openai — its ids are `openai.gpt-5.5` / `xai.grok-4.3`; see
#                    test_rehosted_ids_are_not_mapped_onto_direct_api_profiles
_PROVIDERS_WITH_PROFILE_COVERAGE = {
    "bedrock",
    "azure_openai",
    "google_genai",
    "deepseek",
}


def test_profile_coverage_has_not_collapsed():
    """Each provider that resolves profiles today must still resolve one.

    Half the registry has no profile entry, and that is expected — but a
    provider losing *all* of its coverage is indistinguishable, from the checks
    above, from a clean bill of health.
    """
    covered = {provider for provider, _, _ in _rows_with_profiles()}
    lost = _PROVIDERS_WITH_PROFILE_COVERAGE - covered
    assert not lost, (
        f"these providers no longer resolve any model profile: {sorted(lost)}. "
        "Their package's table, its key spelling, or our full_id changed — "
        "every check in this file is now skipping them silently."
    )
    gained = covered - _PROVIDERS_WITH_PROFILE_COVERAGE
    assert not gained, (
        f"these providers newly resolve profiles: {sorted(gained)}. That is "
        "good — add them to _PROVIDERS_WITH_PROFILE_COVERAGE, and check the "
        "new comparisons are against the right endpoint's limits."
    )


def test_rehosted_ids_are_not_mapped_onto_direct_api_profiles():
    """A re-host's limits are its own; don't borrow the direct API's profile.

    Tempting, because ``openai.gpt-5.5`` → ``gpt-5.5`` would light up three
    more rows. It would also be wrong: the profile says ``max_input_tokens:
    1050000`` for GPT-5.5, while the same model on Bedrock's ``bedrock-mantle``
    endpoint gives 400K — so the comparison would pass a ``context_window``
    over twice the real limit, which is the exact failure this file exists to
    catch. Only ``strip_cross_region_prefix`` is applied, and only because a
    Bedrock inference-profile prefix names the *same* endpoint (langchain-aws's
    own table carries both spellings with identical limits).
    """
    from langchain_openai.chat_models.base import _get_default_model_profile

    loader = get_config_loader()
    rehosted = {m.id: m for m in loader.list_models()["bedrock_openai"]}
    gpt55 = rehosted["gpt5.5-bedrock"]
    direct = _get_default_model_profile("gpt-5.5")

    assert direct.get("max_input_tokens", 0) > (gpt55.context_window or 0), (
        "GPT-5.5's direct-API profile no longer over-states the Bedrock "
        "endpoint's window; re-check whether that mapping is now safe"
    )
    assert not _get_default_model_profile(gpt55.full_id), (
        f"{gpt55.full_id} now resolves a profile directly — verify its limits "
        "describe the bedrock-mantle endpoint before trusting the comparison"
    )
