# tests/test_config.py
"""Tests for configuration management."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from codereview.config import (
    DEFAULT_EXCLUDE_EXTENSIONS,
    DEFAULT_EXCLUDE_PATTERNS,
    MODEL_ALIASES,
    SYSTEM_PROMPT,
    ConfigLoader,
)


def test_default_exclude_patterns():
    """Test default exclusion patterns exist."""
    assert "**/node_modules/**" in DEFAULT_EXCLUDE_PATTERNS
    assert "**/.venv/**" in DEFAULT_EXCLUDE_PATTERNS
    assert "**/__pycache__/**" in DEFAULT_EXCLUDE_PATTERNS


def test_default_exclude_extensions():
    """Test default excluded file extensions."""
    assert ".json" in DEFAULT_EXCLUDE_EXTENSIONS
    assert ".pyc" in DEFAULT_EXCLUDE_EXTENSIONS


def test_config_loader_default_model():
    """Test ConfigLoader loads default model configuration."""
    loader = ConfigLoader()
    provider, model_config = loader.resolve_model("opus")
    assert provider == "bedrock"
    assert model_config.name == "Claude Opus 5"
    assert model_config.pricing.input_per_million > 0


def test_system_prompt_exists():
    """Test system prompt is defined."""
    assert len(SYSTEM_PROMPT) > 0
    assert "code reviewer" in SYSTEM_PROMPT.lower()
    assert "avoid" in SYSTEM_PROMPT.lower()


def test_model_aliases_exist():
    """Test model aliases are defined."""
    assert "opus" in MODEL_ALIASES
    assert "sonnet" in MODEL_ALIASES
    assert "haiku" in MODEL_ALIASES
    assert "minimax-m3-nvidia" in MODEL_ALIASES
    assert "kimi" in MODEL_ALIASES
    assert "glm" in MODEL_ALIASES


def test_resolve_model_id_with_alias():
    """Test resolving short model names to full IDs via ConfigLoader."""
    loader = ConfigLoader()
    provider, model_config = loader.resolve_model("opus")
    assert model_config.full_id == "us.anthropic.claude-opus-5"

    # `sonnet` moved from Sonnet 4.6 to Sonnet 5 when the 4.6 entry was removed
    # (2026-08-29) — generation-neutral names track the current generation.
    provider, model_config = loader.resolve_model("sonnet")
    assert model_config.full_id == "us.anthropic.claude-sonnet-5"

    provider, model_config = loader.resolve_model("haiku")
    assert model_config.full_id == "global.anthropic.claude-haiku-4-5-20251001-v1:0"

    provider, model_config = loader.resolve_model("glm")
    assert model_config.full_id == "glm-5.2"


def test_resolve_model_id_case_insensitive():
    """Test model name resolution handles aliases case-insensitively."""
    loader = ConfigLoader()
    # Aliases in YAML are lowercase, so we test that lowercase works
    provider1, model1 = loader.resolve_model("opus")
    provider2, model2 = loader.resolve_model("sonnet")
    assert model1.name == "Claude Opus 5"
    assert model2.name == "Claude Sonnet 5"


def test_resolve_model_id_with_full_id():
    """Test resolving with full model ID works."""
    loader = ConfigLoader()
    # Short ID (which is used in the YAML as the primary ID)
    provider, model_config = loader.resolve_model("opus5")
    assert model_config.id == "opus5"


def test_all_aliases_map_to_valid_models():
    """Test all aliases map to valid models in ConfigLoader."""
    loader = ConfigLoader()
    for alias in MODEL_ALIASES.keys():
        # Should not raise ValueError
        provider, model_config = loader.resolve_model(alias)
        assert model_config is not None
        assert model_config.name is not None


def test_fable5_pinned_to_us_east_1():
    """fable5 requires the per-region provider_data_share opt-in, which this
    account (and the geo-US profile generally) has in us-east-1 only — the
    model entry must pin region us-east-1 or invocation fails with
    ValidationException: data retention mode 'default' is not available."""
    loader = ConfigLoader()
    provider, model_config = loader.resolve_model("fable5")
    assert provider == "bedrock"
    assert model_config.region == "us-east-1"


def test_fable5_read_timeout_covers_thinking_latency():
    """fable5's adaptive thinking is always on and can't be disabled, and the
    Converse call is non-streaming — think-heavy batches exceed the 300s
    provider-default read_timeout (observed: ReadTimeoutError at 5+ minutes).
    The model entry must carry a read_timeout well above that."""
    loader = ConfigLoader()
    _, model_config = loader.resolve_model("fable5")
    assert model_config.read_timeout is not None
    assert model_config.read_timeout >= 1800


def test_opus5_read_timeout_covers_thinking_latency():
    """opus5 has thinking ON by default (a breaking change from Opus 4.8, where
    it was off unless requested) at default effort "high", and the Converse call
    is non-streaming — no bytes arrive until the full response is generated, so
    think-heavy batches would outlast the 300s provider-default read_timeout.
    Same condition that forced fable5's override."""
    loader = ConfigLoader()
    _, model_config = loader.resolve_model("opus5")
    assert model_config.read_timeout is not None
    assert model_config.read_timeout >= 1800


def test_opus5_context_and_output_match_bedrock_card():
    """Opus 5 advertises 1M context (both default and maximum) / 128K output."""
    loader = ConfigLoader()
    _, config = loader.resolve_model("opus5")
    assert config.context_window == 1_000_000
    assert config.inference_params is not None
    assert config.inference_params.max_output_tokens == 128_000


def test_opus5_omits_sampling_params():
    """Opus 5 is a reasoning model — temperature/top_p/top_k are unsupported.

    The Bedrock provider passes ``allow_none=True`` to ``_resolve_temperature``,
    so an absent ``default_temperature`` in the YAML (loaded into the
    ``temperature`` field) is what opts the model out of sending
    ``temperature`` on the Converse call.
    """
    loader = ConfigLoader()
    _, config = loader.resolve_model("opus5")
    assert config.inference_params is not None
    assert config.inference_params.temperature is None
    assert config.inference_params.top_p is None
    assert config.inference_params.top_k is None


def test_generation_neutral_opus_alias_tracks_opus5():
    """The bare Opus aliases must resolve to the newest Opus entry.

    ``_register_model`` last-write-wins within a single provider (it only warns
    across providers), so a stale entry keeping ``opus`` as its ``id`` would
    silently shadow this alias depending on YAML order.
    """
    loader = ConfigLoader()
    for alias in ("opus", "claude-opus", "claude-opus-5", "opus-5"):
        _, config = loader.resolve_model(alias)
        assert config.id == "opus5", f"{alias!r} resolved to {config.id!r}"


def test_superseded_opus_generation_aliases_are_gone():
    """``--model opus4.6`` must fail loudly, not resolve to Opus 5.

    The 2026-07-25 cleanup initially migrated the removed Opus 4.7/4.6 entries'
    aliases onto Opus 5 to keep scripted invocations working. That turned out to
    be the wrong trade: a name that says "4.6" silently getting a
    two-generations-newer model with different pricing, different sampling-param
    support and a different structured-output path is worse than an error a
    human reads and fixes. They were deleted instead — this guards against a
    well-meaning re-migration.
    """
    loader = ConfigLoader()
    for alias in (
        "opus4.7",
        "opus-4.7",
        "claude-opus-4.7",
        "claude-opus-47",
        "opus4.6",
        "opus-4.6",
        "claude-opus-4.6",
    ):
        with pytest.raises(ValueError, match="Unknown model"):
            loader.resolve_model(alias)


def test_model_id_conflict_detection(caplog):
    """Test that model ID conflicts are detected and logged."""
    import logging

    from codereview.config.models import ModelConfig, PricingConfig

    loader = ConfigLoader()

    # Simulate registering same ID from different provider
    mock_config = ModelConfig(
        id="opus",  # Already registered by bedrock (as an opus5 alias)
        name="Fake Opus",
        aliases=[],
        pricing=PricingConfig(input_per_million=1.0, output_per_million=1.0),
    )

    with caplog.at_level(logging.WARNING):
        loader._register_model("fake_provider", mock_config, "opus")

    # Should warn about conflict
    assert "Model name conflict" in caplog.text
    assert "bedrock" in caplog.text
    assert "fake_provider" in caplog.text

    # Original should still be registered (first wins)
    provider, config = loader.resolve_model("opus")
    assert provider == "bedrock"
    assert config.name == "Claude Opus 5"


def test_same_provider_model_name_conflict_is_warned(caplog):
    """Two entries under ONE provider claiming a name must warn, not go silent.

    Intra-provider registration is deliberately last-write-wins (CLAUDE.md: it
    is what lets a generation-neutral alias move to a newer entry further down
    the YAML). The defect was that this case logged *nothing*: the conflict
    check only fired when the providers differed, so two bedrock entries sharing
    an alias silently made the earlier entry unreachable under that name — the
    exact trap the generation-neutral-alias convention warns about.

    Resolution behavior is unchanged; only the warning is new.
    """
    import logging

    from codereview.config.models import ModelConfig, PricingConfig

    loader = ConfigLoader()

    shadowing = ModelConfig(
        id="some-other-bedrock-entry",
        name="Shadowing Entry",
        aliases=[],
        pricing=PricingConfig(input_per_million=1.0, output_per_million=1.0),
    )

    with caplog.at_level(logging.WARNING):
        loader._register_model("bedrock", shadowing, "opus")

    assert "Model name conflict" in caplog.text
    assert "some-other-bedrock-entry" in caplog.text
    # Names both sides so the message is actionable.
    assert "opus5" in caplog.text or "Claude Opus 5" in caplog.text

    # Documented last-write-wins semantics preserved.
    provider, config = loader.resolve_model("opus")
    assert provider == "bedrock"
    assert config.id == "some-other-bedrock-entry"


def test_reregistering_the_same_entry_is_not_a_conflict(caplog):
    """_register_all_names is idempotent; re-registering one entry must be quiet."""
    import logging

    loader = ConfigLoader()
    _, existing = loader.resolve_model("opus5")

    with caplog.at_level(logging.WARNING):
        loader._register_model("bedrock", existing, "opus5")

    assert "Model name conflict" not in caplog.text


def test_real_registry_loads_without_any_conflict_warning(caplog):
    """models.yaml itself must not trip either conflict branch.

    With the same-provider branch now warning, a duplicate alias inside one
    provider block becomes visible at load time instead of silently shadowing.
    """
    import logging

    from codereview.config import get_config_loader

    get_config_loader.cache_clear()
    with caplog.at_level(logging.WARNING):
        ConfigLoader()

    assert "Model name conflict" not in caplog.text, (
        "models.yaml has a duplicate model id/alias: " + caplog.text
    )


# ---------------------------------------------------------------------------
# Upstream-currency guards
# ---------------------------------------------------------------------------

# Upstream endpoints no registry entry may target: either retired/shut down by
# their provider, or unreachable from the region/resource this project is
# configured for. Audited 2026-05-30 and re-audited 2026-07-25 by probing every
# entry against its live provider endpoint. The entries that pointed here were
# removed and their aliases redirected to live successors; this guard fails if
# a dead full_id is ever reintroduced (e.g. by copy-paste from an old entry).
#
# Superseded-but-still-live endpoints are deliberately NOT listed — re-adding
# those is a judgement call, not a bug. Their aliases are covered by
# test_retired_model_aliases_redirect_to_live_successors instead.
#   minimaxai/minimax-m2.5              — NIM deprecated 2026-05-12
#   moonshotai/kimi-k2.5                — NIM shut down 2026-05-20 (NOTE: the
#                                         dotted Bedrock id is a different
#                                         endpoint and is still live)
#   z-ai/glm5                           — NIM deprecated 2026-04-20
#   z-ai/glm-5.1                        — NIM deprecated ~2026-07
#   gemini-3-pro-preview                — Google shut down 2026-03-09
#   qwen/qwen3-coder-480b-a35b-instruct — NIM endpoint returns 404 (2026-07-25)
#   qwen.qwen3-coder-480b-a35b-v1:0     — Bedrock us-west-2 only; the provider's
#                                         configured region does not offer it
#   Kimi-K2.5 / DeepSeek-V4-Pro (Azure) — DeploymentNotFound on this resource
#                                         (deployment_name, not full_id — see
#                                         DEAD_AZURE_DEPLOYMENT_NAMES below)
#   deepseek-ai/deepseek-v4-pro         — the undated *preview* ids. NIM
#   deepseek-ai/deepseek-v4-flash         end-of-lifed both 2026-08-07 and now
#                                         answers HTTP 410 Gone with an explicit
#                                         end-of-life message (probed 2026-08-29).
#                                         The dated GA ids `-0813` / `-0731` are
#                                         live and the two NVIDIA entries were
#                                         re-pointed onto them, keeping their
#                                         generation-neutral ids and aliases.
#                                         Listed here because an *undated* id
#                                         reads like the current one and is the
#                                         exact thing a copy-paste reintroduces.
#
# Re-audited 2026-08-29 against NIM's live catalog and a completion per entry.
# Five entries were removed in that pass; unlike a "superseded" call these are
# not judgement calls, because NIM answers HTTP 410 and states its own EOL date:
#   mistralai/mistral-small-4-119b-2603 — EOL 2026-07-27; no Mistral remains on
#                                         NIM, so every Mistral alias was deleted
#   mistralai/mistral-medium-3.5-128b   — EOL 2026-08-07 (same)
#   qwen/qwen3.5-397b-a17b              — EOL 2026-07-27; the catalog now returns
#                                         zero `qwen/*` ids at all
#   z-ai/glm-5.2                        — EOL 2026-08-21; zero `z-ai/*` ids left,
#                                         closing out that lineage after glm5 and
#                                         glm-5.1 above
#   stepfun-ai/step-3.7-flash           — EOL 2026-08-28; zero `stepfun*` ids left
#
# A SECOND pass the same day removed nine more entries and deliberately added
# NOTHING here, because every one of those endpoints answered when probed. That
# distinction is the whole point of this set: it means "do not point at this wire
# id again", not "we stopped shipping this model". The nine, all still live:
#   us.anthropic.claude-opus-4-8, global.anthropic.claude-sonnet-4-6,
#   moonshotai.kimi-k2.5, qwen.qwen3-coder-next, minimax.minimax-m2.5 (Bedrock,
#   ListFoundationModels), gemini-3.6-flash (Google), openai.gpt-5.5 and
#   xai.grok-4.3 (bedrock-mantle, HTTP 200), and moonshotai/kimi-k2.6 (listed by
#   NIM's GET /v1/models; its HTTP 404 names the *account*, i.e. provisioning
#   scope, not an EOL — contrast the 410-with-EOL-date entries above).
# Note the near-collisions already in this set: `moonshotai/kimi-k2.5` and
# `minimaxai/minimax-m2.5` are the NVIDIA re-hosts (slash), which really are
# dead; their Bedrock namesakes use a dot and must stay out.
DEAD_UPSTREAM_FULL_IDS = {
    "minimaxai/minimax-m2.5",
    "moonshotai/kimi-k2.5",
    "z-ai/glm5",
    "z-ai/glm-5.1",
    "z-ai/glm-5.2",
    "gemini-3-pro-preview",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen.qwen3-coder-480b-a35b-v1:0",
    "qwen/qwen3.5-397b-a17b",
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
    "mistralai/mistral-small-4-119b-2603",
    "mistralai/mistral-medium-3.5-128b",
    "stepfun-ai/step-3.7-flash",
}

# Azure entries are addressed by deployment_name, not full_id, and only work if
# a deployment with that exact name exists on the resource. Both of these
# returned DeploymentNotFound when probed 2026-07-25.
DEAD_AZURE_DEPLOYMENT_NAMES = {
    "Kimi-K2.5",
    "DeepSeek-V4-Pro",
}


def test_no_model_points_at_dead_upstream_endpoint():
    """No registry entry may target a known-retired upstream endpoint."""
    loader = ConfigLoader()
    offenders = {
        model_id: config.full_id
        for model_id, (_, config) in loader._models_by_id.items()
        if config.full_id in DEAD_UPSTREAM_FULL_IDS
    }
    assert not offenders, f"Entries point at retired endpoints: {offenders}"


def test_no_model_points_at_missing_azure_deployment():
    """No Azure entry may name a deployment that doesn't exist on the resource.

    Unlike Bedrock/NVIDIA catalog models, an Azure entry is only usable if
    someone created a deployment with that exact name — a stale one fails at
    invocation time with DeploymentNotFound rather than at ``--list-models``.
    """
    loader = ConfigLoader()
    offenders = {
        model_id: config.deployment_name
        for model_id, (_, config) in loader._models_by_id.items()
        if config.deployment_name in DEAD_AZURE_DEPLOYMENT_NAMES
    }
    assert not offenders, f"Entries name missing Azure deployments: {offenders}"


def test_retired_model_aliases_redirect_to_live_successors():
    """Aliases inherited from removed entries resolve to a live successor.

    Removing a model should not break a scripted ``--model <alias>`` when the
    successor is a drop-in: the successor absorbs the alias. The exception is a
    name that states a *version* (``opus4.6``, ``minimax-m2.5``, ``glm-5.1``) —
    those were deleted in the 2026-07-25 alias cleanup rather than redirected,
    because silently serving a different generation is worse than a clear error.
    ``RETIRED_ALIASES_DELETED_NOT_REDIRECTED`` below is the counterpart guard.
    """
    loader = ConfigLoader()
    expected = {
        # Kimi/DeepSeek-on-Azure: both deployments are gone from the resource,
        # and the direct APIs are the canonical owners of those families. These
        # names don't state a version, so redirecting is safe.
        "kimi-azure": "kimi-k2.6",
        "kimi25-azure": "kimi-k2.6",
        "deepseek-v4-azure": "deepseek-v4-pro",
        "ds-v4-azure": "deepseek-v4-pro",
        # GLM-on-NVIDIA is entirely gone (glm5 deprecated 2026-04-20, glm-5.1
        # ~2026-07, glm-5.2 EOL 2026-08-21 — NIM serves no `z-ai/*` id now), so
        # 2026-08-29 moved these onto Bedrock's `zai.glm-5`. That is the model
        # they actually name: the spellings say GLM *5*. The NVIDIA-suffixed
        # `glm5-nvidia` was deleted instead — see the counterpart guard.
        "glm5": "zai.glm-5",
        "glm-5": "zai.glm-5",
        # GLM-on-Z.AI: 5.1 removed in favour of 5.2 (same price, 1M context).
        "zai-glm": "glm-5.2",
        "glm-zai": "glm-5.2",
        # Gemini: 3 Pro shut down 2026-03-09; 3 Flash Preview deprecated in
        # favour of the GA Gemini 3.6 Flash, which was itself removed 2026-08-29
        # — so the generation-3 Flash names moved on again, to 3.7 Flash. They
        # stay in this map because 3.7 IS a generation-3 Flash at the same
        # $1.50/$7.50, same 1M context and same 64K output; the *minor*-version
        # spellings (gemini36-flash, gemini3.6-flash) were deleted instead.
        "gemini-3-pro": "gemini-3.1-pro-preview",
        "gemini3-pro": "gemini-3.1-pro-preview",
        "gemini-3-flash": "gemini-3.7-flash",
        "gemini3-flash": "gemini-3.7-flash",
        "g3flash": "gemini-3.7-flash",
        # ---- 2026-08-29 curation pass (nine live entries removed) ----
        # Claude Sonnet 4.6 removed in favour of Sonnet 5 (same $3/$15, 5x the
        # context). `sonnet` was the removed entry's *id* and names the tier, not
        # a version, so it moves up — and as a plain `alias`, not a deprecated
        # one, because sonnet5 genuinely is the current Sonnet.
        "sonnet": "us.anthropic.claude-sonnet-5",
        "claude-sonnet": "us.anthropic.claude-sonnet-5",
        # Kimi K2.5 on *Bedrock* removed; the Moonshot direct API is the family's
        # canonical owner and ships the newer K2.6 at a lower output price, so
        # this is the same move `kimi-azure` made. Crosses a provider boundary,
        # hence deprecated rather than advertised.
        "kimi-bedrock": "kimi-k2.6",
        # GPT-5.5 on Bedrock removed; GPT-5.6 Sol is the only OpenAI entry left
        # on bedrock-mantle. `gpt-bedrock` names "the GPT on Bedrock", which Sol
        # now is — but following it forward DOUBLES the rate ($2.50/$15 ->
        # $5/$30), which is why it stays a deprecated_alias.
        "gpt-bedrock": "openai.gpt-5.6-sol",
        # Deliberately NOT in this map, though the convention's default would put
        # them here: every `qwen*` and `grok*` spelling. Both families left the
        # registry entirely in the same pass, so there is no successor that is
        # honestly the same thing — see the counterpart guard.
        #
        # A removed entry's *id* is a --model spelling too, not just its
        # aliases — these were ids of removed entries and are easy to forget.
        "deepseek-v4-pro-azure": "deepseek-v4-pro",
        "kimi-k2.5-azure": "kimi-k2.6",
    }
    for alias, live_full_id in expected.items():
        _, config = loader.resolve_model(alias)
        assert config.full_id == live_full_id, (
            f"alias {alias!r} resolved to {config.full_id!r}, expected {live_full_id!r}"
        )


# Identifiers that once shipped and were deliberately DELETED — in the
# 2026-07-25 alias cleanup, the 2026-08-29 dead-endpoint pass, or the 2026-08-29
# curation pass — rather than redirected onto a successor. Each either states a
# specific model version, names a vendor/provider that no longer appears in the
# registry at all, or is a redundant short form. Resolving any of them to a newer
# generation would silently change pricing, sampling-param support and the
# structured-output path, so failing fast is the correct behavior.
#
# Whether the endpoint is still alive upstream does NOT enter into it: the
# 2026-08-29 curation group below is nine live endpoints, and their names are
# deleted on exactly the same rule as the ones NVIDIA end-of-lifed.
#
# This is the allowlist for test_no_historical_model_id_is_orphaned — anything
# NOT listed here must still resolve.
RETIRED_ALIASES_DELETED_NOT_REDIRECTED = frozenset(
    {
        # Opus 4.7 / 4.6 (removed entries) — Opus 5 is two generations newer.
        "opus4.7",
        "opus-4.7",
        "claude-opus-4.7",
        "claude-opus-47",
        "opus4.6",
        "opus-4.6",
        "claude-opus-4.6",
        # MiniMax-on-NVIDIA M2.5 / M2.7 — both NIM endpoints are gone.
        "minimax-m2.5",
        "minimax-m2.5-nvidia",
        "mm2.5-nvidia",
        "mm25",
        "minimax-m2.7",
        "minimax-m2.7-nvidia",
        "mm2.7-nvidia",
        "mm27",
        # Kimi K2.5 on NVIDIA — endpoint shut down 2026-05-20.
        "kimi-k2.5",
        "kimi-k2.5-nvidia",
        "kimi25",
        # GLM-5.1 (both the NVIDIA re-host and the Z.AI entry, whose id it was).
        "glm51",
        "glm51-nvidia",
        "glm-5.1",
        "glm5.1",
        "glm5.1-zai",
        "zhipuai/glm-5.1",
        # ---- 2026-08-29 removal pass: five NIM endpoints answering HTTP 410 ----
        # GLM-5.2 on NVIDIA (EOL 2026-08-21). `glm5`/`glm-5` migrated to
        # `glm5-bedrock`; every NVIDIA-suffixed spelling died with the endpoint,
        # since NIM now serves no GLM at all.
        "glm52",
        "glm52-nvidia",
        "glm5.2-nvidia",
        "glm-5.2-nvidia",
        "glm5-nvidia",
        # Step 3.5 Flash (superseded 2026-07-25) and Step 3.7 Flash (EOL
        # 2026-08-28). `step-flash`/`step-nvidia` are version-neutral and would
        # normally migrate, but no Step model remains in the registry to carry
        # them, so the whole family fails fast.
        "step35",
        "step-3.5-flash",
        "step-3.7-flash",
        "step-3.7",
        "step37",
        "step37-nvidia",
        "step-flash",
        "step-nvidia",
        # Qwen3.5 on NVIDIA (EOL 2026-07-27); NIM serves no `qwen/*` id now. The
        # version-neutral `qwen-nvidia`/`qwen3-nvidia`/`qwen-coder-nvidia` are
        # deleted *on purpose* rather than migrated: the only live Qwen here is
        # `qwen-next-bedrock`, and a `-nvidia` name resolving to Bedrock would
        # move the user from a free endpoint to a billed one on another provider.
        "qwen3.5-nvidia",
        "qwen3.5",
        "qwen35",
        "qwen35-nvidia",
        "qwen-nvidia",
        "qwen3-nvidia",
        "qwen-coder-nvidia",
        # Mistral Small 4 (EOL 2026-07-27) and Mistral Medium 3.5 (EOL
        # 2026-08-07). NIM carries no Mistral successor, so no Mistral model
        # remains in this registry and a Mistral name has nothing honest to
        # resolve to.
        "mistral-small-nvidia",
        "mistral-small",
        "mistral-small-4",
        "ms4",
        "mistral-medium-nvidia",
        "mistral-medium",
        "mistral-medium-3.5",
        # GPT-5.4 on Bedrock — gpt-bedrock still redirects (now to GPT-5.6 Sol).
        "gpt5.4-bedrock",
        # ---- 2026-08-29 curation pass: nine LIVE entries removed ----
        # Unlike every group above, nothing here died upstream. These endpoints
        # all still answer; the entries were cut because a surviving entry
        # covered the same ground. The alias rule does not change for that
        # reason: a name that pins a version still must not answer from a
        # different one. See the models.yaml removal notes for the per-entry
        # evidence, and DEAD_UPSTREAM_FULL_IDS for why none of their wire ids
        # were blacklisted.
        #
        # Opus 4.8 — superseded by opus5 at identical $5/$25, context and output.
        # `opus`/`claude-opus` were already on opus5, so nothing migrated.
        "opus4.8",
        "opus-4.8",
        "claude-opus-4.8",
        "claude-opus-48",
        # Sonnet 4.6 — `sonnet`/`claude-sonnet` migrated to sonnet5 (see the
        # redirect map); only the version-explicit pair dies. Worth knowing what
        # `--model sonnet4.6` used to buy that sonnet5 does not: temperature
        # support and the tool-use structured-output path.
        "sonnet4.6",
        "claude-sonnet-4.6",
        # Kimi K2.5 on Bedrock — `kimi-bedrock` migrated to Moonshot-direct K2.6.
        "kimi-k2.5-bedrock",
        "kimi25-bedrock",
        # Kimi K2.6 on NVIDIA — the surviving NIM Kimi is K3, a different
        # generation, so these do not follow it. Use `kimi-nvidia-3` or `kimi`.
        "kimi-k2.6-nvidia",
        "kimi-nvidia-26",
        "kimi26-nvidia",
        # Qwen3 Coder Next on Bedrock — the LAST Qwen anywhere in the registry,
        # so even the version-neutral `qwen`/`qwen-coder` are deleted rather than
        # migrated (nothing Qwen-shaped remains to migrate onto). `qwen-bedrock`
        # moves here from the redirect map for the same reason. This also removed
        # the cheapest Bedrock entry ($0.50/$1.20) and one of only three
        # tool-use-path Bedrock entries.
        "qwen-next-bedrock",
        "qwen-bedrock",
        "qwen-next",
        "qwen3-next",
        "qwen-coder-next",
        "qwen",
        "qwen-coder",
        # MiniMax M2.5 on Bedrock — superseded within the registry by
        # `minimax-m3` on NIM. Pointing a `-bedrock` name at NVIDIA would cross a
        # provider *and* a billing boundary while claiming to be Bedrock, so it
        # joins the already-deleted mm25/mm2.7 names above.
        "minimax-m2.5-bedrock",
        "mm2.5-bedrock",
        # Gemini 3.6 Flash — 3.7 Flash is identical on price, context and output,
        # so the generation-3 names migrated (see the redirect map); these pin the
        # minor version and do not.
        "gemini-3.6-flash",
        "gemini36-flash",
        "gemini3.6-flash",
        # GPT-5.5 on Bedrock — `gpt-bedrock` migrated to GPT-5.6 Sol.
        "gpt5.5-bedrock",
        # Grok 4.3 on Bedrock — xAI leaves the registry entirely, so even
        # `grok`/`grok-bedrock` are deleted: resolving a Grok name to an OpenAI
        # model would be a vendor swap, not a version bump. Costliest removal of
        # the pass — it was bedrock_openai's cheapest entry ($1.25/$2.50 vs
        # $5/$30), its widest context (1M vs 272K), and its only entry accepting
        # temperature/top_p.
        "grok-4.3-bedrock",
        "grok",
        "grok-4.3",
        "grok43",
        "grok-bedrock",
        # Redundant/cryptic short forms of live models, dropped as noise.
        "gpt54p",
        "glm5b",
        "dsv4f",
        "dsv4pro",
        "dsv4-azure",
        "g31pro",
        "g3pro",
        "g36flash",
        "kimi-moonshot",
        "mm35",
        "mmed",
        "gpt5.6-sol",
        "sol",
    }
)


def test_deprecated_aliases_resolve_but_are_not_advertised():
    """The two lists must differ in display only, never in resolution.

    This is the whole contract of the ``aliases`` / ``deprecated_aliases``
    split. If ``_register_all_names`` ever skipped the deprecated list, every
    back-compat name would break at once while ``--list-models`` looked fine.
    """
    loader = ConfigLoader()
    checked = 0
    for models in loader.list_models().values():
        for config in models:
            for name in config.deprecated_aliases:
                provider, resolved = loader.resolve_model(name)
                assert resolved.id == config.id, (
                    f"deprecated alias {name!r} resolved to {resolved.id!r}, "
                    f"expected {config.id!r}"
                )
                checked += 1
    assert checked, "no deprecated aliases in the registry — is the split wired?"


def test_no_model_lists_its_own_id_as_an_alias():
    """The id is already a valid --model spelling; repeating it is pure noise.

    ``gpt5.5-bedrock`` and ``deepseek-v4-flash-nvidia`` both shipped listing
    their own id, padding the ``--list-models`` Aliases column with a name
    already in the ID column. ``ModelConfig`` now rejects it at load time; this
    asserts the real registry is clean.
    """
    loader = ConfigLoader()
    offenders = [
        config.id
        for models in loader.list_models().values()
        for config in models
        if config.id in (*config.aliases, *config.deprecated_aliases)
    ]
    assert not offenders, f"entries listing their own id as an alias: {offenders}"


def test_model_config_rejects_self_alias():
    """The schema — not just the registry — must reject a self-alias."""
    from codereview.config.models import ModelConfig, PricingConfig

    with pytest.raises(ValidationError, match="its own id"):
        ModelConfig(
            id="dupe",
            name="Dupe",
            aliases=["dupe"],
            pricing=PricingConfig(input_per_million=1.0, output_per_million=1.0),
        )


def test_model_config_rejects_duplicate_alias_across_both_lists():
    """A name in both lists has no defined display answer, so it's an error."""
    from codereview.config.models import ModelConfig, PricingConfig

    with pytest.raises(ValidationError, match="repeats alias"):
        ModelConfig(
            id="m",
            name="M",
            aliases=["shared"],
            deprecated_aliases=["shared"],
            pricing=PricingConfig(input_per_million=1.0, output_per_million=1.0),
        )


def test_deleted_aliases_do_not_resolve():
    """The deleted names must raise, not quietly resolve.

    Complements ``test_retired_model_aliases_redirect_to_live_successors``: that
    one pins what still works, this one pins what deliberately stopped working.
    Without it, re-adding ``mm25`` as an M3 alias would pass every other test.
    """
    loader = ConfigLoader()
    for name in sorted(RETIRED_ALIASES_DELETED_NOT_REDIRECTED):
        with pytest.raises(ValueError, match="Unknown model"):
            loader.resolve_model(name)


def test_no_historical_model_id_is_orphaned():
    """Every id/alias that ever shipped resolves, unless explicitly retired.

    The hand-written table above documents *which* successor each retired name
    maps to; this test is the exhaustive net that catches a name nobody
    remembered to migrate. It reads previous revisions of ``models.yaml``
    straight from git, so it needs no maintenance when entries are removed —
    only that the removal migrates the names, or records them in
    ``RETIRED_ALIASES_DELETED_NOT_REDIRECTED``.

    A removed entry's ``id`` counts: ``--model <id>`` is exactly as valid an
    invocation as ``--model <alias>``, and ids are the ones that get forgotten
    (``glm51``, ``kimi-k2.5-azure``, ``deepseek-v4-pro-azure`` and
    ``zhipuai/glm-5.1`` all shipped orphaned before this test existed).

    Deliberate deletions go in the allowlist — which is the point of having one:
    dropping a name becomes an explicit, reviewable line of code rather than a
    silently weakened test.

    Skips when git history isn't available (e.g. an sdist install).
    """
    repo_root = Path(__file__).resolve().parent.parent
    yaml_rel = "codereview/config/models.yaml"

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    try:
        revs = git("log", "--format=%H", "-8", "--", yaml_rel).split()
    except subprocess.CalledProcessError, FileNotFoundError:
        pytest.skip("git history unavailable")
    if not revs:
        pytest.skip("no history for models.yaml")

    historical: set[str] = set()
    for rev in revs:
        doc = yaml.safe_load(git("show", f"{rev}:{yaml_rel}"))
        for provider_cfg in (doc.get("providers") or {}).values():
            for model in provider_cfg.get("models") or []:
                historical.add(model["id"])
                historical.update(model.get("aliases") or [])
                historical.update(model.get("deprecated_aliases") or [])

    loader = ConfigLoader()
    orphaned = []
    for name in sorted(historical - RETIRED_ALIASES_DELETED_NOT_REDIRECTED):
        try:
            loader.resolve_model(name)
        except ValueError:
            orphaned.append(name)

    assert not orphaned, (
        "These model names shipped previously but no longer resolve — either "
        "migrate each onto a live successor's aliases, or, if dropping them is "
        "intended, add them to RETIRED_ALIASES_DELETED_NOT_REDIRECTED with a "
        f"reason: {orphaned}"
    )

    # The allowlist must stay honest, but "absent from the scanned history" is
    # NOT the check for that: the window is only the last 8 revisions, and a
    # name added and deleted within the same uncommitted change never appears
    # in committed history at all. The check that actually matters — that every
    # allowlisted name really fails to resolve — is
    # test_deleted_aliases_do_not_resolve.


def test_documented_model_names_all_resolve():
    """Every ``--model X`` in the user-facing docs must be a real model.

    The 2026-07-25 alias cleanup deleted 40 names that the README, usage guide
    and examples still advertised as "route here". A doc that tells someone to
    run ``--model mm25`` is worse than no doc: they hit an error on a command we
    published. Removing or renaming an alias now fails here until the prose
    catches up.

    Scoped to `--model <name>` occurrences on purpose — prose *about* a deleted
    alias (the migration table, the removal notes) must keep naming it.
    """
    repo_root = Path(__file__).resolve().parent.parent
    docs = [
        repo_root / "README.md",
        repo_root / "docs" / "usage.md",
        repo_root / "docs" / "examples.md",
    ]
    pattern = re.compile(r"--model\s+([A-Za-z0-9][A-Za-z0-9./-]*)")

    loader = ConfigLoader()
    broken: list[str] = []
    checked = 0
    for doc in docs:
        if not doc.exists():  # pragma: no cover - docs ship with the repo
            continue
        for name in sorted(set(pattern.findall(doc.read_text()))):
            # Placeholders in generic syntax lines, not real model names.
            if name in {"X", "id-or-alias"}:
                continue
            checked += 1
            try:
                loader.resolve_model(name)
            except ValueError:
                broken.append(f"{doc.name}: {name}")

    assert checked, "regex matched no --model examples; the pattern is wrong"
    assert not broken, (
        "Docs advertise --model names that no longer resolve. Either restore "
        "the alias or update the prose to a live spelling: " + ", ".join(broken)
    )


def test_every_pricing_and_inference_key_in_the_yaml_is_actually_read():
    """A key the loader never reads is a silent lie about what the tool does.

    ``PricingConfig`` and ``InferenceParams`` are not ``extra="forbid"``, and
    ``_parse_model_config`` copies fields across **by name, one at a time** — so
    a YAML key nobody reads loads without error, is dropped on the floor, and
    still reads to a human as configuration. Ten such keys shipped:
    ``cache_read_per_million``/``cache_write_per_million`` on six Claude entries
    and ``cached_input_per_million`` on four more, advertising prompt-caching
    rates that could never reach a cost figure. The same shape as the
    ``NVIDIAConfig.max_retries`` bug (CLAUDE.md, ConfigLoader gotcha), one level
    down: present in the YAML, absent from the constructor.

    Scoped to ``pricing`` and ``inference_params`` because those are pure data
    blocks — unlike ``capabilities``/``architecture``/``notes``, which CLAUDE.md
    documents as deliberately informational.

    The expected key set is scraped from ``loader.py`` rather than listed here:
    the YAML spelling differs from the field name (``default_temperature`` →
    ``temperature``), so the loader is the only place the mapping exists, and a
    hand-copied list here would just be a second thing to forget.
    """
    repo_root = Path(__file__).resolve().parent.parent
    loader_src = (repo_root / "codereview" / "config" / "loader.py").read_text()
    read_keys = set(
        re.findall(r'(?:pricing_data|params_data)(?:\.get\(|\[)"([^"]+)"', loader_src)
    )
    assert "input_per_million" in read_keys and "max_output_tokens" in read_keys, (
        "the scrape found no recognisable keys — _parse_model_config was "
        "restructured and this test is now vacuous"
    )

    doc = yaml.safe_load(
        (repo_root / "codereview" / "config" / "models.yaml").read_text()
    )
    unread: list[str] = []
    for provider, provider_cfg in (doc.get("providers") or {}).items():
        for model in provider_cfg.get("models") or []:
            for block in ("pricing", "inference_params"):
                for key in model.get(block) or {}:
                    if key not in read_keys:
                        unread.append(f"{provider}/{model['id']}: {block}.{key}")

    assert not unread, (
        "models.yaml sets keys that ConfigLoader never reads, so they affect "
        "nothing while looking like they do:\n  "
        + "\n  ".join(unread)
        + "\nEither wire the key through _parse_model_config (and the Pydantic "
        "model) or delete it."
    )


def test_every_provider_default_in_the_defaults_block_names_a_live_model():
    """The doc-only ``defaults:`` block still has to name models that exist.

    CLAUDE.md documents this block as informational — nothing in the code reads
    it, and the CLI's real default ``--model`` is hardcoded to ``opus5``. That is
    exactly why it rots unnoticed: ``nvidia_default`` sat on
    ``mistral-medium-nvidia`` after the 2026-08-29 removal pass deleted that
    entry, and no test, no loader path and no CLI surface objected. A reader
    reaching for "the recommended NVIDIA model" would have been handed a name
    that fails to resolve.

    Unread configuration that *looks* authoritative is the same hazard as the
    sixteen inert pricing keys, minus the loader — so pin it here instead of
    deleting the block. Also asserts the named model actually belongs to the
    provider in the key, since ``<provider>_default`` claims both.

    The key prefix must *equal* a provider name, bar the one documented
    shorthand below. Prefix matching would be the obvious shortcut and is wrong
    here: ``bedrock`` is a prefix of ``bedrock_openai``, so ``bedrock_default``
    could name a ``bedrock-mantle`` model and still pass, which is the exact
    provider confusion CLAUDE.md warns about.
    """
    key_provider_aliases = {"azure": "azure_openai"}

    repo_root = Path(__file__).resolve().parent.parent
    doc = yaml.safe_load(
        (repo_root / "codereview" / "config" / "models.yaml").read_text()
    )
    configured_providers = set(doc.get("providers") or {})
    stale_aliases = set(key_provider_aliases.values()) - configured_providers
    assert not stale_aliases, (
        f"key_provider_aliases points at providers models.yaml no longer "
        f"defines: {sorted(stale_aliases)}"
    )

    defaults = doc.get("defaults") or {}
    provider_defaults = {
        key: value
        for key, value in defaults.items()
        if key.endswith("_default") and isinstance(value, str)
    }
    assert provider_defaults, (
        "no <provider>_default keys found in the defaults block — it was "
        "renamed or removed and this test is now vacuous"
    )

    loader = ConfigLoader()
    broken: list[str] = []
    for key, name in sorted(provider_defaults.items()):
        prefix = key.removesuffix("_default")
        expected_provider = key_provider_aliases.get(prefix, prefix)
        try:
            provider, config = loader.resolve_model(name)
        except ValueError:
            broken.append(f"{key}: {name!r} does not resolve to any model")
            continue
        if provider != expected_provider:
            broken.append(
                f"{key}: {name!r} resolves to the {provider!r} provider "
                f"(entry {config.id!r}), not {expected_provider!r}"
            )

    assert not broken, (
        "the defaults block in models.yaml names models that no longer exist "
        "or belong to another provider:\n  " + "\n  ".join(broken)
    )


def test_adaptive_thinking_claude_models_disable_tool_use():
    """Adaptive-thinking Claude models must NOT use tool-based structured output.

    Opus 4.7/4.8 only supported ``thinking.type: "adaptive"`` and engaged
    thinking server-side per request; Opus 5 goes further and has thinking on by
    default. ``with_structured_output()`` sets a forced ``tool_choice``, and on
    the batches where these models think, the tool call comes back as markup
    *text* — failing CodeReviewReport validation with a list_type error on
    ``issues``. So they must route through prompt-based JSON parsing
    (``supports_tool_use: false``), same as Kimi K2.6 on Moonshot.

    That failure is **empirical, not a documented API restriction**: Anthropic
    limits ``tool_choice`` to auto/none only under *manual*
    ``thinking.type: "enabled"``, and explicitly documents forced tool use as
    supported with adaptive thinking, "including on models where thinking is on
    by default". langchain-aws agrees — its
    ``thinking_forced_tool_use_unsupported()`` excludes ``claude-opus-4-8`` and
    never listed the 5-generation models. The observation is what stands here
    (it was reproduced live on Opus 4.8, see commit de5e2fc); the mechanism is
    not established.
    Opus 5 has independent confirmation: its Bedrock model card lists
    "Structured outputs: Not Supported" on bedrock-runtime and bedrock-mantle.

    ``opus4.8`` is no longer in this list because that entry was removed
    2026-08-29 (superseded by opus5 at identical pricing) — the *evidence* it
    produced still underpins the flag on every entry here, which is why the
    reproduction is written down rather than just cited.
    """
    loader = ConfigLoader()
    for alias in ("opus5", "sonnet5", "fable5"):
        _, config = loader.resolve_model(alias)
        assert config.supports_tool_use is False, (
            f"{alias} is an adaptive-thinking model and must set "
            "supports_tool_use: false to avoid forced tool_choice"
        )


def test_glm52_zai_disables_tool_use():
    """GLM-5.2 on Z.AI must use prompt-based JSON parsing.

    Z.AI's OpenAI-compat endpoint ignores OpenAI's json_schema response_format
    that with_structured_output() relies on and returns markdown-fenced JSON,
    which the json_schema parser rejects ("Invalid JSON: expected value at line
    1 column 1" in the field) — and GLM-5.2 is additionally a thinking model.
    Both reasons keep it on the PydanticOutputParser path, which strips the
    fences. Resolves via every advertised alias — the version-explicit GLM-5.1
    names were deleted in the 2026-07-25 alias cleanup, not absorbed, so they
    are deliberately absent here (see
    ``RETIRED_ALIASES_DELETED_NOT_REDIRECTED``).
    """
    loader = ConfigLoader()
    aliases = (
        "zhipuai/glm-5.2",
        "glm",
        "glm-5.2",
        "glm5.2",
        "glm5.2-zai",
        "zai-glm",
        "glm-zai",
    )
    for alias in aliases:
        provider, config = loader.resolve_model(alias)
        assert provider == "zai", f"{alias} should route to the zai provider"
        assert config.id == "zhipuai/glm-5.2"
        assert config.supports_tool_use is False, (
            f"{alias} (GLM-5.2) must set supports_tool_use: false — Z.AI returns "
            "markdown-fenced JSON and it's a thinking model"
        )
        assert config.context_window == 1048576


def test_kimi_k3_nvidia_matches_the_model_card():
    """Kimi K3 on NIM: card figures, and no thinking knob that does nothing.

    The card's Output properties say *"Thinking is always enabled"* and expose
    effort levels (low/high/max) rather than an on/off switch, so this entry
    must not ship ``thinking`` / ``enable_thinking`` / ``clear_thinking``:
    ``NVIDIAProvider._create_model`` would forward them as
    ``chat_template_kwargs`` where they cannot change anything, which is the
    ConfigLoader "key that looks like configuration but affects nothing"
    hazard in CLAUDE.md.

    ``reasoning_effort`` is likewise absent: ``InferenceParams`` only permits
    ``none``/``low``/``medium``/``high`` while the card names ``max``, and NIM's
    free tier rate-limited every attempt to verify the wire spelling — an
    unverified knob is worse than the endpoint default.
    """
    loader = ConfigLoader()
    provider, config = loader.resolve_model("kimi-k3-nvidia")
    assert provider == "nvidia"
    assert config.full_id == "moonshotai/kimi-k3"
    # Input Context Length (ISL) from the card, exact rather than rounded.
    assert config.context_window == 1048576
    assert config.inference_params is not None
    assert config.inference_params.temperature == 1.0
    assert config.inference_params.top_p == 0.95
    assert config.inference_params.max_output_tokens == 32768
    for knob in ("thinking", "enable_thinking", "clear_thinking", "reasoning_effort"):
        assert getattr(config.inference_params, knob) is None, (
            f"kimi-k3-nvidia sets {knob}: K3's thinking is always on with no "
            "off switch, so this forwards a parameter that changes nothing"
        )


def test_kimi_k3_nvidia_disables_tool_use():
    """Kimi K3 must use prompt-based JSON parsing, via every advertised name.

    Thinking is always enabled and cannot be disabled — the *constant* form of
    the forced-``tool_choice``-while-thinking failure (like Fable 5 and Grok
    4.3), not the intermittent form. Tool-use is unproven on this endpoint, so
    the assume-prompt-parsing rule applies; both K2.6 entries (NIM and
    Moonshot-direct) are prompt-path for the same reason. Flip to ``true`` only
    with a live review run that shows ``parsing_error`` None while the model
    actually reasoned.
    """
    loader = ConfigLoader()
    for name in ("kimi-k3-nvidia", "kimi-nvidia-3", "kimi3-nvidia"):
        provider, config = loader.resolve_model(name)
        assert provider == "nvidia", f"{name} should route to the nvidia provider"
        assert config.id == "kimi-k3-nvidia"
        assert config.supports_tool_use is False, (
            f"{name} (Kimi K3) must set supports_tool_use: false — thinking is "
            "always on and forced tool_choice is unproven on this endpoint"
        )


def test_bare_kimi_k3_names_stay_reserved_for_the_direct_provider():
    """The NIM re-host must not squat the canonical ``kimi-k3`` spellings.

    Per ``docs/model-registry.md``, a vendor's direct API owns the bare aliases
    and a re-host keeps provider-suffixed ones — which is why the K2.6 re-host
    is ``kimi-nvidia-26``, not ``kimi-k2.6``. Moonshot direct ships K2.6 today;
    when it gains K3 the bare names must be free for it, otherwise ``--model
    kimi-k3`` would silently keep pointing at the free NIM trial endpoint.
    """
    loader = ConfigLoader()
    for name in ("kimi-k3", "kimi3", "k3"):
        with pytest.raises(ValueError):
            loader.resolve_model(name)


def test_deepseek_v4_nvidia_entries_name_the_dated_ga_endpoints():
    """Both NVIDIA DeepSeek-V4 entries must target the dated GA ids.

    The undated preview ids (``deepseek-ai/deepseek-v4-pro`` /
    ``-v4-flash``) were end-of-lifed 2026-08-07 and return HTTP 410 Gone;
    ``DEAD_UPSTREAM_FULL_IDS`` guards against reintroducing them. This is the
    positive half: the entries keep their generation-neutral ids and aliases
    (the successors are drop-in), so only ``full_id`` moved, and a future
    dated release must move it again rather than adding a parallel entry.
    """
    loader = ConfigLoader()
    expected = {
        "deepseek-v4-pro-nvidia": "deepseek-ai/deepseek-v4-pro-0813",
        "deepseek-v4-flash-nvidia": "deepseek-ai/deepseek-v4-flash-0731",
    }
    for model_id, full_id in expected.items():
        provider, config = loader.resolve_model(model_id)
        assert provider == "nvidia"
        assert config.full_id == full_id, (
            f"{model_id} points at {config.full_id!r}; the live NIM catalog "
            f"serves {full_id!r}"
        )

    # The aliases carried over — a scripted `--model dsv4-nvidia` still works.
    for alias in ("dsv4-nvidia", "ds-v4-nvidia", "deepseek-v4-nvidia"):
        _, config = loader.resolve_model(alias)
        assert config.full_id == "deepseek-ai/deepseek-v4-pro-0813"
    for alias in ("dsv4-flash-nvidia", "ds-v4-flash-nvidia"):
        _, config = loader.resolve_model(alias)
        assert config.full_id == "deepseek-ai/deepseek-v4-flash-0731"


def test_gemini37_flash_context_and_output_match_model_card():
    """Gemini 3.7 Flash advertises a 1,048,576-token context and 64K output.

    ``context_window`` is deliberately the conservative 1,000,000 rather than
    the card's exact 1,048,576 (as the removed 3.6 entry also was): under-stating
    the window only makes batches smaller, while over-stating it overflows.
    """
    loader = ConfigLoader()
    provider, config = loader.resolve_model("gemini-3.7-flash")
    assert provider == "google_genai"
    assert config.full_id == "gemini-3.7-flash"
    assert config.context_window == 1_000_000
    assert config.inference_params is not None
    assert config.inference_params.max_output_tokens == 65536


def test_gemini37_flash_omits_sampling_params():
    """From Gemini 3.6 Flash onward, temperature/top_p/top_k are deprecated.

    Google's API ignores all three today and documents an HTTP 400 for future
    model generations. The Google provider passes ``allow_none=True`` to
    ``_resolve_temperature`` and drops ``top_p``/``top_k`` when unset, so
    omitting ``default_temperature``/``default_top_p``/``default_top_k`` from
    the YAML (loaded into ``temperature``/``top_p``/``top_k``) is what keeps
    them off the wire.

    This pinned 3.6 Flash until that entry was removed 2026-08-29; 3.7 Flash is
    now the oldest Gemini entry the rule covers. The cutoff itself is still 3.6 —
    see ``test_every_modern_gemini_entry_omits_sampling_params``, which is what
    catches the *next* entry.
    """
    loader = ConfigLoader()
    _, config = loader.resolve_model("gemini-3.7-flash")
    assert config.inference_params is not None
    assert config.inference_params.temperature is None
    assert config.inference_params.top_p is None
    assert config.inference_params.top_k is None


def test_gemini37_flash_keeps_tool_use_path():
    """Gemini 3.7 Flash earned the tool-use path with a live run.

    The assume-prompt-parsing rule says a new thinking model ships
    ``supports_tool_use: false`` until a live run proves otherwise. Three runs
    on 2026-08-17 each returned a valid ``CodeReviewReport`` with
    ``parsing_error`` None *and* non-zero ``output_token_details.reasoning``,
    i.e. tool-use held while the model was actually thinking. Don't flip this
    to ``false`` without a reproduction.
    """
    loader = ConfigLoader()
    _, config = loader.resolve_model("gemini-3.7-flash")
    assert config.supports_tool_use is True


def test_generation_neutral_gemini_flash_alias_tracks_the_newest_flash():
    """``gemini-flash`` tracks the current Flash generation; a minor version can't.

    The generation-neutral name moved 3.6 -> 3.7 when 3.7 Flash shipped, per the
    convention in ``docs/model-registry.md``. When the 3.6 entry itself was
    removed (2026-08-29) the split held but the sides changed:

    * ``gemini-3-flash``/``gemini3-flash``/``g3flash`` say generation *3*, and
      3.7 Flash is a generation-3 Flash at the same $1.50/$7.50, same 1M context
      and same 64K output — so they follow, as ``deprecated_aliases``.
    * ``gemini36-flash``/``gemini3.6-flash`` pin the *minor* version and were
      deleted instead, so they raise (``test_deleted_aliases_do_not_resolve``).

    That is the line the convention draws: a name may follow the model it names,
    never a version it doesn't.
    """
    loader = ConfigLoader()

    for alias in ("gemini-flash", "gemini-3-flash", "gemini3-flash", "g3flash"):
        _, config = loader.resolve_model(alias)
        assert config.id == "gemini-3.7-flash", (
            f"{alias!r} resolved to {config.id!r} — it must follow the newest "
            "Flash entry"
        )

    for deleted in ("gemini36-flash", "gemini3.6-flash", "gemini-3.6-flash"):
        with pytest.raises(ValueError, match="Unknown model"):
            loader.resolve_model(deleted)


def test_every_modern_gemini_entry_omits_sampling_params():
    """Reflective guard: no Gemini entry from 3.6 onward may ship a sampler.

    ``test_gemini37_flash_omits_sampling_params`` pins one entry; this one
    fails when a *new* Gemini entry reintroduces ``default_temperature`` /
    ``default_top_p`` / ``default_top_k``, which the API ignores today and
    documents an HTTP 400 for on future generations. Gemini 3.1 Pro predates
    the deprecation and keeps its sampling params, so the cutoff stays 3.6 even
    though no 3.6 entry is registered any more.
    """
    loader = ConfigLoader()
    models = loader.list_models().get("google_genai", [])
    assert models, "no google_genai models configured"

    checked = []
    for config in models:
        version = re.search(r"(\d+)\.(\d+)", config.id)
        if version is None:
            continue
        if (int(version.group(1)), int(version.group(2))) < (3, 6):
            continue
        checked.append(config.id)
        assert config.inference_params is not None, config.id
        for param in ("temperature", "top_p", "top_k"):
            assert getattr(config.inference_params, param) is None, (
                f"{config.id} ships default_{param}: sampling params are "
                "deprecated from Gemini 3.6 Flash onward — omit all three"
            )

    assert checked, "no Gemini entry at 3.6 or newer was checked"


# ---------------------------------------------------------------------------
# Per-language prompt slicing
# ---------------------------------------------------------------------------


def test_build_system_prompt_includes_only_requested_languages():
    from codereview.config import LANGUAGE_RULES, build_system_prompt

    prompt = build_system_prompt({"python", "go"})
    assert LANGUAGE_RULES["python"] in prompt
    assert LANGUAGE_RULES["go"] in prompt
    # Sections that should not be present when the batch is python+go only
    assert LANGUAGE_RULES["java"] not in prompt
    assert LANGUAGE_RULES["typescript"] not in prompt


def test_build_system_prompt_falls_back_to_all_when_empty():
    from codereview.config import LANGUAGE_RULES, build_system_prompt

    prompt = build_system_prompt(set())
    for block in LANGUAGE_RULES.values():
        assert block in prompt


def test_build_system_prompt_unknown_keys_fall_back_to_all():
    """An entirely-unknown set yields the all-languages prompt, not an empty one."""
    from codereview.config import LANGUAGE_RULES, build_system_prompt

    prompt = build_system_prompt({"cobol", "fortran"})
    for block in LANGUAGE_RULES.values():
        assert block in prompt


def test_build_system_prompt_preserves_canonical_order():
    """Output is stable across runs even when the input is a set."""
    from codereview.config import build_system_prompt

    a = build_system_prompt({"go", "python"})
    b = build_system_prompt({"python", "go"})
    assert a == b


def test_build_system_prompt_has_no_unsubstituted_placeholders():
    """Every {placeholder} in the template must be filled for both gatings.

    Guards against a new template token being added without a substitution
    (which would otherwise ship a literal ``{token}`` to the model). The
    shell rule's ``"${var}"`` example is the one legitimate brace sequence.
    """
    import re

    from codereview.config import build_system_prompt

    for linters_ran in (True, False):
        prompt = build_system_prompt({"python"}, linters_ran=linters_ran)
        leftover = [m for m in re.findall(r"\{[a-z_]+\}", prompt) if m != "{var}"]
        assert not leftover, f"unsubstituted placeholders: {leftover}"


def test_build_system_prompt_linter_guidance_is_gated():
    """R4: the 'linters already ran' framing only ships when linters ran.

    When static analysis did NOT run (the default), telling the model to
    defer to linters would silently suppress findings the user can't get
    any other way.
    """
    from codereview.config import build_system_prompt

    ran = build_system_prompt({"python"}, linters_ran=True)
    not_ran = build_system_prompt({"python"}, linters_ran=False)

    assert "HAVE already run" in ran
    assert "No linter has run" not in ran
    assert "No linter has run" in not_ran
    assert "HAVE already run" not in not_ran


def test_build_system_prompt_defaults_to_linters_ran():
    """Default (no arg) preserves the prior 'linters ran' behavior."""
    from codereview.config import build_system_prompt

    assert build_system_prompt({"python"}) == build_system_prompt(
        {"python"}, linters_ran=True
    )


def test_build_system_prompt_protects_critical_high_from_issue_cap():
    """R1: the issue cap must never drop a Critical/High finding."""
    from codereview.config import build_system_prompt

    prompt = build_system_prompt({"python"})
    assert "NEVER drop a Critical or High" in prompt


def test_build_system_prompt_includes_line_number_gutter_example():
    """R2: a worked example teaches reading the NNN | gutter for line numbers."""
    from codereview.config import build_system_prompt

    prompt = build_system_prompt({"python"})
    assert "read them from the gutter" in prompt


def test_detect_languages_from_paths_basic():
    from codereview.config import detect_languages_from_paths

    langs = detect_languages_from_paths(
        ["app/main.py", "lib/util.go", "scripts/run.sh", "Frame.java"]
    )
    assert langs == {"python", "go", "shell", "java"}


def test_detect_languages_from_paths_unknown_extensions_ignored():
    from codereview.config import detect_languages_from_paths

    langs = detect_languages_from_paths(["readme.md", "data.json", "image.png"])
    assert langs == set()


def test_detect_languages_handles_uppercase_extensions():
    from codereview.config import detect_languages_from_paths

    assert detect_languages_from_paths(["Foo.PY", "Bar.JAVA"]) == {"python", "java"}


def test_system_prompt_alias_matches_full_render():
    """SYSTEM_PROMPT (legacy export) equals build_system_prompt() with no args."""
    from codereview.config import SYSTEM_PROMPT, build_system_prompt

    assert SYSTEM_PROMPT == build_system_prompt()


def test_canonical_owner_aliases_route_to_direct_api():
    """Lock the canonical-owner convention (CLAUDE.md).

    When a model is exposed by both the vendor's direct API and a re-hoster,
    the direct API owns the canonical aliases. Alias collisions resolve
    first-registration-wins with only a log warning, so without this test a
    re-hoster gaining a canonical alias (or a reorder of the provider parsing
    branches in loader.py) would silently reroute these — changing pricing
    and transport for anyone using the alias.
    """
    loader = ConfigLoader()
    canonical_owners = {
        "deepseek-v4-pro": "deepseek",  # not NVIDIA's free re-host
        "kimi": "moonshot",  # not Bedrock's K2.5 or NVIDIA's K2.6
        "kimi-k2.6": "moonshot",
    }
    for alias, owner in canonical_owners.items():
        provider, _ = loader.resolve_model(alias)
        assert provider == owner, (
            f"canonical alias {alias!r} must route to {owner!r} (direct API), "
            f"got {provider!r} — re-host entries keep suffixed aliases only"
        )


# ---------------------------------------------------------------------------
# Provider-level YAML keys must actually reach the provider config object
# ---------------------------------------------------------------------------

# Every non-default provider-level value to write into a scratch models.yaml,
# and the attribute it must show up on. Deliberately distinctive numbers so a
# class default can't accidentally match.
_PROVIDER_LEVEL_OVERRIDES: dict[str, dict[str, object]] = {
    "bedrock": {"read_timeout": 111, "connect_timeout": 22},
    "azure_openai": {"request_timeout": 444},
    "nvidia": {"polling_timeout": 333, "max_retries": 9},
    "google_genai": {"request_timeout": 555},
    "deepseek": {"request_timeout": 666},
    "moonshot": {"request_timeout": 777},
    "zai": {"request_timeout": 888},
    "bedrock_openai": {"request_timeout": 999},
}

# Credentials each provider's branch requires before it registers a config at
# all (the loader skips unconfigured providers so --list-models still works).
_PROVIDER_CREDENTIALS: dict[str, dict[str, str]] = {
    "azure_openai": {
        "endpoint": "https://example.openai.azure.com",
        "api_key": "a" * 40,
        "api_version": "2025-04-01-preview",
    },
    "nvidia": {"api_key": "nvapi-" + "x" * 30},
    "google_genai": {"api_key": "g" * 40},
    "deepseek": {"api_key": "d" * 40},
    "moonshot": {"api_key": "m" * 40},
    "zai": {"api_key": "z" * 40},
    "bedrock_openai": {
        "api_key": "b" * 40,
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    },
}


def _loader_with_provider_overrides(tmp_path: Path) -> ConfigLoader:
    """A ConfigLoader over the real models.yaml with every knob turned."""
    raw = yaml.safe_load(
        (Path("codereview/config/models.yaml")).read_text(encoding="utf-8")
    )
    for provider, overrides in _PROVIDER_LEVEL_OVERRIDES.items():
        block = raw["providers"][provider]
        block.update(_PROVIDER_CREDENTIALS.get(provider, {}))
        block.update(overrides)

    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return ConfigLoader(path)


@pytest.mark.parametrize(
    "provider, field, expected",
    [
        (provider, field, value)
        for provider, overrides in _PROVIDER_LEVEL_OVERRIDES.items()
        for field, value in overrides.items()
    ],
)
def test_provider_level_yaml_value_reaches_the_config_object(
    tmp_path, provider, field, expected
):
    """A provider-level key in models.yaml must not be silently inert.

    ``_parse_providers`` constructs each ``*Config`` with an explicit keyword
    list, so a field the class declares but the branch forgets to forward keeps
    its class default and the YAML value becomes a comment. Five were being
    dropped: Bedrock's ``read_timeout``/``connect_timeout``, NVIDIA's
    ``polling_timeout``/``max_retries``, and Azure's ``request_timeout`` —
    including the two the docs advertise as the tuning knobs for exactly the
    failures they address (Converse read timeouts on always-thinking models,
    and NIM's frequent gateway 504s).
    """
    loader = _loader_with_provider_overrides(tmp_path)

    config = loader.get_provider_config(provider)

    assert getattr(config, field) == expected, (
        f"providers.{provider}.{field} in models.yaml never reached "
        f"{type(config).__name__}; the class default won and the YAML value "
        "has no effect"
    )


def test_every_declared_provider_config_field_is_forwarded_by_the_loader(tmp_path):
    """Coverage guard: no settable provider-level field goes unforwarded.

    The parametrized test above only checks the fields listed in
    ``_PROVIDER_LEVEL_OVERRIDES``. This one reflects over each Pydantic config
    class and fails when a *new* tunable field appears that neither the loader
    forwards nor this file covers — which is how the original five slipped in.
    """
    loader = _loader_with_provider_overrides(tmp_path)

    # Not provider-level knobs: models comes from the models: list, and the
    # credential/identity fields are covered by _PROVIDER_CREDENTIALS above.
    structural = {
        "models",
        "api_key",
        "endpoint",
        "api_version",
        "base_url",
        "api_base",
    }

    for provider, overrides in _PROVIDER_LEVEL_OVERRIDES.items():
        config = loader.get_provider_config(provider)
        tunable = {
            name
            for name in type(config).model_fields
            if name not in structural and name != "region"
        }
        missing = tunable - set(overrides)
        assert not missing, (
            f"{type(config).__name__} declares {sorted(missing)}, which "
            f"_PROVIDER_LEVEL_OVERRIDES does not exercise — add it there (and "
            f"forward it in loader.py's {provider} branch) so the YAML key "
            "cannot be silently inert"
        )


# ---------------------------------------------------------------------------
# Config-error diagnostics
#
# ConfigLoader runs from __init__, so a malformed models.yaml surfaces on
# *every* command — including --list-models, which needs no credentials. The
# raw exceptions Pydantic and dict indexing raise name neither the file nor the
# entry, which is useless when the file holds ~30 model entries and the user
# may be editing a copy in a different directory.
# ---------------------------------------------------------------------------


def _config_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_model_key_names_the_file_and_the_entry(tmp_path):
    """A missing required key must be a ValueError naming file *and* entry.

    Regression: this escaped as a bare ``KeyError: 'pricing'`` from inside
    ``_parse_model_config``. With one line of traceback pointing at the loader,
    the user learns neither which YAML file was read (it can be a copy, or an
    overridden path) nor which of ~30 entries lacks the key.
    """
    path = _config_file(
        tmp_path,
        """
providers:
  bedrock:
    models:
      - id: broken-entry
        name: Broken
        full_id: vendor.broken
""",
    )

    with pytest.raises(ValueError) as excinfo:
        ConfigLoader(path)

    message = str(excinfo.value)
    assert "broken-entry" in message
    assert str(path) in message
    assert "pricing" in message


def test_invalid_model_value_names_the_file_and_the_entry(tmp_path):
    """A schema violation must be a ValueError, not a raw ValidationError.

    Pydantic's message says *which field* but not which entry or file, and
    ``ValidationError`` is not a ``ValueError`` subclass callers can rely on
    catching alongside the loader's other failures.
    """
    path = _config_file(
        tmp_path,
        """
providers:
  bedrock:
    models:
      - id: ""
        name: Nameless Id
        full_id: vendor.x
        pricing:
          input_per_million: 1.0
          output_per_million: 2.0
""",
    )

    with pytest.raises(ValueError) as excinfo:
        ConfigLoader(path)

    message = str(excinfo.value)
    # `id` is the broken field, so the entry is identified by its name.
    assert "Nameless Id" in message
    assert str(path) in message


def test_entry_with_no_identifier_at_all_still_reports_the_file(tmp_path):
    """An entry missing every identifying key must not crash the reporter.

    The label falls back id → name → full_id → placeholder precisely because a
    missing ``id`` is one of the failures being reported.
    """
    path = _config_file(
        tmp_path,
        """
providers:
  bedrock:
    models:
      - pricing:
          input_per_million: 1.0
          output_per_million: 2.0
""",
    )

    with pytest.raises(ValueError) as excinfo:
        ConfigLoader(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "unnamed entry" in message


def test_invalid_non_model_section_names_the_file(tmp_path):
    """A bad value outside the models list must also name the config file.

    ``scanning:`` is parsed by its own method, so it needs the top-level
    ``_load_config`` net rather than ``_parse_model_config``'s.
    """
    path = _config_file(tmp_path, 'scanning:\n  max_file_size_kb: "not a number"\n')

    with pytest.raises(ValueError) as excinfo:
        ConfigLoader(path)

    assert str(path) in str(excinfo.value)


def test_malformed_yaml_names_the_file(tmp_path):
    """The YAML branch named the parse error but not which file failed."""
    path = _config_file(tmp_path, "providers: [unterminated\n")

    with pytest.raises(ValueError) as excinfo:
        ConfigLoader(path)

    assert str(path) in str(excinfo.value)


def test_the_shipped_config_loads_without_diagnostics(tmp_path):
    """The error paths above must not have made the real config unloadable."""
    loader = ConfigLoader(Path("codereview/config/models.yaml"))
    assert loader.list_models()


# ---------------------------------------------------------------------------
# Legacy module-level constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, accessor_name",
    [
        ("DEFAULT_EXCLUDE_PATTERNS", "get_default_exclude_patterns"),
        ("DEFAULT_EXCLUDE_EXTENSIONS", "get_default_exclude_extensions"),
        ("MAX_FILE_SIZE_KB", "get_max_file_size_kb"),
        ("WARN_FILE_SIZE_KB", "get_warn_file_size_kb"),
        ("MODEL_ALIASES", "get_model_aliases"),
    ],
)
def test_legacy_constant_agrees_with_its_accessor_after_a_cache_clear(
    tmp_path, monkeypatch, name, accessor_name
):
    """The legacy names must follow ``get_config_loader.cache_clear()``.

    Regression: these five were assigned once at package import
    (``MAX_FILE_SIZE_KB = get_max_file_size_kb()``), so a test or caller that
    reloaded config via the documented ``cache_clear()`` reset got the *new*
    value from the accessor and the *old* value from the constant — two
    spellings of one setting silently disagreeing, with no error to notice.
    """
    import codereview.config as config_pkg
    from codereview.config import get_config_loader

    raw = yaml.safe_load(
        Path("codereview/config/models.yaml").read_text(encoding="utf-8")
    )
    raw["scanning"]["max_file_size_kb"] = 42
    raw["scanning"]["warn_file_size_kb"] = 7
    raw["scanning"]["exclude_patterns"] = ["**/only_this/**"]
    raw["scanning"]["exclude_extensions"] = [".only"]
    # A single model entry, so MODEL_ALIASES is unmistakably different too.
    raw["providers"] = {
        "bedrock": {
            "models": [
                {
                    "id": "solo",
                    "name": "Solo",
                    "full_id": "vendor.solo",
                    "pricing": {"input_per_million": 1.0, "output_per_million": 2.0},
                }
            ]
        }
    }
    alternate = tmp_path / "models.yaml"
    alternate.write_text(yaml.safe_dump(raw), encoding="utf-8")

    original_init = ConfigLoader.__init__

    def init_from_alternate(self, config_path=None):
        original_init(self, alternate)

    monkeypatch.setattr(ConfigLoader, "__init__", init_from_alternate)
    get_config_loader.cache_clear()
    try:
        expected = getattr(config_pkg, accessor_name)()
        assert getattr(config_pkg, name) == expected
    finally:
        monkeypatch.undo()
        get_config_loader.cache_clear()


def test_unknown_config_attribute_still_raises_attribute_error():
    """The module __getattr__ must not turn typos into something else."""
    import codereview.config as config_pkg

    with pytest.raises(AttributeError, match="no attribute 'NOT_A_SETTING'"):
        _ = config_pkg.NOT_A_SETTING


def test_legacy_constants_stay_visible_to_dir():
    """Lazy attributes are invisible to dir() unless __dir__ lists them."""
    import codereview.config as config_pkg

    listing = dir(config_pkg)
    for name in config_pkg.__all__:
        assert name in listing, f"{name} is exported but not discoverable"
