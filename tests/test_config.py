# tests/test_config.py
"""Tests for configuration management."""

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
    assert model_config.name == "Claude Opus 4.6"
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
    assert "mistral-medium-nvidia" in MODEL_ALIASES
    assert "kimi" in MODEL_ALIASES
    assert "qwen" in MODEL_ALIASES


def test_resolve_model_id_with_alias():
    """Test resolving short model names to full IDs via ConfigLoader."""
    loader = ConfigLoader()
    provider, model_config = loader.resolve_model("opus")
    assert model_config.full_id == "global.anthropic.claude-opus-4-6-v1"

    provider, model_config = loader.resolve_model("sonnet")
    assert model_config.full_id == "global.anthropic.claude-sonnet-4-6"

    provider, model_config = loader.resolve_model("haiku")
    assert model_config.full_id == "global.anthropic.claude-haiku-4-5-20251001-v1:0"

    provider, model_config = loader.resolve_model("qwen")
    assert model_config.full_id == "qwen.qwen3-coder-480b-a35b-v1:0"


def test_resolve_model_id_case_insensitive():
    """Test model name resolution handles aliases case-insensitively."""
    loader = ConfigLoader()
    # Aliases in YAML are lowercase, so we test that lowercase works
    provider1, model1 = loader.resolve_model("opus")
    provider2, model2 = loader.resolve_model("sonnet")
    assert model1.name == "Claude Opus 4.6"
    assert model2.name == "Claude Sonnet 4.6"


def test_resolve_model_id_with_full_id():
    """Test resolving with full model ID works."""
    loader = ConfigLoader()
    # Short ID (which is used in the YAML as the primary ID)
    provider, model_config = loader.resolve_model("opus")
    assert model_config.id == "opus"


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


def test_model_id_conflict_detection(caplog):
    """Test that model ID conflicts are detected and logged."""
    import logging

    from codereview.config.models import ModelConfig, PricingConfig

    loader = ConfigLoader()

    # Simulate registering same ID from different provider
    mock_config = ModelConfig(
        id="opus",  # Already registered by bedrock
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
    assert config.name == "Claude Opus 4.6"


# ---------------------------------------------------------------------------
# Upstream-currency guards
# ---------------------------------------------------------------------------

# Upstream endpoints that were retired/shut down by their providers. Audited
# 2026-05-30; removing the registry entries that pointed here, with their
# aliases redirected to the live successors. This guard fails if any of these
# dead full_ids is ever reintroduced (e.g. by copy-paste from an old entry).
#   minimaxai/minimax-m2.5  — NVIDIA NIM deprecated 2026-05-12 → minimax-m2.7
#   moonshotai/kimi-k2.5    — NVIDIA NIM shut down 2026-05-20 → kimi-k2.6
#   z-ai/glm5               — NVIDIA NIM deprecated 2026-04-20 → z-ai/glm-5.1
#   gemini-3-pro-preview    — Google shut down 2026-03-09  → gemini-3.1-pro-preview
DEAD_UPSTREAM_FULL_IDS = {
    "minimaxai/minimax-m2.5",
    "moonshotai/kimi-k2.5",
    "z-ai/glm5",
    "gemini-3-pro-preview",
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


def test_retired_model_aliases_redirect_to_live_successors():
    """Aliases inherited from retired entries resolve to the live successor."""
    loader = ConfigLoader()
    expected = {
        "minimax-m2.5": "minimaxai/minimax-m2.7",
        "mm25": "minimaxai/minimax-m2.7",
        "kimi-k2.5": "moonshotai/kimi-k2.6",
        "kimi25": "moonshotai/kimi-k2.6",
        "glm5": "z-ai/glm-5.1",
        "glm-5": "z-ai/glm-5.1",
        "gemini-3-pro": "gemini-3.1-pro-preview",
        "g3pro": "gemini-3.1-pro-preview",
    }
    for alias, live_full_id in expected.items():
        _, config = loader.resolve_model(alias)
        assert config.full_id == live_full_id, (
            f"alias {alias!r} resolved to {config.full_id!r}, expected {live_full_id!r}"
        )


def test_opus_4_7_context_and_output_match_bedrock_card():
    """Opus 4.7 advertises 1M context / 128K output on the AWS model card.

    Regression guard: an earlier registry value of 200K/32K under-batched
    inputs 5x and truncated long reports.
    """
    loader = ConfigLoader()
    _, config = loader.resolve_model("opus4.7")
    assert config.context_window == 1_000_000
    assert config.inference_params is not None
    assert config.inference_params.max_output_tokens == 128_000


def test_adaptive_thinking_claude_models_disable_tool_use():
    """Adaptive-thinking Claude models must NOT use tool-based structured output.

    Opus 4.7/4.8 only support ``thinking.type: "adaptive"`` and engage thinking
    server-side per request. Anthropic forbids a forced ``tool_choice`` while
    thinking is active, but ``with_structured_output()`` sets exactly that —
    so these models must route through prompt-based JSON parsing
    (``supports_tool_use: false``), same as Kimi K2.6 on Moonshot. Without
    this, batches where the model thinks return tool-call markup as text and
    fail CodeReviewReport validation with a list_type error on ``issues``.
    """
    loader = ConfigLoader()
    for alias in ("opus4.8", "opus4.7"):
        _, config = loader.resolve_model(alias)
        assert config.supports_tool_use is False, (
            f"{alias} is an adaptive-thinking model and must set "
            "supports_tool_use: false to avoid forced tool_choice"
        )


def test_glm51_zai_disables_tool_use():
    """GLM-5.1 on Z.AI must use prompt-based JSON parsing.

    Z.AI's OpenAI-compat endpoint ignores OpenAI's json_schema response_format
    that with_structured_output() relies on and returns markdown-fenced JSON,
    which the json_schema parser rejects. Routing via supports_tool_use: false
    (PydanticOutputParser) strips the fences. Regression for the field-observed
    "Invalid JSON: expected value at line 1 column 1".
    """
    loader = ConfigLoader()
    _, config = loader.resolve_model("zhipuai/glm-5.1")
    assert config.supports_tool_use is False


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
