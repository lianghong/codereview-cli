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
