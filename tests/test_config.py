# tests/test_config.py
from codereview.config import (
    DEFAULT_EXCLUDE_EXTENSIONS,
    DEFAULT_EXCLUDE_PATTERNS,
    MODEL_ALIASES,
    MODEL_CONFIG,
    SUPPORTED_MODELS,
    SYSTEM_PROMPT,
    resolve_model_id,
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


def test_model_config():
    """Test AWS Bedrock model configuration."""
    assert MODEL_CONFIG["model_id"] == "global.anthropic.claude-opus-4-5-20251101-v1:0"
    assert MODEL_CONFIG["temperature"] == 0.1
    assert MODEL_CONFIG["max_tokens"] > 0


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
    assert "minimax" in MODEL_ALIASES
    assert "mistral" in MODEL_ALIASES
    assert "kimi" in MODEL_ALIASES
    assert "qwen" in MODEL_ALIASES


def test_resolve_model_id_with_alias():
    """Test resolving short model names to full IDs."""
    assert resolve_model_id("opus") == "global.anthropic.claude-opus-4-5-20251101-v1:0"
    assert (
        resolve_model_id("sonnet") == "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    assert (
        resolve_model_id("haiku") == "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    assert resolve_model_id("minimax") == "minimax.minimax-m2"
    assert resolve_model_id("qwen") == "qwen.qwen3-coder-480b-a35b-v1:0"


def test_resolve_model_id_case_insensitive():
    """Test model name resolution is case-insensitive."""
    assert resolve_model_id("OPUS") == resolve_model_id("opus")
    assert resolve_model_id("Sonnet") == resolve_model_id("sonnet")


def test_resolve_model_id_with_full_id():
    """Test passing full model ID returns it unchanged."""
    full_id = "global.anthropic.claude-opus-4-5-20251101-v1:0"
    assert resolve_model_id(full_id) == full_id


def test_all_aliases_map_to_supported_models():
    """Test all aliases map to models in SUPPORTED_MODELS."""
    for alias, model_id in MODEL_ALIASES.items():
        assert (
            model_id in SUPPORTED_MODELS
        ), f"Alias '{alias}' maps to unknown model '{model_id}'"
