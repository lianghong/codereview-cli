# tests/test_config.py
from codereview.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_EXTENSIONS,
    MODEL_CONFIG,
    SYSTEM_PROMPT
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
