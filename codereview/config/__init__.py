"""Configuration package for model and provider configuration management."""

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

# Import Pydantic models
# Import ConfigLoader
from codereview.config.loader import ConfigLoader
from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    InferenceParams,
    ModelConfig,
    ModelsConfigFile,
    PricingConfig,
    ProviderConfig,
    ScanningConfig,
)

# Import system prompt
from codereview.config.prompts import (
    LANGUAGE_RULES,
    SYSTEM_PROMPT,
    build_system_prompt,
    detect_languages_from_paths,
)


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    """Get the default ConfigLoader instance (singleton).

    Uses lru_cache for thread-safe lazy initialization.
    Call get_config_loader.cache_clear() to reset for testing.
    """
    return ConfigLoader()


# Convenience accessors for scanning config
def get_default_exclude_patterns() -> list[str]:
    """Get default file exclusion patterns."""
    return list(get_config_loader().scanning_config.exclude_patterns)


def get_default_exclude_extensions() -> list[str]:
    """Get default file extension exclusions."""
    return list(get_config_loader().scanning_config.exclude_extensions)


def get_max_file_size_kb() -> int:
    """Get maximum file size in KB."""
    return get_config_loader().scanning_config.max_file_size_kb


def get_warn_file_size_kb() -> int:
    """Get file size warning threshold in KB."""
    return get_config_loader().scanning_config.warn_file_size_kb


# Convenience accessor for model aliases (for CLI)
def get_model_aliases() -> dict[str, str]:
    """Get all model aliases mapped to their primary IDs."""
    return get_config_loader().get_model_aliases()


# Legacy compatibility exports.
#
# Resolved lazily through the module-level __getattr__ (PEP 562) rather than
# assigned at import time, so each name re-reads the *current* config on every
# attribute access.
#
# The bug this fixes: the eager snapshot was taken once, at first import, so it
# could not follow `get_config_loader.cache_clear()` — the documented way to
# reset config in tests (CLAUDE.md). Every accessor above picked up the
# reloaded config while these five names kept the values from the first import,
# a silent disagreement between two spellings of the same setting.
#
# What it does *not* fix: importing this package can still run the YAML load
# eagerly, because `scanner.py` and `cli.py` bind some of these names with a
# module-level `from codereview.config import ...`. A `from ... import` copies
# the value at the *importing* module's import time, which no amount of
# laziness here can defer. That's why the accessor functions above remain the
# preferred spelling for new code.
_LEGACY_ACCESSORS: dict[str, Callable[[], object]] = {
    "DEFAULT_EXCLUDE_PATTERNS": get_default_exclude_patterns,
    "DEFAULT_EXCLUDE_EXTENSIONS": get_default_exclude_extensions,
    "MAX_FILE_SIZE_KB": get_max_file_size_kb,
    "WARN_FILE_SIZE_KB": get_warn_file_size_kb,
    "MODEL_ALIASES": get_model_aliases,
}

if TYPE_CHECKING:
    # A module-level __getattr__ returns one type for every name, so without
    # these declarations mypy infers `object` for all five and every use site
    # downstream fails (`MAX_FILE_SIZE_KB` as an int default, iterating the
    # pattern lists, `MODEL_ALIASES.keys()`). Type-checking-only: at runtime
    # these names must stay absent from the module globals or __getattr__ is
    # never consulted and the laziness is undone. Keep the annotations in step
    # with the accessors' return types above.
    DEFAULT_EXCLUDE_PATTERNS: list[str]
    DEFAULT_EXCLUDE_EXTENSIONS: list[str]
    MAX_FILE_SIZE_KB: int
    WARN_FILE_SIZE_KB: int
    MODEL_ALIASES: dict[str, str]


def __getattr__(name: str) -> object:
    """Resolve the legacy module-level constants on first access."""
    try:
        accessor = _LEGACY_ACCESSORS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return accessor()


def __dir__() -> list[str]:
    """Keep the lazy names discoverable by dir() and tab completion."""
    return sorted({*globals(), *_LEGACY_ACCESSORS})


__all__ = [
    # Pydantic models
    "PricingConfig",
    "InferenceParams",
    "ModelConfig",
    "ProviderConfig",
    "BedrockConfig",
    "AzureOpenAIConfig",
    "ModelsConfigFile",
    "ScanningConfig",
    # ConfigLoader
    "ConfigLoader",
    "get_config_loader",
    # System prompt
    "SYSTEM_PROMPT",
    "LANGUAGE_RULES",
    "build_system_prompt",
    "detect_languages_from_paths",
    # Convenience accessors
    "get_default_exclude_patterns",
    "get_default_exclude_extensions",
    "get_max_file_size_kb",
    "get_warn_file_size_kb",
    "get_model_aliases",
    # Legacy compatibility
    "DEFAULT_EXCLUDE_PATTERNS",
    "DEFAULT_EXCLUDE_EXTENSIONS",
    "MAX_FILE_SIZE_KB",
    "WARN_FILE_SIZE_KB",
    "MODEL_ALIASES",
]
