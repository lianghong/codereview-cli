"""Configuration package for model and provider configuration management."""

from functools import lru_cache

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
from codereview.config.prompts import SYSTEM_PROMPT


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


# Legacy compatibility exports
# These are provided for backward compatibility during migration
DEFAULT_EXCLUDE_PATTERNS = get_default_exclude_patterns()
DEFAULT_EXCLUDE_EXTENSIONS = get_default_exclude_extensions()
MAX_FILE_SIZE_KB = get_max_file_size_kb()
WARN_FILE_SIZE_KB = get_warn_file_size_kb()
MODEL_ALIASES = get_model_aliases()

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
