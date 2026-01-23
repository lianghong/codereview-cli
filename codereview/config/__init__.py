"""Configuration package for model and provider configuration management."""

# Import new Pydantic models
from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    InferenceParams,
    ModelConfig,
    ModelsConfigFile,
    PricingConfig,
    ProviderConfig,
)

# Re-export everything from old config.py for backward compatibility
# This allows existing code to continue importing from codereview.config
import sys
from pathlib import Path

# Import from parent's config.py (the old module)
parent_dir = Path(__file__).parent.parent
config_py = parent_dir / "config.py"

if config_py.exists():
    # Use exec to load the old config.py and make its contents available
    import importlib.util
    spec = importlib.util.spec_from_file_location("_old_config", config_py)
    if spec and spec.loader:
        _old_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_old_config)

        # Re-export all public names from old config
        for name in dir(_old_config):
            if not name.startswith('_'):
                globals()[name] = getattr(_old_config, name)

__all__ = [
    # New Pydantic models
    "PricingConfig",
    "InferenceParams",
    "ModelConfig",
    "ProviderConfig",
    "BedrockConfig",
    "AzureOpenAIConfig",
    "ModelsConfigFile",
    # Old config.py exports (will be added dynamically)
    "ModelInfo",
    "DEFAULT_EXCLUDE_PATTERNS",
    "DEFAULT_EXCLUDE_EXTENSIONS",
    "MODEL_CONFIG",
    "SUPPORTED_MODELS",
    "SYSTEM_PROMPT",
    "MAX_FILE_SIZE_KB",
]
