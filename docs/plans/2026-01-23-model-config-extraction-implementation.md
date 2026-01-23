# Model Configuration Extraction - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract model configurations to external YAML with multi-provider support (Bedrock + Azure OpenAI)

**Architecture:** Provider adapter pattern with YAML-based config, Pydantic validation, and auto-detection of provider from model name

**Tech Stack:** Python 3.14, Pydantic V2, PyYAML, LangChain (Bedrock + OpenAI), Rich CLI

---

## Phase 1: Configuration System (No Breaking Changes)

### Task 1.1: Add PyYAML Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add pyyaml and langchain-openai to dependencies**

Edit `pyproject.toml` dependencies section:

```toml
dependencies = [
    "langchain>=0.3.21",
    "langchain-aws>=0.2.14",
    "langchain-openai>=0.2.14",  # NEW
    "pydantic>=2.10.6",
    "click>=8.1.8",
    "rich>=14.0.0",
    "boto3>=1.37.12",
    "pyyaml>=6.0",  # NEW
]
```

**Step 2: Install new dependencies**

Run: `uv pip install -e .`
Expected: Successfully installs pyyaml and langchain-openai

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pyyaml and langchain-openai dependencies"
```

---

### Task 1.2: Create Pydantic Data Models

**Files:**
- Create: `codereview/config/__init__.py`
- Create: `codereview/config/models.py`

**Step 1: Write test for PricingConfig validation**

Create `tests/test_config_models.py`:

```python
"""Tests for configuration data models."""

import pytest
from pydantic import ValidationError

from codereview.config.models import (
    InferenceParams,
    ModelConfig,
    PricingConfig,
)


def test_pricing_config_valid():
    """Test valid pricing configuration."""
    pricing = PricingConfig(input_per_million=5.0, output_per_million=25.0)
    assert pricing.input_per_million == 5.0
    assert pricing.output_per_million == 25.0


def test_pricing_config_rejects_negative():
    """Test pricing rejects negative values."""
    with pytest.raises(ValidationError):
        PricingConfig(input_per_million=-1.0, output_per_million=25.0)


def test_pricing_config_rejects_zero():
    """Test pricing rejects zero values."""
    with pytest.raises(ValidationError):
        PricingConfig(input_per_million=0.0, output_per_million=25.0)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_models.py::test_pricing_config_valid -v`
Expected: FAIL with "No module named 'codereview.config'"

**Step 3: Create config package init**

Create `codereview/config/__init__.py`:

```python
"""Configuration management for code review tool."""

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
    ProviderConfig,
)

__all__ = [
    "AzureOpenAIConfig",
    "BedrockConfig",
    "InferenceParams",
    "ModelConfig",
    "PricingConfig",
    "ProviderConfig",
]
```

**Step 4: Write minimal implementation of data models**

Create `codereview/config/models.py`:

```python
"""Data models for provider and model configuration."""

from typing import Any

from pydantic import BaseModel, Field


class PricingConfig(BaseModel):
    """Pricing information for a model."""

    input_per_million: float = Field(gt=0, description="Input price per million tokens")
    output_per_million: float = Field(
        gt=0, description="Output price per million tokens"
    )


class InferenceParams(BaseModel):
    """Model inference parameters."""

    default_temperature: float | None = Field(
        None, ge=0, le=2, description="Default temperature"
    )
    default_top_p: float | None = Field(None, ge=0, le=1, description="Default top_p")
    default_top_k: int | None = Field(None, ge=0, description="Default top_k")
    max_output_tokens: int | None = Field(None, gt=0, description="Max output tokens")


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    id: str = Field(description="Short model identifier for CLI")
    name: str = Field(description="Human-readable model name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    pricing: PricingConfig
    inference_params: InferenceParams = Field(default_factory=InferenceParams)

    # Provider-specific fields (validated per provider)
    full_id: str | None = Field(None, description="Full model ID (Bedrock)")
    deployment_name: str | None = Field(
        None, description="Deployment name (Azure OpenAI)"
    )


class ProviderConfig(BaseModel):
    """Base configuration for a provider."""

    models: list[ModelConfig]


class BedrockConfig(ProviderConfig):
    """Bedrock-specific configuration."""

    region: str = Field(default="us-west-2", description="AWS region")


class AzureOpenAIConfig(ProviderConfig):
    """Azure OpenAI-specific configuration."""

    endpoint: str = Field(description="Azure OpenAI endpoint URL")
    api_key: str = Field(description="Azure OpenAI API key")
    api_version: str = Field(description="Azure OpenAI API version")


class ModelsConfigFile(BaseModel):
    """Root configuration file structure."""

    providers: dict[str, dict[str, Any]]
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_models.py -v`
Expected: All 3 tests PASS

**Step 6: Add more validation tests**

Add to `tests/test_config_models.py`:

```python
def test_inference_params_temperature_range():
    """Test temperature must be 0-2."""
    params = InferenceParams(default_temperature=1.0)
    assert params.default_temperature == 1.0

    with pytest.raises(ValidationError):
        InferenceParams(default_temperature=3.0)


def test_inference_params_top_p_range():
    """Test top_p must be 0-1."""
    params = InferenceParams(default_top_p=0.95)
    assert params.default_top_p == 0.95

    with pytest.raises(ValidationError):
        InferenceParams(default_top_p=1.5)


def test_model_config_creation():
    """Test creating a complete model config."""
    model = ModelConfig(
        id="test-model",
        name="Test Model",
        aliases=["test", "tm"],
        pricing=PricingConfig(input_per_million=5.0, output_per_million=25.0),
        inference_params=InferenceParams(default_temperature=0.1),
        full_id="provider.test-model-v1",
    )

    assert model.id == "test-model"
    assert model.name == "Test Model"
    assert "test" in model.aliases
    assert model.pricing.input_per_million == 5.0
    assert model.full_id == "provider.test-model-v1"


def test_bedrock_config_default_region():
    """Test Bedrock config has default region."""
    config = BedrockConfig(models=[])
    assert config.region == "us-west-2"


def test_azure_config_requires_fields():
    """Test Azure config requires endpoint, key, version."""
    with pytest.raises(ValidationError):
        AzureOpenAIConfig(models=[])
```

**Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_models.py -v`
Expected: All 8 tests PASS

**Step 8: Commit**

```bash
git add codereview/config/ tests/test_config_models.py
git commit -m "feat: add Pydantic data models for config validation"
```

---

### Task 1.3: Create YAML Configuration File

**Files:**
- Create: `codereview/config/models.yaml`

**Step 1: Create models.yaml with all Bedrock models**

Create `codereview/config/models.yaml`:

```yaml
# Model pricing and configuration
# Prices are per million tokens

providers:
  bedrock:
    region: us-west-2
    models:
      - id: opus
        full_id: global.anthropic.claude-opus-4-5-20251101-v1:0
        name: Claude Opus 4.5
        aliases: [claude-opus]
        pricing:
          input_per_million: 5.00
          output_per_million: 25.00
        inference_params:
          default_temperature: 0.1

      - id: sonnet
        full_id: global.anthropic.claude-sonnet-4-5-20250929-v1:0
        name: Claude Sonnet 4.5
        aliases: [claude-sonnet]
        pricing:
          input_per_million: 3.00
          output_per_million: 15.00
        inference_params:
          default_temperature: 0.1

      - id: haiku
        full_id: global.anthropic.claude-haiku-4-5-20251001-v1:0
        name: Claude Haiku 4.5
        aliases: [claude-haiku]
        pricing:
          input_per_million: 1.00
          output_per_million: 5.00
        inference_params:
          default_temperature: 0.1

      - id: minimax
        full_id: minimax.minimax-m2
        name: Minimax M2
        aliases: [minimax-m2]
        pricing:
          input_per_million: 0.30
          output_per_million: 1.20
        inference_params:
          default_temperature: 1.0
          default_top_p: 0.95
          default_top_k: 40
          max_output_tokens: 8192

      - id: mistral
        full_id: mistral.mistral-large-3-675b-instruct
        name: Mistral Large 3
        aliases: [mistral-large]
        pricing:
          input_per_million: 2.00
          output_per_million: 6.00
        inference_params:
          default_temperature: 0.1
          default_top_p: 0.5
          default_top_k: 5

      - id: kimi
        full_id: moonshot.kimi-k2-thinking
        name: Kimi K2 Thinking
        aliases: [kimi-k2]
        pricing:
          input_per_million: 0.50
          output_per_million: 2.00
        inference_params:
          default_temperature: 1.0
          max_output_tokens: 16000

      - id: qwen
        full_id: qwen.qwen3-coder-480b-a35b-v1:0
        name: Qwen3 Coder 480B
        aliases: [qwen-coder]
        pricing:
          input_per_million: 0.22
          output_per_million: 1.40
        inference_params:
          default_temperature: 0.7
          default_top_p: 0.8
          default_top_k: 20
          max_output_tokens: 65536

  azure_openai:
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    api_version: "2026-01-14"
    models:
      - id: gpt-5.2-codex
        deployment_name: gpt-5.2-codex
        name: GPT-5.2 Codex
        aliases: [gpt52, codex]
        pricing:
          input_per_million: 1.75
          output_per_million: 14.00
        inference_params:
          default_temperature: 0.0
          default_top_p: 0.95
          max_output_tokens: 16000
```

**Step 2: Verify YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('codereview/config/models.yaml'))"`
Expected: No errors

**Step 3: Commit**

```bash
git add codereview/config/models.yaml
git commit -m "feat: add models.yaml with all Bedrock models and Azure OpenAI"
```

---

### Task 1.4: Create Configuration Loader

**Files:**
- Create: `codereview/config/loader.py`
- Create: `tests/test_config_loader.py`

**Step 1: Write test for ConfigLoader initialization**

Create `tests/test_config_loader.py`:

```python
"""Tests for configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest

from codereview.config.loader import ConfigLoader


@pytest.fixture
def test_yaml_content():
    """Test YAML configuration content."""
    return """
providers:
  bedrock:
    region: us-west-2
    models:
      - id: test-opus
        full_id: test.opus
        name: Test Opus
        aliases: [opus]
        pricing:
          input_per_million: 5.0
          output_per_million: 25.0
        inference_params:
          default_temperature: 0.1

  azure_openai:
    endpoint: https://test.openai.azure.com
    api_key: test-key-123
    api_version: "2026-01-14"
    models:
      - id: test-gpt
        deployment_name: test-gpt
        name: Test GPT
        aliases: [gpt]
        pricing:
          input_per_million: 1.75
          output_per_million: 14.0
        inference_params:
          default_temperature: 0.0
          default_top_p: 0.95
"""


@pytest.fixture
def test_config_file(test_yaml_content):
    """Create temporary config file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(test_yaml_content)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink()


def test_config_loader_initialization(test_config_file):
    """Test ConfigLoader loads YAML file."""
    loader = ConfigLoader(test_config_file)
    assert loader.config_path == Path(test_config_file)


def test_config_loader_default_path():
    """Test ConfigLoader uses default path if not provided."""
    loader = ConfigLoader()
    assert loader.config_path.name == "models.yaml"
    assert "config" in str(loader.config_path)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_loader.py::test_config_loader_initialization -v`
Expected: FAIL with "No module named 'codereview.config.loader'"

**Step 3: Write minimal ConfigLoader implementation**

Create `codereview/config/loader.py`:

```python
"""Load and parse model configuration from YAML."""

import os
from pathlib import Path
from typing import Any

import yaml

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    ModelConfig,
    ProviderConfig,
)


class ConfigLoader:
    """Loads model configurations from YAML file."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to models.yaml (defaults to package config/)
        """
        if config_path is None:
            # Default to config/models.yaml in package directory
            package_dir = Path(__file__).parent.parent
            config_path = package_dir / "config" / "models.yaml"

        self.config_path = Path(config_path)
        self._models_by_id: dict[str, tuple[str, ModelConfig]] = {}
        self._models_by_alias: dict[str, tuple[str, ModelConfig]] = {}
        self._provider_configs: dict[str, BedrockConfig | AzureOpenAIConfig] = {}

        self._load_config()

    def _load_config(self) -> None:
        """Load and parse YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # Expand environment variables
        raw_config = self._expand_env_vars(raw_config)

        # Parse provider configurations
        providers = raw_config.get("providers", {})

        if "bedrock" in providers:
            self._provider_configs["bedrock"] = BedrockConfig(**providers["bedrock"])
            self._register_models("bedrock", self._provider_configs["bedrock"])

        if "azure_openai" in providers:
            self._provider_configs["azure_openai"] = AzureOpenAIConfig(
                **providers["azure_openai"]
            )
            self._register_models("azure_openai", self._provider_configs["azure_openai"])

    def _expand_env_vars(self, config: Any) -> Any:
        """Recursively expand ${VAR} environment variables."""
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.environ.get(var_name, config)
        return config

    def _register_models(
        self, provider_name: str, provider_config: ProviderConfig
    ) -> None:
        """Register models and aliases for quick lookup."""
        for model in provider_config.models:
            # Register by primary ID
            self._models_by_id[model.id] = (provider_name, model)

            # Register aliases
            for alias in model.aliases:
                self._models_by_alias[alias] = (provider_name, model)

    def resolve_model(self, model_name: str) -> tuple[str, ModelConfig]:
        """
        Resolve model name to provider and config.

        Args:
            model_name: Short name or alias

        Returns:
            Tuple of (provider_name, ModelConfig)

        Raises:
            ValueError: If model not found
        """
        # Try primary ID first
        if model_name in self._models_by_id:
            return self._models_by_id[model_name]

        # Try aliases
        if model_name in self._models_by_alias:
            return self._models_by_alias[model_name]

        raise ValueError(f"Unknown model: {model_name}")

    def get_provider_config(
        self, provider_name: str
    ) -> BedrockConfig | AzureOpenAIConfig:
        """Get provider-specific configuration."""
        if provider_name not in self._provider_configs:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self._provider_configs[provider_name]

    def list_models(self) -> list[tuple[str, str, ModelConfig]]:
        """
        List all available models.

        Returns:
            List of (provider, model_id, ModelConfig) tuples
        """
        return [
            (provider, model.id, model)
            for provider, config in self._provider_configs.items()
            for model in config.models
        ]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: 2 tests PASS

**Step 5: Add more loader tests**

Add to `tests/test_config_loader.py`:

```python
def test_resolve_model_by_id(test_config_file):
    """Test resolving model by primary ID."""
    loader = ConfigLoader(test_config_file)
    provider, model = loader.resolve_model("test-opus")

    assert provider == "bedrock"
    assert model.id == "test-opus"
    assert model.name == "Test Opus"


def test_resolve_model_by_alias(test_config_file):
    """Test resolving model by alias."""
    loader = ConfigLoader(test_config_file)
    provider, model = loader.resolve_model("opus")

    assert provider == "bedrock"
    assert model.id == "test-opus"


def test_resolve_model_not_found(test_config_file):
    """Test error when model not found."""
    loader = ConfigLoader(test_config_file)

    with pytest.raises(ValueError, match="Unknown model: nonexistent"):
        loader.resolve_model("nonexistent")


def test_get_provider_config_bedrock(test_config_file):
    """Test getting Bedrock provider config."""
    loader = ConfigLoader(test_config_file)
    config = loader.get_provider_config("bedrock")

    from codereview.config.models import BedrockConfig

    assert isinstance(config, BedrockConfig)
    assert config.region == "us-west-2"


def test_get_provider_config_azure(test_config_file):
    """Test getting Azure OpenAI provider config."""
    loader = ConfigLoader(test_config_file)
    config = loader.get_provider_config("azure_openai")

    from codereview.config.models import AzureOpenAIConfig

    assert isinstance(config, AzureOpenAIConfig)
    assert config.api_version == "2026-01-14"


def test_list_models(test_config_file):
    """Test listing all models."""
    loader = ConfigLoader(test_config_file)
    models = loader.list_models()

    assert len(models) == 2
    providers = [m[0] for m in models]
    assert "bedrock" in providers
    assert "azure_openai" in providers


def test_env_var_expansion():
    """Test environment variable expansion in config."""
    yaml_with_env = """
providers:
  azure_openai:
    endpoint: "${TEST_ENDPOINT}"
    api_key: "${TEST_API_KEY}"
    api_version: "2026-01-14"
    models: []
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_with_env)
        config_path = f.name

    # Set env vars
    os.environ["TEST_ENDPOINT"] = "https://test.example.com"
    os.environ["TEST_API_KEY"] = "secret-key-123"

    loader = ConfigLoader(config_path)
    config = loader.get_provider_config("azure_openai")

    assert config.endpoint == "https://test.example.com"
    assert config.api_key == "secret-key-123"

    # Cleanup
    Path(config_path).unlink()
    del os.environ["TEST_ENDPOINT"]
    del os.environ["TEST_API_KEY"]
```

**Step 6: Run all tests to verify they pass**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: All 10 tests PASS

**Step 7: Commit**

```bash
git add codereview/config/loader.py tests/test_config_loader.py
git commit -m "feat: add ConfigLoader with YAML parsing and env var expansion"
```

---

## Phase 2: Provider Abstraction Layer

### Task 2.1: Create Provider Base Class

**Files:**
- Create: `codereview/providers/__init__.py`
- Create: `codereview/providers/base.py`
- Create: `tests/test_provider_base.py`

**Step 1: Write test for provider interface**

Create `tests/test_provider_base.py`:

```python
"""Tests for provider base class."""

from unittest.mock import Mock

import pytest

from codereview.config.models import InferenceParams, ModelConfig, PricingConfig
from codereview.providers.base import ModelProvider


class ConcreteProvider(ModelProvider):
    """Concrete provider for testing."""

    def analyze_batch(self, batch_number, total_batches, files_content, max_retries=3):
        return Mock()

    def get_model_display_name(self):
        return "Test Provider"


def test_provider_initialization():
    """Test provider initializes with model config."""
    model_config = ModelConfig(
        id="test",
        name="Test Model",
        pricing=PricingConfig(input_per_million=5.0, output_per_million=25.0),
        inference_params=InferenceParams(default_temperature=0.1),
    )

    provider = ConcreteProvider(model_config, None)

    assert provider.model_config == model_config
    assert provider.temperature == 0.1


def test_provider_temperature_override():
    """Test provider respects temperature override."""
    model_config = ModelConfig(
        id="test",
        name="Test Model",
        pricing=PricingConfig(input_per_million=5.0, output_per_million=25.0),
        inference_params=InferenceParams(default_temperature=0.1),
    )

    provider = ConcreteProvider(model_config, None, temperature=0.5)

    assert provider.temperature == 0.5


def test_provider_token_tracking():
    """Test provider tracks tokens."""
    model_config = ModelConfig(
        id="test",
        name="Test Model",
        pricing=PricingConfig(input_per_million=5.0, output_per_million=25.0),
    )

    provider = ConcreteProvider(model_config, None)

    assert provider.total_input_tokens == 0
    assert provider.total_output_tokens == 0

    provider.total_input_tokens = 1000
    provider.total_output_tokens = 500

    assert provider.total_input_tokens == 1000
    assert provider.total_output_tokens == 500


def test_provider_reset_state():
    """Test provider resets state."""
    model_config = ModelConfig(
        id="test",
        name="Test Model",
        pricing=PricingConfig(input_per_million=5.0, output_per_million=25.0),
    )

    provider = ConcreteProvider(model_config, None)
    provider.total_input_tokens = 1000
    provider.total_output_tokens = 500
    provider.skipped_files = ["file.py"]

    provider.reset_state()

    assert provider.total_input_tokens == 0
    assert provider.total_output_tokens == 0
    assert provider.skipped_files == []


def test_provider_estimate_cost():
    """Test provider estimates cost correctly."""
    model_config = ModelConfig(
        id="test",
        name="Test Model",
        pricing=PricingConfig(input_per_million=5.0, output_per_million=25.0),
    )

    provider = ConcreteProvider(model_config, None)
    provider.total_input_tokens = 1_000_000  # 1M tokens
    provider.total_output_tokens = 500_000  # 0.5M tokens

    cost = provider.estimate_cost()

    assert cost["input_tokens"] == 1_000_000
    assert cost["output_tokens"] == 500_000
    assert cost["input_cost"] == 5.0  # 1M * $5/M
    assert cost["output_cost"] == 12.5  # 0.5M * $25/M
    assert cost["total_cost"] == 17.5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_base.py::test_provider_initialization -v`
Expected: FAIL with "No module named 'codereview.providers'"

**Step 3: Create providers package**

Create `codereview/providers/__init__.py`:

```python
"""Provider abstraction layer for multi-provider support."""

from codereview.providers.base import ModelProvider

__all__ = ["ModelProvider"]
```

**Step 4: Write minimal provider base implementation**

Create `codereview/providers/base.py`:

```python
"""Abstract base class for model providers."""

from abc import ABC, abstractmethod
from typing import Any

from codereview.config.models import ModelConfig
from codereview.models import CodeReviewReport


class ModelProvider(ABC):
    """Abstract interface for LLM providers."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: Any,
        temperature: float | None = None,
    ):
        """
        Initialize provider.

        Args:
            model_config: Model-specific configuration
            provider_config: Provider-specific settings (BedrockConfig/AzureOpenAIConfig)
            temperature: Override default temperature
        """
        self.model_config = model_config
        self.provider_config = provider_config

        # Use override or model default or fallback
        self.temperature = (
            temperature
            if temperature is not None
            else (
                model_config.inference_params.default_temperature
                if model_config.inference_params.default_temperature is not None
                else 0.1
            )
        )

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.skipped_files: list[str] = []

    @abstractmethod
    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """
        Analyze a batch of files.

        Args:
            batch_number: Current batch index
            total_batches: Total number of batches
            files_content: Dict mapping file paths to content
            max_retries: Max retry attempts for transient errors

        Returns:
            CodeReviewReport from the model
        """
        pass

    @abstractmethod
    def get_model_display_name(self) -> str:
        """Get human-readable model name for display."""
        pass

    def reset_state(self) -> None:
        """Reset token counters and skipped files."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.skipped_files = []

    def estimate_cost(self) -> dict[str, float]:
        """Calculate cost based on token usage."""
        input_cost = (
            self.total_input_tokens / 1_000_000
        ) * self.model_config.pricing.input_per_million

        output_cost = (
            self.total_output_tokens / 1_000_000
        ) * self.model_config.pricing.output_per_million

        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(input_cost + output_cost, 4),
        }
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider_base.py -v`
Expected: All 6 tests PASS

**Step 6: Commit**

```bash
git add codereview/providers/ tests/test_provider_base.py
git commit -m "feat: add ModelProvider abstract base class"
```

---

### Task 2.2: Create Bedrock Provider

**Files:**
- Create: `codereview/providers/bedrock.py`
- Create: `tests/test_bedrock_provider.py`

**Step 1: Write test for Bedrock provider**

Create `tests/test_bedrock_provider.py`:

```python
"""Tests for Bedrock provider."""

from unittest.mock import Mock, patch

import pytest

from codereview.config.models import (
    BedrockConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport


@pytest.fixture
def bedrock_model_config():
    """Sample Bedrock model config."""
    return ModelConfig(
        id="test-opus",
        name="Test Opus",
        full_id="test.opus",
        pricing=PricingConfig(input_per_million=5.0, output_per_million=25.0),
        inference_params=InferenceParams(
            default_temperature=0.1, default_top_p=0.95, max_output_tokens=16000
        ),
    )


@pytest.fixture
def bedrock_provider_config():
    """Sample Bedrock provider config."""
    return BedrockConfig(region="us-west-2", models=[])


def test_bedrock_provider_initialization(bedrock_model_config, bedrock_provider_config):
    """Test Bedrock provider initializes correctly."""
    from codereview.providers.bedrock import BedrockProvider

    provider = BedrockProvider(
        bedrock_model_config, bedrock_provider_config, temperature=0.2
    )

    assert provider.model_config == bedrock_model_config
    assert provider.region == "us-west-2"
    assert provider.temperature == 0.2


def test_bedrock_provider_display_name(bedrock_model_config, bedrock_provider_config):
    """Test Bedrock provider display name."""
    from codereview.providers.bedrock import BedrockProvider

    provider = BedrockProvider(bedrock_model_config, bedrock_provider_config)

    assert provider.get_model_display_name() == "Test Opus (Bedrock)"


@patch("codereview.providers.bedrock.ChatBedrockConverse")
def test_bedrock_provider_analyze_batch(
    mock_bedrock, bedrock_model_config, bedrock_provider_config
):
    """Test Bedrock provider can analyze a batch."""
    from codereview.providers.bedrock import BedrockProvider

    # Mock LangChain response
    mock_report = Mock(spec=CodeReviewReport)
    mock_report.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    mock_model = Mock()
    mock_model.invoke.return_value = mock_report

    mock_bedrock_instance = Mock()
    mock_bedrock_instance.with_structured_output.return_value = mock_model
    mock_bedrock.return_value = mock_bedrock_instance

    provider = BedrockProvider(bedrock_model_config, bedrock_provider_config)

    result = provider.analyze_batch(
        batch_number=1,
        total_batches=1,
        files_content={"test.py": "print('hello')"},
    )

    assert result == mock_report
    assert provider.total_input_tokens == 100
    assert provider.total_output_tokens == 50


@patch("codereview.providers.bedrock.ChatBedrockConverse")
def test_bedrock_provider_retry_on_throttling(
    mock_bedrock, bedrock_model_config, bedrock_provider_config
):
    """Test Bedrock provider retries on throttling."""
    from botocore.exceptions import ClientError

    from codereview.providers.bedrock import BedrockProvider

    # Mock throttling error then success
    mock_report = Mock(spec=CodeReviewReport)
    mock_report.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    throttling_error = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "InvokeModel",
    )

    mock_model = Mock()
    mock_model.invoke.side_effect = [throttling_error, mock_report]

    mock_bedrock_instance = Mock()
    mock_bedrock_instance.with_structured_output.return_value = mock_model
    mock_bedrock.return_value = mock_bedrock_instance

    provider = BedrockProvider(bedrock_model_config, bedrock_provider_config)

    with patch("time.sleep"):  # Speed up test
        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"test.py": "print('hello')"},
        )

    assert result == mock_report
    assert mock_model.invoke.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bedrock_provider.py::test_bedrock_provider_initialization -v`
Expected: FAIL with "No module named 'codereview.providers.bedrock'"

**Step 3: Write Bedrock provider implementation**

Create `codereview/providers/bedrock.py`:

```python
"""AWS Bedrock provider implementation."""

import time
from typing import Any

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse

from codereview.config import SYSTEM_PROMPT
from codereview.config.models import BedrockConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider


class BedrockProvider(ModelProvider):
    """AWS Bedrock implementation."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: BedrockConfig,
        temperature: float | None = None,
    ):
        super().__init__(model_config, provider_config, temperature)
        self.region = provider_config.region
        self.model = self._create_model()

    def _create_model(self) -> Any:
        """Create LangChain model with structured output."""
        # Build inference parameters
        inference_config: dict[str, Any] = {"temperature": self.temperature}

        params = self.model_config.inference_params
        if params.default_top_p is not None:
            inference_config["topP"] = params.default_top_p
        if params.default_top_k is not None:
            inference_config["topK"] = params.default_top_k
        if params.max_output_tokens is not None:
            inference_config["maxTokens"] = params.max_output_tokens

        # Create Bedrock client
        bedrock_model = ChatBedrockConverse(
            model=self.model_config.full_id,
            region_name=self.region,
            **inference_config,
        )

        # Attach structured output parser
        return bedrock_model.with_structured_output(CodeReviewReport)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """Analyze batch with retry logic for throttling."""
        prompt = self._prepare_batch_context(
            batch_number, total_batches, files_content
        )

        for attempt in range(max_retries):
            try:
                result = self.model.invoke(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )

                # Track tokens (Bedrock returns usage metadata)
                if hasattr(result, "usage_metadata"):
                    self.total_input_tokens += result.usage_metadata.get(
                        "input_tokens", 0
                    )
                    self.total_output_tokens += result.usage_metadata.get(
                        "output_tokens", 0
                    )

                return result

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")

                # Retry on throttling
                if error_code in ["ThrottlingException", "TooManyRequestsException"]:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        time.sleep(wait_time)
                        continue

                # Re-raise other errors
                raise

        raise Exception(f"Max retries ({max_retries}) exceeded")

    def _prepare_batch_context(
        self, batch_number: int, total_batches: int, files_content: dict[str, str]
    ) -> str:
        """Build prompt for batch analysis."""
        context_parts = [
            f"Batch {batch_number}/{total_batches}\n",
            f"Analyzing {len(files_content)} files:\n\n",
        ]

        for file_path, content in files_content.items():
            context_parts.append(f"=== {file_path} ===\n{content}\n\n")

        return "".join(context_parts)

    def get_model_display_name(self) -> str:
        return f"{self.model_config.name} (Bedrock)"
```

**Step 4: Update config imports**

Add to `codereview/config/__init__.py`:

```python
from codereview.config import SYSTEM_PROMPT  # Import from old config.py
```

Actually, we need to import from the old location. Update the import in `bedrock.py`:

```python
from codereview.config import SYSTEM_PROMPT  # This imports from codereview/config.py (old file)
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_bedrock_provider.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add codereview/providers/bedrock.py tests/test_bedrock_provider.py
git commit -m "feat: add BedrockProvider implementation"
```

---

### Task 2.3: Create Azure OpenAI Provider

**Files:**
- Create: `codereview/providers/azure_openai.py`
- Create: `tests/test_azure_provider.py`

**Step 1: Write test for Azure provider**

Create `tests/test_azure_provider.py`:

```python
"""Tests for Azure OpenAI provider."""

from unittest.mock import Mock, patch

import pytest

from codereview.config.models import (
    AzureOpenAIConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport


@pytest.fixture
def azure_model_config():
    """Sample Azure model config."""
    return ModelConfig(
        id="test-gpt",
        name="Test GPT",
        deployment_name="test-gpt-deployment",
        pricing=PricingConfig(input_per_million=1.75, output_per_million=14.0),
        inference_params=InferenceParams(
            default_temperature=0.0, default_top_p=0.95, max_output_tokens=16000
        ),
    )


@pytest.fixture
def azure_provider_config():
    """Sample Azure provider config."""
    return AzureOpenAIConfig(
        endpoint="https://test.openai.azure.com",
        api_key="test-key-123",
        api_version="2026-01-14",
        models=[],
    )


def test_azure_provider_initialization(azure_model_config, azure_provider_config):
    """Test Azure provider initializes correctly."""
    from codereview.providers.azure_openai import AzureOpenAIProvider

    provider = AzureOpenAIProvider(
        azure_model_config, azure_provider_config, temperature=0.1
    )

    assert provider.model_config == azure_model_config
    assert provider.temperature == 0.1


def test_azure_provider_display_name(azure_model_config, azure_provider_config):
    """Test Azure provider display name."""
    from codereview.providers.azure_openai import AzureOpenAIProvider

    provider = AzureOpenAIProvider(azure_model_config, azure_provider_config)

    assert provider.get_model_display_name() == "Test GPT (Azure OpenAI)"


@patch("codereview.providers.azure_openai.AzureChatOpenAI")
def test_azure_provider_analyze_batch(
    mock_azure, azure_model_config, azure_provider_config
):
    """Test Azure provider can analyze a batch."""
    from codereview.providers.azure_openai import AzureOpenAIProvider

    # Mock LangChain response
    mock_report = Mock(spec=CodeReviewReport)
    mock_report.response_metadata = {
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
    }

    mock_model = Mock()
    mock_model.invoke.return_value = mock_report

    mock_azure_instance = Mock()
    mock_azure_instance.with_structured_output.return_value = mock_model
    mock_azure.return_value = mock_azure_instance

    provider = AzureOpenAIProvider(azure_model_config, azure_provider_config)

    result = provider.analyze_batch(
        batch_number=1,
        total_batches=1,
        files_content={"test.py": "print('hello')"},
    )

    assert result == mock_report
    assert provider.total_input_tokens == 100
    assert provider.total_output_tokens == 50


@patch("codereview.providers.azure_openai.AzureChatOpenAI")
def test_azure_provider_retry_on_rate_limit(
    mock_azure, azure_model_config, azure_provider_config
):
    """Test Azure provider retries on rate limit."""
    from openai import RateLimitError

    from codereview.providers.azure_openai import AzureOpenAIProvider

    # Mock rate limit error then success
    mock_report = Mock(spec=CodeReviewReport)
    mock_report.response_metadata = {
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
    }

    rate_limit_error = RateLimitError(
        message="Rate limit exceeded",
        response=Mock(status_code=429),
        body=None,
    )

    mock_model = Mock()
    mock_model.invoke.side_effect = [rate_limit_error, mock_report]

    mock_azure_instance = Mock()
    mock_azure_instance.with_structured_output.return_value = mock_model
    mock_azure.return_value = mock_azure_instance

    provider = AzureOpenAIProvider(azure_model_config, azure_provider_config)

    with patch("time.sleep"):  # Speed up test
        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"test.py": "print('hello')"},
        )

    assert result == mock_report
    assert mock_model.invoke.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_azure_provider.py::test_azure_provider_initialization -v`
Expected: FAIL with "No module named 'codereview.providers.azure_openai'"

**Step 3: Write Azure provider implementation**

Create `codereview/providers/azure_openai.py`:

```python
"""Azure OpenAI provider implementation."""

import time
from typing import Any

from langchain_openai import AzureChatOpenAI
from openai import RateLimitError

from codereview.config import SYSTEM_PROMPT
from codereview.config.models import AzureOpenAIConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider


class AzureOpenAIProvider(ModelProvider):
    """Azure OpenAI implementation."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: AzureOpenAIConfig,
        temperature: float | None = None,
    ):
        super().__init__(model_config, provider_config, temperature)
        self.model = self._create_model()

    def _create_model(self) -> Any:
        """Create LangChain Azure OpenAI model."""
        params = self.model_config.inference_params

        azure_model = AzureChatOpenAI(
            azure_endpoint=self.provider_config.endpoint,
            api_key=self.provider_config.api_key,
            api_version=self.provider_config.api_version,
            deployment_name=self.model_config.deployment_name,
            temperature=self.temperature,
            top_p=params.default_top_p,
            max_tokens=params.max_output_tokens,
        )

        return azure_model.with_structured_output(CodeReviewReport)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """Analyze batch with retry logic."""
        prompt = self._prepare_batch_context(
            batch_number, total_batches, files_content
        )

        for attempt in range(max_retries):
            try:
                result = self.model.invoke(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )

                # Track tokens from response metadata
                if hasattr(result, "response_metadata"):
                    usage = result.response_metadata.get("token_usage", {})
                    self.total_input_tokens += usage.get("prompt_tokens", 0)
                    self.total_output_tokens += usage.get("completion_tokens", 0)

                return result

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue
                raise

        raise Exception(f"Max retries ({max_retries}) exceeded")

    def _prepare_batch_context(
        self, batch_number: int, total_batches: int, files_content: dict[str, str]
    ) -> str:
        """Build prompt for batch analysis."""
        context_parts = [
            f"Batch {batch_number}/{total_batches}\n",
            f"Analyzing {len(files_content)} files:\n\n",
        ]

        for file_path, content in files_content.items():
            context_parts.append(f"=== {file_path} ===\n{content}\n\n")

        return "".join(context_parts)

    def get_model_display_name(self) -> str:
        return f"{self.model_config.name} (Azure OpenAI)"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_azure_provider.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add codereview/providers/azure_openai.py tests/test_azure_provider.py
git commit -m "feat: add AzureOpenAIProvider implementation"
```

---

### Task 2.4: Create Provider Factory

**Files:**
- Create: `codereview/providers/factory.py`
- Create: `tests/test_provider_factory.py`

**Step 1: Write test for provider factory**

Create `tests/test_provider_factory.py`:

```python
"""Tests for provider factory."""

import tempfile
from pathlib import Path

import pytest

from codereview.config.loader import ConfigLoader
from codereview.providers.azure_openai import AzureOpenAIProvider
from codereview.providers.bedrock import BedrockProvider


@pytest.fixture
def test_config_file():
    """Create test config file."""
    yaml_content = """
providers:
  bedrock:
    region: us-west-2
    models:
      - id: test-opus
        full_id: test.opus
        name: Test Opus
        aliases: [opus]
        pricing:
          input_per_million: 5.0
          output_per_million: 25.0
        inference_params:
          default_temperature: 0.1

  azure_openai:
    endpoint: https://test.openai.azure.com
    api_key: test-key
    api_version: "2026-01-14"
    models:
      - id: test-gpt
        deployment_name: test-gpt
        name: Test GPT
        aliases: [gpt]
        pricing:
          input_per_million: 1.75
          output_per_million: 14.0
        inference_params:
          default_temperature: 0.0
          default_top_p: 0.95
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        config_path = f.name

    yield config_path

    Path(config_path).unlink()


def test_factory_creates_bedrock_provider(test_config_file):
    """Test factory creates Bedrock provider for Bedrock models."""
    from codereview.providers.factory import ProviderFactory

    loader = ConfigLoader(test_config_file)
    factory = ProviderFactory(loader)

    provider = factory.create_provider("test-opus")

    assert isinstance(provider, BedrockProvider)
    assert provider.model_config.id == "test-opus"


def test_factory_creates_azure_provider(test_config_file):
    """Test factory creates Azure provider for Azure models."""
    from codereview.providers.factory import ProviderFactory

    loader = ConfigLoader(test_config_file)
    factory = ProviderFactory(loader)

    provider = factory.create_provider("test-gpt")

    assert isinstance(provider, AzureOpenAIProvider)
    assert provider.model_config.id == "test-gpt"


def test_factory_resolves_alias(test_config_file):
    """Test factory resolves model aliases."""
    from codereview.providers.factory import ProviderFactory

    loader = ConfigLoader(test_config_file)
    factory = ProviderFactory(loader)

    provider_opus = factory.create_provider("opus")
    provider_gpt = factory.create_provider("gpt")

    assert isinstance(provider_opus, BedrockProvider)
    assert isinstance(provider_gpt, AzureOpenAIProvider)


def test_factory_temperature_override(test_config_file):
    """Test factory respects temperature override."""
    from codereview.providers.factory import ProviderFactory

    loader = ConfigLoader(test_config_file)
    factory = ProviderFactory(loader)

    provider = factory.create_provider("test-opus", temperature=0.5)

    assert provider.temperature == 0.5


def test_factory_unknown_model_raises_error(test_config_file):
    """Test factory raises error for unknown model."""
    from codereview.providers.factory import ProviderFactory

    loader = ConfigLoader(test_config_file)
    factory = ProviderFactory(loader)

    with pytest.raises(ValueError, match="Unknown model: nonexistent"):
        factory.create_provider("nonexistent")


def test_factory_list_available_models(test_config_file):
    """Test factory lists available models."""
    from codereview.providers.factory import ProviderFactory

    loader = ConfigLoader(test_config_file)
    factory = ProviderFactory(loader)

    models = factory.list_available_models()

    assert len(models) == 2
    assert any(m["id"] == "test-opus" for m in models)
    assert any(m["id"] == "test-gpt" for m in models)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_factory.py::test_factory_creates_bedrock_provider -v`
Expected: FAIL with "No module named 'codereview.providers.factory'"

**Step 3: Write provider factory implementation**

Create `codereview/providers/factory.py`:

```python
"""Factory for creating model providers."""

from codereview.config.loader import ConfigLoader
from codereview.config.models import AzureOpenAIConfig, BedrockConfig
from codereview.providers.azure_openai import AzureOpenAIProvider
from codereview.providers.base import ModelProvider
from codereview.providers.bedrock import BedrockProvider


class ProviderFactory:
    """Factory for creating provider instances."""

    def __init__(self, config_loader: ConfigLoader | None = None):
        """
        Initialize factory.

        Args:
            config_loader: Optional config loader (creates default if not provided)
        """
        self.config_loader = config_loader or ConfigLoader()

    def create_provider(
        self,
        model_name: str,
        temperature: float | None = None,
    ) -> ModelProvider:
        """
        Create appropriate provider for the given model.

        Args:
            model_name: Model short name or alias
            temperature: Override default temperature

        Returns:
            Configured ModelProvider instance

        Raises:
            ValueError: If model not found or provider not supported
        """
        # Resolve model to provider and config
        provider_name, model_config = self.config_loader.resolve_model(model_name)
        provider_config = self.config_loader.get_provider_config(provider_name)

        # Create provider based on type
        if provider_name == "bedrock":
            if not isinstance(provider_config, BedrockConfig):
                raise ValueError(
                    f"Invalid config type for bedrock: {type(provider_config)}"
                )
            return BedrockProvider(model_config, provider_config, temperature)

        elif provider_name == "azure_openai":
            if not isinstance(provider_config, AzureOpenAIConfig):
                raise ValueError(
                    f"Invalid config type for azure_openai: {type(provider_config)}"
                )
            return AzureOpenAIProvider(model_config, provider_config, temperature)

        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    def list_available_models(self) -> list[dict[str, str]]:
        """
        List all available models across providers.

        Returns:
            List of dicts with model info (id, name, provider)
        """
        models = []
        for provider, model_id, model_config in self.config_loader.list_models():
            models.append(
                {
                    "id": model_id,
                    "name": model_config.name,
                    "provider": provider,
                    "aliases": ", ".join(model_config.aliases)
                    if model_config.aliases
                    else "",
                }
            )
        return models
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_provider_factory.py -v`
Expected: All 6 tests PASS

**Step 5: Update providers package exports**

Update `codereview/providers/__init__.py`:

```python
"""Provider abstraction layer for multi-provider support."""

from codereview.providers.azure_openai import AzureOpenAIProvider
from codereview.providers.base import ModelProvider
from codereview.providers.bedrock import BedrockProvider
from codereview.providers.factory import ProviderFactory

__all__ = [
    "AzureOpenAIProvider",
    "BedrockProvider",
    "ModelProvider",
    "ProviderFactory",
]
```

**Step 6: Commit**

```bash
git add codereview/providers/factory.py codereview/providers/__init__.py tests/test_provider_factory.py
git commit -m "feat: add ProviderFactory for auto-detection"
```

---

## Phase 3: Refactor CodeAnalyzer (Backward Compatible)

### Task 3.1: Refactor CodeAnalyzer to Use Providers

**Files:**
- Modify: `codereview/analyzer.py`
- Modify: `tests/test_analyzer.py`

**Step 1: Back up old analyzer tests**

Run: `cp tests/test_analyzer.py tests/test_analyzer_backup.py`

**Step 2: Write new test for refactored analyzer**

Replace content of `tests/test_analyzer.py`:

```python
"""Tests for code analyzer with provider system."""

from unittest.mock import Mock, patch

import pytest

from codereview.analyzer import CodeAnalyzer
from codereview.models import CodeReviewReport


@patch("codereview.providers.factory.ProviderFactory")
def test_analyzer_initialization_with_model_name(mock_factory_class):
    """Test analyzer initializes with model name."""
    mock_factory = Mock()
    mock_provider = Mock()
    mock_factory.create_provider.return_value = mock_provider
    mock_factory_class.return_value = mock_factory

    analyzer = CodeAnalyzer(model_name="opus")

    mock_factory.create_provider.assert_called_once_with("opus", None)
    assert analyzer.provider == mock_provider


@patch("codereview.providers.factory.ProviderFactory")
def test_analyzer_with_temperature_override(mock_factory_class):
    """Test analyzer passes temperature to provider."""
    mock_factory = Mock()
    mock_provider = Mock()
    mock_factory.create_provider.return_value = mock_provider
    mock_factory_class.return_value = mock_factory

    analyzer = CodeAnalyzer(model_name="sonnet", temperature=0.5)

    mock_factory.create_provider.assert_called_once_with("sonnet", 0.5)


@patch("codereview.providers.factory.ProviderFactory")
def test_analyzer_analyze_batch(mock_factory_class):
    """Test analyzer delegates to provider."""
    mock_factory = Mock()
    mock_provider = Mock()
    mock_report = Mock(spec=CodeReviewReport)
    mock_provider.analyze_batch.return_value = mock_report
    mock_factory.create_provider.return_value = mock_provider
    mock_factory_class.return_value = mock_factory

    analyzer = CodeAnalyzer(model_name="opus")
    result = analyzer.analyze_batch(
        batch_number=1,
        total_batches=1,
        files_content={"test.py": "print('hello')"},
    )

    assert result == mock_report
    mock_provider.analyze_batch.assert_called_once()


@patch("codereview.providers.factory.ProviderFactory")
def test_analyzer_get_model_display_name(mock_factory_class):
    """Test analyzer delegates display name to provider."""
    mock_factory = Mock()
    mock_provider = Mock()
    mock_provider.get_model_display_name.return_value = "Test Model"
    mock_factory.create_provider.return_value = mock_provider
    mock_factory_class.return_value = mock_factory

    analyzer = CodeAnalyzer(model_name="opus")
    display_name = analyzer.get_model_display_name()

    assert display_name == "Test Model"


@patch("codereview.providers.factory.ProviderFactory")
def test_analyzer_reset_state(mock_factory_class):
    """Test analyzer resets provider state."""
    mock_factory = Mock()
    mock_provider = Mock()
    mock_factory.create_provider.return_value = mock_provider
    mock_factory_class.return_value = mock_factory

    analyzer = CodeAnalyzer(model_name="opus")
    analyzer.reset_state()

    mock_provider.reset_state.assert_called_once()


@patch("codereview.providers.factory.ProviderFactory")
def test_analyzer_estimate_cost(mock_factory_class):
    """Test analyzer delegates cost estimation to provider."""
    mock_factory = Mock()
    mock_provider = Mock()
    mock_provider.estimate_cost.return_value = {"total_cost": 1.23}
    mock_factory.create_provider.return_value = mock_provider
    mock_factory_class.return_value = mock_factory

    analyzer = CodeAnalyzer(model_name="opus")
    cost = analyzer.estimate_cost()

    assert cost == {"total_cost": 1.23}


@patch("codereview.providers.factory.ProviderFactory")
def test_analyzer_token_properties(mock_factory_class):
    """Test analyzer exposes provider token properties."""
    mock_factory = Mock()
    mock_provider = Mock()
    mock_provider.total_input_tokens = 1000
    mock_provider.total_output_tokens = 500
    mock_provider.skipped_files = ["file.py"]
    mock_factory.create_provider.return_value = mock_provider
    mock_factory_class.return_value = mock_factory

    analyzer = CodeAnalyzer(model_name="opus")

    assert analyzer.total_input_tokens == 1000
    assert analyzer.total_output_tokens == 500
    assert analyzer.skipped_files == ["file.py"]
```

**Step 3: Run new tests to verify they fail**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: Tests fail (old implementation doesn't match new tests)

**Step 4: Refactor CodeAnalyzer**

Replace content of `codereview/analyzer.py`:

```python
"""Code analyzer using provider abstraction."""

from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider
from codereview.providers.factory import ProviderFactory


class CodeAnalyzer:
    """Analyzes code using configured model provider."""

    def __init__(
        self,
        model_name: str = "opus",
        temperature: float | None = None,
        provider_factory: ProviderFactory | None = None,
    ):
        """
        Initialize analyzer with model provider.

        Args:
            model_name: Model short name or alias (e.g., 'opus', 'gpt-5.2-codex')
            temperature: Override default temperature
            provider_factory: Optional factory (creates default if not provided)
        """
        self.model_name = model_name
        self.factory = provider_factory or ProviderFactory()

        # Create provider instance
        self.provider: ModelProvider = self.factory.create_provider(
            model_name, temperature
        )

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """
        Analyze a batch of files.

        Args:
            batch_number: Current batch index (1-based)
            total_batches: Total number of batches
            files_content: Dict mapping file paths to content
            max_retries: Max retry attempts for transient errors

        Returns:
            CodeReviewReport from the model
        """
        return self.provider.analyze_batch(
            batch_number, total_batches, files_content, max_retries
        )

    def get_model_display_name(self) -> str:
        """Get human-readable model name for display."""
        return self.provider.get_model_display_name()

    def reset_state(self) -> None:
        """Reset token counters and skipped files."""
        self.provider.reset_state()

    def estimate_cost(self) -> dict[str, float]:
        """Calculate cost based on token usage."""
        return self.provider.estimate_cost()

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens consumed."""
        return self.provider.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens consumed."""
        return self.provider.total_output_tokens

    @property
    def skipped_files(self) -> list[str]:
        """Files skipped during analysis."""
        return self.provider.skipped_files
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: All 8 tests PASS

**Step 6: Run ALL tests to verify nothing broke**

Run: `uv run pytest tests/ -v`
Expected: Some tests may fail that directly instantiate old CodeAnalyzer with model_id/region

**Step 7: Fix broken integration tests**

Update integration tests that use CodeAnalyzer to use new API. Check `tests/test_integration.py`:

Run: `uv run pytest tests/test_integration.py -v`

If tests fail, update them to use `model_name` instead of `model_id` and `region`.

**Step 8: Commit**

```bash
git add codereview/analyzer.py tests/test_analyzer.py
git commit -m "refactor: update CodeAnalyzer to use provider system"
```

---

## Phase 4: Update CLI

### Task 4.1: Add --list-models Flag

**Files:**
- Modify: `codereview/cli.py`

**Step 1: Add list_models helper function**

Add to `codereview/cli.py` before the `main` function:

```python
def display_available_models():
    """Display table of available models."""
    from codereview.providers.factory import ProviderFactory

    factory = ProviderFactory()
    models = factory.list_available_models()

    table = Table(title="Available Models", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Provider", style="yellow")
    table.add_column("Aliases", style="dim")

    for model in models:
        table.add_row(
            model["id"],
            model["name"],
            model["provider"],
            model["aliases"],
        )

    console.print(table)
    console.print("\n[dim]Usage: codereview ./src --model <id>[/dim]")
```

**Step 2: Add --list-models option to CLI**

Find the `@click.option` section and add:

```python
@click.option(
    "--list-models",
    is_flag=True,
    help="List all available models and exit",
)
```

**Step 3: Handle --list-models in main function**

At the start of the `main()` function, add:

```python
def main(
    path: str,
    list_models: bool,  # Add this parameter
    # ... other params
):
    """AI-powered code review tool supporting multiple providers."""

    # Handle --list-models flag
    if list_models:
        display_available_models()
        return 0

    # Rest of existing logic...
```

**Step 4: Test --list-models flag**

Run: `uv run codereview --list-models`
Expected: Displays table of all available models

**Step 5: Commit**

```bash
git add codereview/cli.py
git commit -m "feat: add --list-models flag to CLI"
```

---

### Task 4.2: Update CLI to Use model_name

**Files:**
- Modify: `codereview/cli.py`

**Step 1: Update --model option help text**

Find the `--model` option and update:

```python
@click.option(
    "--model",
    "-m",
    default="opus",
    help="Model to use (e.g., opus, sonnet, haiku, gpt-5.2-codex). Use --list-models to see all.",
)
```

**Step 2: Update CodeAnalyzer instantiation**

Find where `CodeAnalyzer` is created in `main()` and update to use `model_name`:

```python
try:
    # Create analyzer with auto-detected provider
    analyzer = CodeAnalyzer(model_name=model, temperature=temperature)

    if verbose:
        console.print(f"[cyan]Using model:[/cyan] {analyzer.get_model_display_name()}")

    # Rest of existing logic...

except ValueError as e:
    console.print(f"[red]Error:[/red] {e}")
    console.print("\nUse --list-models to see available models")
    return 1
```

**Step 3: Remove old --model-id and --aws-region options if they exist**

Search for these options and remove them (they're deprecated).

**Step 4: Test CLI with different models**

Run: `uv run codereview ./tests/fixtures/sample_code --model opus --dry-run`
Expected: Works with Bedrock Opus

**Step 5: Commit**

```bash
git add codereview/cli.py
git commit -m "feat: update CLI to use model_name with auto-detection"
```

---

## Phase 5: Documentation & Cleanup

### Task 5.1: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update architecture section**

Update the "Architecture" section to reflect new provider system:

```markdown
### Pipeline Flow
```
FileScanner → FileBatcher → CodeAnalyzer → ProviderFactory → BedrockProvider/AzureOpenAIProvider → LLM → Aggregation → Renderer
```

### Provider System
- **ConfigLoader**: Loads models.yaml, expands env vars, validates with Pydantic
- **ProviderFactory**: Auto-detects provider from model name
- **ModelProvider**: Abstract interface (BedrockProvider, AzureOpenAIProvider)
- **CodeAnalyzer**: Thin wrapper delegating to provider
```

**Step 2: Update CLI examples**

Update usage examples:

```markdown
### Running the Tool
```bash
# List available models
uv run codereview --list-models

# Use Bedrock Claude
uv run codereview /path/to/code --model opus
uv run codereview /path/to/code -m sonnet

# Use Azure OpenAI
uv run codereview /path/to/code --model gpt-5.2-codex

# With temperature override
uv run codereview ./src --model haiku --temperature 0.2
```
```

**Step 3: Add Azure OpenAI setup section**

Add new section:

```markdown
### Azure OpenAI Setup

Set environment variables:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
```

Verify access:

```bash
uv run codereview --list-models  # Should show gpt-5.2-codex
uv run codereview ./src --model gpt-5.2-codex --dry-run
```
```

**Step 4: Update "Adding New Models" section**

```markdown
### Adding New Models

**To add a Bedrock model:**
1. Edit `codereview/config/models.yaml`
2. Add entry under `providers.bedrock.models`
3. Include: id, full_id, name, aliases, pricing, inference_params
4. Done! No code changes needed.

**To add an Azure OpenAI model:**
1. Edit `codereview/config/models.yaml`
2. Add entry under `providers.azure_openai.models`
3. Include: id, deployment_name, name, aliases, pricing, inference_params
4. Ensure deployment exists in Azure

**To add a new provider:**
1. Create `codereview/providers/newprovider.py`
2. Implement `ModelProvider` interface
3. Add to `ProviderFactory.create_provider()`
4. Add provider config to models.yaml
5. Add tests
```

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for provider system"
```

---

### Task 5.2: Add README Section for Azure OpenAI

**Files:**
- Modify: `README.md` (if exists) or create it

**Step 1: Add Azure OpenAI section to README**

If README exists, add section. Otherwise, this can be skipped.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Azure OpenAI setup to README"
```

---

### Task 5.3: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --cov=codereview`
Expected: All tests PASS, good coverage

**Step 2: If any tests fail, fix them**

Update tests that rely on old API.

**Step 3: Final commit**

```bash
git add tests/
git commit -m "test: update tests for provider system"
```

---

### Task 5.4: Update pyproject.toml Metadata

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update project description**

Update description to mention multi-provider support:

```toml
description = "AI-powered code review tool with multi-provider support (AWS Bedrock, Azure OpenAI)"
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "docs: update project description for multi-provider"
```

---

### Task 5.5: Create Migration Guide

**Files:**
- Create: `docs/MIGRATION.md`

**Step 1: Create migration guide**

Create `docs/MIGRATION.md`:

```markdown
# Migration Guide: Provider System

## Overview

Version 2.0 introduces a provider system for multi-model support. The changes are **fully backward compatible**.

## For CLI Users

### No Changes Required

Existing commands work unchanged:

```bash
# Old way (still works)
codereview ./src --model opus

# New convenience
codereview --list-models  # Discover all models
codereview ./src --model gpt-5.2-codex  # Use Azure OpenAI
```

## For Library Users

### Old API (Deprecated but Functional)

```python
from codereview.analyzer import CodeAnalyzer

# This still works but is deprecated
analyzer = CodeAnalyzer(
    model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
    region="us-west-2"
)
```

### New API (Recommended)

```python
from codereview.analyzer import CodeAnalyzer

# Simple and provider-agnostic
analyzer = CodeAnalyzer(model_name="opus")
analyzer = CodeAnalyzer(model_name="gpt-5.2-codex")
```

## For Contributors

### Adding New Models

**Before:** Edit Python code, update constants, modify logic

**After:** Edit `codereview/config/models.yaml` only

### Adding New Providers

1. Implement `ModelProvider` interface
2. Add to `ProviderFactory`
3. Add provider config to models.yaml
4. Tests

See design doc for details.

## Breaking Changes

**None!** All existing code continues to work.

## Deprecation Timeline

- v2.0: Old API deprecated but functional
- v2.1: Deprecation warnings added
- v3.0: Old API removed (planned)
```

**Step 2: Commit**

```bash
git add docs/MIGRATION.md
git commit -m "docs: add migration guide for provider system"
```

---

## Final Verification

### Task 6.1: End-to-End Integration Test

**Step 1: Test with real Bedrock model (if AWS configured)**

Run: `uv run codereview ./tests/fixtures/sample_code --model haiku --verbose`
Expected: Completes successfully, shows model name

**Step 2: Test dry-run mode**

Run: `uv run codereview ./tests/fixtures/sample_code --model opus --dry-run`
Expected: Shows files and cost estimate

**Step 3: Test --list-models**

Run: `uv run codereview --list-models`
Expected: Shows all 8 models (7 Bedrock + 1 Azure)

**Step 4: Verify all tests pass**

Run: `uv run pytest tests/ -v`
Expected: 100+ tests PASS

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete model config extraction with multi-provider support

- Extract configs to models.yaml
- Add provider abstraction layer
- Support Azure OpenAI (GPT-5.2 Codex)
- Auto-detect provider from model name
- Maintain full backward compatibility
- Add --list-models CLI flag

All tests passing. Ready for production."
```

---

## Success Criteria Checklist

- [ ] Single YAML file for all model configs
- [ ] Easy to add new models (edit YAML only)
- [ ] Easy to add new providers (implement interface)
- [ ] Auto-detection of provider from model name
- [ ] Azure OpenAI support for GPT-5.2 Codex
- [ ] Backward compatible with existing code
- [ ] All 100+ tests pass
- [ ] Type-safe with Pydantic validation
- [ ] Environment variable support for secrets
- [ ] Clean separation of concerns
- [ ] Documentation updated
- [ ] Migration guide created

---

## Plan Complete!

This implementation plan provides:
- **Phase 1:** Configuration system (YAML + Pydantic)
- **Phase 2:** Provider abstraction (Bedrock + Azure)
- **Phase 3:** Refactor CodeAnalyzer
- **Phase 4:** Update CLI
- **Phase 5:** Documentation

Each task follows TDD:
1. Write failing test
2. Run to verify failure
3. Implement minimal code
4. Run to verify success
5. Commit

Total estimated implementation time: 4-6 hours for experienced developer.
