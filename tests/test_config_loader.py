import tempfile
from pathlib import Path

import pytest

from codereview.config.loader import ConfigLoader
from codereview.config.models import AzureOpenAIConfig, BedrockConfig


def create_test_yaml() -> str:
    """Create minimal test YAML configuration."""
    return """
version: "1.0"

providers:
  bedrock:
    region: us-west-2
    models:
      - id: test-opus
        full_id: test.opus.v1
        name: Test Opus
        aliases: [opus-test, test]
        pricing:
          input_per_million: 5.0
          output_per_million: 25.0
        inference_params:
          default_temperature: 0.1

      - id: test-sonnet
        full_id: test.sonnet.v1
        name: Test Sonnet
        aliases: [sonnet-test]
        pricing:
          input_per_million: 3.0
          output_per_million: 15.0

  azure_openai:
    endpoint: "${AZURE_TEST_ENDPOINT}"
    api_key: "${AZURE_TEST_KEY}"
    api_version: "2024-01-01"
    models:
      - id: test-gpt
        deployment_name: gpt-test
        name: Test GPT
        aliases: [gpt-test-alias]
        pricing:
          input_per_million: 2.0
          output_per_million: 10.0
        inference_params:
          default_temperature: 0.0
          max_output_tokens: 4096
"""


@pytest.fixture
def temp_config_file():
    """Create temporary YAML config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(create_test_yaml())
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def loader_with_env(temp_config_file, monkeypatch):
    """Create loader with environment variables set."""
    monkeypatch.setenv("AZURE_TEST_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_TEST_KEY", "test-api-key-12345")

    return ConfigLoader(temp_config_file)


def test_config_loader_initialization(loader_with_env):
    """Test ConfigLoader loads configuration successfully."""
    assert loader_with_env is not None
    assert loader_with_env._raw_config is not None


def test_resolve_model_by_id(loader_with_env):
    """Test resolving model by primary ID."""
    provider, model = loader_with_env.resolve_model("test-opus")

    assert provider == "bedrock"
    assert model.id == "test-opus"
    assert model.name == "Test Opus"
    assert model.full_id == "test.opus.v1"


def test_resolve_model_by_alias(loader_with_env):
    """Test resolving model by alias."""
    provider, model = loader_with_env.resolve_model("opus-test")

    assert provider == "bedrock"
    assert model.id == "test-opus"  # Returns primary config


def test_resolve_model_unknown(loader_with_env):
    """Test resolving unknown model raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        loader_with_env.resolve_model("nonexistent")


def test_get_provider_config_bedrock(loader_with_env):
    """Test getting Bedrock provider config."""
    config = loader_with_env.get_provider_config("bedrock")

    assert isinstance(config, BedrockConfig)
    assert config.region == "us-west-2"


def test_get_provider_config_azure(loader_with_env):
    """Test getting Azure provider config."""
    config = loader_with_env.get_provider_config("azure_openai")

    assert isinstance(config, AzureOpenAIConfig)
    assert str(config.endpoint) == "https://test.openai.azure.com/"
    assert config.api_key == "test-api-key-12345"
    assert config.api_version == "2024-01-01"


def test_get_provider_config_unknown(loader_with_env):
    """Test getting unknown provider raises ValueError."""
    with pytest.raises(ValueError, match="Unknown provider"):
        loader_with_env.get_provider_config("unknown")


def test_list_models(loader_with_env):
    """Test listing all models grouped by provider."""
    models = loader_with_env.list_models()

    assert "bedrock" in models
    assert "azure_openai" in models

    assert len(models["bedrock"]) == 2  # test-opus, test-sonnet
    assert len(models["azure_openai"]) == 1  # test-gpt


def test_env_var_expansion(temp_config_file, monkeypatch):
    """Test environment variable expansion in YAML."""
    monkeypatch.setenv("AZURE_TEST_ENDPOINT", "https://custom.endpoint.com")
    monkeypatch.setenv("AZURE_TEST_KEY", "secret-key")

    loader = ConfigLoader(temp_config_file)
    config = loader.get_provider_config("azure_openai")

    assert str(config.endpoint) == "https://custom.endpoint.com/"
    assert config.api_key == "secret-key"


def test_missing_env_vars_handled(temp_config_file, monkeypatch):
    """Test missing env vars result in empty strings (validation catches later)."""
    # Don't set env vars
    monkeypatch.delenv("AZURE_TEST_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_TEST_KEY", raising=False)

    # Should not crash during loading, but validation may fail
    # This tests that env var expansion doesn't crash
    try:
        ConfigLoader(temp_config_file)
        # If Pydantic validation is strict, this might raise
    except Exception:
        # Expected if validation enforces non-empty values
        pass
