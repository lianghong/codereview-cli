import tempfile
from pathlib import Path

import pytest

from codereview.config.loader import ConfigLoader
from codereview.providers.azure_openai import AzureOpenAIProvider
from codereview.providers.bedrock import BedrockProvider
from codereview.providers.factory import ProviderFactory


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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(create_test_yaml())
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def config_loader_with_env(temp_config_file, monkeypatch):
    """Create loader with environment variables set."""
    monkeypatch.setenv("AZURE_TEST_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_TEST_KEY", "test-api-key-12345")

    return ConfigLoader(temp_config_file)


@pytest.fixture
def factory(config_loader_with_env):
    """Create ProviderFactory with test config."""
    return ProviderFactory(config_loader_with_env)


def test_factory_initialization(factory):
    """Test ProviderFactory can be instantiated."""
    assert factory is not None
    assert factory.config_loader is not None


def test_factory_default_config_loader(monkeypatch):
    """Test factory creates default ConfigLoader if none provided."""
    # Set dummy env vars for default config
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-key")

    factory = ProviderFactory()
    assert factory.config_loader is not None


def test_create_bedrock_provider_by_id(factory):
    """Test creating Bedrock provider by model ID."""
    provider = factory.create_provider("test-opus")

    assert isinstance(provider, BedrockProvider)
    assert provider.model_config.id == "test-opus"
    assert provider.model_config.name == "Test Opus"


def test_create_bedrock_provider_by_alias(factory):
    """Test creating Bedrock provider by alias."""
    provider = factory.create_provider("opus-test")

    assert isinstance(provider, BedrockProvider)
    assert provider.model_config.id == "test-opus"  # Resolves to primary ID


def test_create_azure_provider_by_id(factory):
    """Test creating Azure provider by model ID."""
    provider = factory.create_provider("test-gpt")

    assert isinstance(provider, AzureOpenAIProvider)
    assert provider.model_config.id == "test-gpt"
    assert provider.model_config.name == "Test GPT"


def test_create_azure_provider_by_alias(factory):
    """Test creating Azure provider by alias."""
    provider = factory.create_provider("gpt-test-alias")

    assert isinstance(provider, AzureOpenAIProvider)
    assert provider.model_config.id == "test-gpt"


def test_create_provider_with_custom_temperature(factory):
    """Test temperature override."""
    provider = factory.create_provider("test-opus", temperature=0.5)

    assert isinstance(provider, BedrockProvider)
    assert provider.temperature == 0.5


def test_create_provider_unknown_model(factory):
    """Test unknown model raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        factory.create_provider("nonexistent-model")


def test_list_available_models(factory):
    """Test listing all available models."""
    models = factory.list_available_models()

    assert "bedrock" in models
    assert "azure_openai" in models

    # Check Bedrock models
    assert len(models["bedrock"]) == 2
    opus_model = next(m for m in models["bedrock"] if m["id"] == "test-opus")
    assert opus_model["name"] == "Test Opus"
    assert "opus-test" in opus_model["aliases"]

    # Check Azure models
    assert len(models["azure_openai"]) == 1
    gpt_model = models["azure_openai"][0]
    assert gpt_model["id"] == "test-gpt"
    assert gpt_model["name"] == "Test GPT"
