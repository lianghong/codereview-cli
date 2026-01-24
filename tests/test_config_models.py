"""Tests for configuration data models."""

import pytest
from pydantic import ValidationError

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)


def test_pricing_config_valid():
    """Test that PricingConfig accepts valid positive prices."""
    config = PricingConfig(input_per_million=1.5, output_per_million=5.0)
    assert config.input_per_million == 1.5
    assert config.output_per_million == 5.0


def test_pricing_config_rejects_negative():
    """Test that PricingConfig rejects negative prices."""
    with pytest.raises(ValidationError) as exc_info:
        PricingConfig(input_per_million=-1.0, output_per_million=5.0)
    assert "greater than or equal to 0" in str(exc_info.value).lower()


def test_pricing_config_accepts_zero():
    """Test that PricingConfig accepts zero prices (for free tier models)."""
    config = PricingConfig(input_per_million=0.0, output_per_million=0.0)
    assert config.input_per_million == 0.0
    assert config.output_per_million == 0.0


def test_inference_params_temperature_range():
    """Test that InferenceParams validates temperature range (0.0-2.0)."""
    # Valid temperatures
    params = InferenceParams(temperature=0.0)
    assert params.temperature == 0.0

    params = InferenceParams(temperature=1.0)
    assert params.temperature == 1.0

    params = InferenceParams(temperature=2.0)
    assert params.temperature == 2.0

    # Invalid: below range
    with pytest.raises(ValidationError) as exc_info:
        InferenceParams(temperature=-0.1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    # Invalid: above range
    with pytest.raises(ValidationError) as exc_info:
        InferenceParams(temperature=2.1)
    assert "less than or equal to 2" in str(exc_info.value).lower()


def test_inference_params_top_p_range():
    """Test that InferenceParams validates top_p range (0.0-1.0)."""
    # Valid top_p values
    params = InferenceParams(top_p=0.0)
    assert params.top_p == 0.0

    params = InferenceParams(top_p=0.5)
    assert params.top_p == 0.5

    params = InferenceParams(top_p=1.0)
    assert params.top_p == 1.0

    # Invalid: below range
    with pytest.raises(ValidationError) as exc_info:
        InferenceParams(top_p=-0.1)
    assert "greater than or equal to 0" in str(exc_info.value).lower()

    # Invalid: above range
    with pytest.raises(ValidationError) as exc_info:
        InferenceParams(top_p=1.1)
    assert "less than or equal to 1" in str(exc_info.value).lower()


def test_model_config_creation():
    """Test that ModelConfig can be created with all fields."""
    pricing = PricingConfig(input_per_million=3.0, output_per_million=15.0)
    inference = InferenceParams(
        temperature=0.1, top_p=0.9, top_k=50, max_output_tokens=4096
    )
    model = ModelConfig(
        id="test-model",
        name="Test Model",
        aliases=["test", "tm"],
        pricing=pricing,
        inference_params=inference,
        full_id="provider.test-model-v1",
        deployment_name="test-deployment",
    )

    assert model.id == "test-model"
    assert model.name == "Test Model"
    assert model.aliases == ["test", "tm"]
    assert model.pricing.input_per_million == 3.0
    assert model.inference_params.temperature == 0.1
    assert model.full_id == "provider.test-model-v1"
    assert model.deployment_name == "test-deployment"


def test_bedrock_config_default_region():
    """Test that BedrockConfig has default region."""
    config = BedrockConfig()
    assert config.region == "us-west-2"
    assert config.models == []

    # Can override default region
    config = BedrockConfig(region="eu-west-1")
    assert config.region == "eu-west-1"


def test_azure_config_requires_fields():
    """Test that AzureOpenAIConfig requires endpoint, api_key, and api_version."""
    # Should raise ValidationError when required fields are missing
    with pytest.raises(ValidationError) as exc_info:
        AzureOpenAIConfig()
    errors = str(exc_info.value).lower()
    assert "endpoint" in errors
    assert "api_key" in errors
    assert "api_version" in errors

    # Should succeed with all required fields
    config = AzureOpenAIConfig(
        endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="2024-02-01",
    )
    assert str(config.endpoint) == "https://example.openai.azure.com/"
    assert config.api_key == "test-key"
    assert config.api_version == "2024-02-01"
    assert config.models == []


def test_model_config_rejects_empty_strings():
    """Test that ModelConfig rejects empty strings for id and name."""
    pricing = PricingConfig(input_per_million=3.0, output_per_million=15.0)

    # Empty id should be rejected
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig(id="", name="Test Model", pricing=pricing)
    assert "at least 1 character" in str(exc_info.value).lower()

    # Empty name should be rejected
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig(id="test-model", name="", pricing=pricing)
    assert "at least 1 character" in str(exc_info.value).lower()


def test_azure_config_invalid_url():
    """Test that AzureOpenAIConfig rejects invalid URLs."""
    # Invalid URL format
    with pytest.raises(ValidationError) as exc_info:
        AzureOpenAIConfig(
            endpoint="not-a-valid-url", api_key="test-key", api_version="2024-02-01"
        )
    assert "url" in str(exc_info.value).lower()

    # Empty endpoint
    with pytest.raises(ValidationError) as exc_info:
        AzureOpenAIConfig(endpoint="", api_key="test-key", api_version="2024-02-01")
    errors = str(exc_info.value).lower()
    assert "url" in errors or "endpoint" in errors

    # Empty api_key
    with pytest.raises(ValidationError) as exc_info:
        AzureOpenAIConfig(
            endpoint="https://example.openai.azure.com",
            api_key="",
            api_version="2024-02-01",
        )
    assert "at least 1 character" in str(exc_info.value).lower()

    # Empty api_version
    with pytest.raises(ValidationError) as exc_info:
        AzureOpenAIConfig(
            endpoint="https://example.openai.azure.com",
            api_key="test-key",
            api_version="",
        )
    assert "at least 1 character" in str(exc_info.value).lower()


def test_models_are_immutable():
    """Test that all model classes are immutable (frozen)."""
    # PricingConfig
    pricing = PricingConfig(input_per_million=3.0, output_per_million=15.0)
    with pytest.raises(ValidationError) as exc_info:
        pricing.input_per_million = 5.0
    assert "frozen" in str(exc_info.value).lower()

    # InferenceParams
    inference = InferenceParams(temperature=0.5)
    with pytest.raises(ValidationError) as exc_info:
        inference.temperature = 1.0
    assert "frozen" in str(exc_info.value).lower()

    # ModelConfig
    model = ModelConfig(id="test", name="Test Model", pricing=pricing)
    with pytest.raises(ValidationError) as exc_info:
        model.id = "new-test"
    assert "frozen" in str(exc_info.value).lower()

    # BedrockConfig
    bedrock = BedrockConfig()
    with pytest.raises(ValidationError) as exc_info:
        bedrock.region = "us-east-1"
    assert "frozen" in str(exc_info.value).lower()

    # AzureOpenAIConfig
    azure = AzureOpenAIConfig(
        endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="2024-02-01",
    )
    with pytest.raises(ValidationError) as exc_info:
        azure.api_key = "new-key"
    assert "frozen" in str(exc_info.value).lower()


def test_string_fields_reject_empty_when_provided():
    """Test that optional string fields reject empty strings."""
    pricing = PricingConfig(input_per_million=1.0, output_per_million=1.0)

    # Test ModelConfig.full_id rejects empty string
    with pytest.raises(ValidationError):
        ModelConfig(id="test", name="Test", pricing=pricing, full_id="")

    # Test ModelConfig.deployment_name rejects empty string
    with pytest.raises(ValidationError):
        ModelConfig(id="test", name="Test", pricing=pricing, deployment_name="")

    # Test BedrockConfig.region rejects empty string (override)
    with pytest.raises(ValidationError):
        BedrockConfig(models=[], region="")
