"""Tests for NVIDIA NIM API provider."""

from unittest.mock import Mock, patch

import httpx
import pytest

from codereview.config.models import (
    InferenceParams,
    ModelConfig,
    NVIDIAConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport, ReviewMetrics
from codereview.providers.nvidia import NVIDIAProvider


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        id="test-devstral",
        name="Test Devstral",
        aliases=["devstral-test"],
        full_id="mistralai/devstral-2-123b-instruct-2512",
        pricing=PricingConfig(
            input_per_million=0.0,  # Free tier
            output_per_million=0.0,
        ),
        inference_params=InferenceParams(
            temperature=0.15,
            top_p=0.95,
            max_output_tokens=8192,
        ),
    )


@pytest.fixture
def provider_config():
    """Create test provider configuration."""
    return NVIDIAConfig(
        api_key="nvapi-test-key-12345",
    )


@pytest.fixture
def provider_config_with_base_url():
    """Create test provider configuration with custom base URL."""
    return NVIDIAConfig(
        api_key="nvapi-test-key-12345",
        base_url="https://custom-nim.example.com/v1",
    )


@pytest.fixture
def mock_report():
    """Create mock CodeReviewReport."""
    return CodeReviewReport(
        summary="Test analysis",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=0, critical_issues=0),
        issues=[],
        system_design_insights="No major design issues",
        recommendations=["Keep up the good work"],
        improvement_suggestions=[],
    )


def test_nvidia_provider_initialization(model_config, provider_config):
    """Test NVIDIAProvider can be instantiated."""
    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        provider = NVIDIAProvider(model_config, provider_config)
        assert provider is not None
        assert provider.temperature == 0.15


def test_nvidia_provider_default_temperature(provider_config):
    """Test default temperature when not specified in model config."""
    model_config_no_temp = ModelConfig(
        id="test-model",
        name="Test Model",
        aliases=[],
        full_id="test/model-id",
        pricing=PricingConfig(
            input_per_million=0.0,
            output_per_million=0.0,
        ),
    )

    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        provider = NVIDIAProvider(model_config_no_temp, provider_config)
        assert provider.temperature == 0.15  # NVIDIA default


def test_nvidia_provider_custom_temperature(model_config, provider_config):
    """Test temperature override."""
    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        provider = NVIDIAProvider(model_config, provider_config, temperature=0.7)
        assert provider.temperature == 0.7


def test_nvidia_provider_invalid_temperature(model_config, provider_config):
    """Test invalid temperature raises ValueError."""
    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        with pytest.raises(ValueError, match="Temperature must be between"):
            NVIDIAProvider(model_config, provider_config, temperature=2.5)


def test_nvidia_provider_missing_full_id(provider_config):
    """Test missing full_id raises ValueError."""
    model_config_no_full_id = ModelConfig(
        id="test-model",
        name="Test Model",
        aliases=[],
        pricing=PricingConfig(
            input_per_million=0.0,
            output_per_million=0.0,
        ),
    )

    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        with pytest.raises(ValueError, match="missing required full_id"):
            NVIDIAProvider(model_config_no_full_id, provider_config)


def test_nvidia_provider_with_base_url(model_config, provider_config_with_base_url):
    """Test provider with custom base URL for self-hosted NIMs."""
    with patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        NVIDIAProvider(model_config, provider_config_with_base_url)

        # Verify base_url was passed to ChatNVIDIA
        mock_nvidia.assert_called_once()
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs.get("base_url") == "https://custom-nim.example.com/v1"


def test_analyze_batch(model_config, provider_config, mock_report):
    """Test analyze_batch returns CodeReviewReport."""
    with patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia:
        # Setup mock
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)

        # Mock the chain's invoke method
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report

        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"test.py": "print('hello')"},
        )

        assert isinstance(result, CodeReviewReport)
        assert result.summary == "Test analysis"


def test_token_tracking(model_config, provider_config, mock_report):
    """Test token tracking from response metadata."""
    with patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia:
        # Setup mock with usage metadata
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "usage": {"prompt_tokens": 200, "completion_tokens": 100}
        }
        mock_report_with_metadata.model_dump_json.return_value = "{}"

        # Copy attributes from mock_report
        for attr in [
            "summary",
            "metrics",
            "issues",
            "system_design_insights",
            "recommendations",
            "improvement_suggestions",
        ]:
            setattr(mock_report_with_metadata, attr, getattr(mock_report, attr))

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)

        # Mock the chain's invoke method
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        # Initial state
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0

        # After analyze
        provider.analyze_batch(1, 1, {"test.py": "code"})
        assert provider.total_input_tokens == 200
        assert provider.total_output_tokens == 100


def test_token_tracking_alternative_format(model_config, provider_config, mock_report):
    """Test token tracking with alternative metadata format (token_usage)."""
    with patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia:
        # Setup mock with alternative usage metadata format
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "token_usage": {"input_tokens": 150, "output_tokens": 75}
        }
        mock_report_with_metadata.model_dump_json.return_value = "{}"

        for attr in [
            "summary",
            "metrics",
            "issues",
            "system_design_insights",
            "recommendations",
            "improvement_suggestions",
        ]:
            setattr(mock_report_with_metadata, attr, getattr(mock_report, attr))

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        provider.analyze_batch(1, 1, {"test.py": "code"})
        assert provider.total_input_tokens == 150
        assert provider.total_output_tokens == 75


def test_cost_estimation_free_tier(model_config, provider_config, mock_report):
    """Test cost estimation with free tier pricing ($0)."""
    with patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "usage": {"prompt_tokens": 100000, "completion_tokens": 50000}
        }

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        provider.analyze_batch(1, 1, {"test.py": "code"})

        cost = provider.estimate_cost()

        # Free tier: $0 pricing
        assert cost["input_tokens"] == 100000
        assert cost["output_tokens"] == 50000
        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0
        assert cost["total_cost"] == 0.0


def test_cost_estimation_paid_tier(provider_config, mock_report):
    """Test cost estimation with paid pricing."""
    model_config_paid = ModelConfig(
        id="test-model-paid",
        name="Test Model Paid",
        aliases=[],
        full_id="test/model-id",
        pricing=PricingConfig(
            input_per_million=2.0,  # $2 per million input
            output_per_million=8.0,  # $8 per million output
        ),
    )

    with patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "usage": {"prompt_tokens": 100000, "completion_tokens": 50000}
        }

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config_paid, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        provider.analyze_batch(1, 1, {"test.py": "code"})

        cost = provider.estimate_cost()

        # 100000 tokens * $2/million = $0.20
        # 50000 tokens * $8/million = $0.40
        assert cost["input_tokens"] == 100000
        assert cost["output_tokens"] == 50000
        assert cost["input_cost"] == pytest.approx(0.2)
        assert cost["output_cost"] == pytest.approx(0.4)
        assert cost["total_cost"] == pytest.approx(0.6)


def test_reset_state(model_config, provider_config, mock_report):
    """Test reset_state clears token counters."""
    with patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        provider.analyze_batch(1, 1, {"test.py": "code"})

        assert provider.total_input_tokens > 0

        provider.reset_state()
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0


def test_get_model_display_name(model_config, provider_config):
    """Test get_model_display_name returns model name."""
    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        provider = NVIDIAProvider(model_config, provider_config)
        assert provider.get_model_display_name() == "Test Devstral"


def test_get_pricing(model_config, provider_config):
    """Test get_pricing returns pricing info."""
    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        provider = NVIDIAProvider(model_config, provider_config)
        pricing = provider.get_pricing()
        assert pricing["input_price_per_million"] == 0.0
        assert pricing["output_price_per_million"] == 0.0


def test_retry_logic_on_rate_limit(model_config, provider_config, mock_report):
    """Test exponential backoff retry on HTTP 429 rate limit."""
    with (
        patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia,
        patch("time.sleep") as mock_sleep,
    ):

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)

        # Create HTTP 429 rate limit error
        mock_request = Mock()
        mock_response = Mock()
        mock_response.status_code = 429
        rate_limit_error = httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=mock_request,
            response=mock_response,
        )

        # Mock the chain's invoke method with side effects
        provider.chain = Mock()
        provider.chain.invoke.side_effect = [
            rate_limit_error,
            rate_limit_error,
            mock_report,
        ]

        result = provider.analyze_batch(1, 1, {"test.py": "code"})

        # Should succeed after 2 retries
        assert result == mock_report

        # Verify exponential backoff: 1s, 2s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)  # 2^0
        mock_sleep.assert_any_call(2)  # 2^1


def test_retry_exhausted_raises_error(model_config, provider_config):
    """Test that error is raised when all retries are exhausted."""
    with (
        patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia,
        patch("time.sleep"),
    ):

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)

        # Create HTTP 429 rate limit error
        mock_request = Mock()
        mock_response = Mock()
        mock_response.status_code = 429
        rate_limit_error = httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=mock_request,
            response=mock_response,
        )

        # Always fail
        provider.chain = Mock()
        provider.chain.invoke.side_effect = rate_limit_error

        with pytest.raises(httpx.HTTPStatusError):
            provider.analyze_batch(1, 1, {"test.py": "code"}, max_retries=3)


def test_non_rate_limit_error_not_retried(model_config, provider_config):
    """Test that non-429 errors are not retried."""
    with (
        patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia,
        patch("time.sleep") as mock_sleep,
    ):

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)

        # Create HTTP 500 error (not rate limit)
        mock_request = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        server_error = httpx.HTTPStatusError(
            "Internal server error",
            request=mock_request,
            response=mock_response,
        )

        provider.chain = Mock()
        provider.chain.invoke.side_effect = server_error

        with pytest.raises(httpx.HTTPStatusError):
            provider.analyze_batch(1, 1, {"test.py": "code"})

        # Should not have retried (no sleep calls)
        mock_sleep.assert_not_called()


def test_retry_logic_on_gateway_timeout(model_config, provider_config, mock_report):
    """Test retry with longer backoff on HTTP 504 gateway timeout."""
    with (
        patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia,
        patch("time.sleep") as mock_sleep,
    ):

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)

        # Create HTTP 504 Gateway Timeout error
        mock_request = Mock()
        mock_response = Mock()
        mock_response.status_code = 504
        gateway_timeout_error = httpx.HTTPStatusError(
            "Gateway Timeout",
            request=mock_request,
            response=mock_response,
        )

        # Mock the chain's invoke method with side effects
        provider.chain = Mock()
        provider.chain.invoke.side_effect = [
            gateway_timeout_error,
            gateway_timeout_error,
            mock_report,
        ]

        result = provider.analyze_batch(1, 1, {"test.py": "code"})

        # Should succeed after 2 retries
        assert result == mock_report

        # Verify longer exponential backoff for 504: 4^0=1s, 4^1=4s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)  # 4^0
        mock_sleep.assert_any_call(4)  # 4^1


def test_retry_logic_on_service_unavailable(model_config, provider_config, mock_report):
    """Test retry on HTTP 502/503 service unavailable errors."""
    with (
        patch("codereview.providers.nvidia.ChatNVIDIA") as mock_nvidia,
        patch("time.sleep") as mock_sleep,
    ):

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_nvidia.return_value = mock_instance

        provider = NVIDIAProvider(model_config, provider_config)

        # Create HTTP 503 Service Unavailable error
        mock_request = Mock()
        mock_response = Mock()
        mock_response.status_code = 503
        service_unavailable_error = httpx.HTTPStatusError(
            "Service Unavailable",
            request=mock_request,
            response=mock_response,
        )

        # Mock success after one 503 error
        provider.chain = Mock()
        provider.chain.invoke.side_effect = [
            service_unavailable_error,
            mock_report,
        ]

        result = provider.analyze_batch(1, 1, {"test.py": "code"})

        # Should succeed after 1 retry
        assert result == mock_report

        # Verify exponential backoff: 2^0=1s (using base 2 for non-504 errors)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(1)


def test_validate_credentials_valid(model_config, provider_config):
    """Test credential validation with valid config."""
    with (
        patch("codereview.providers.nvidia.ChatNVIDIA"),
        patch("codereview.providers.nvidia.httpx.Client") as mock_client,
    ):
        # Mock successful connection test
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client_instance = Mock()
        mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = Mock(return_value=False)
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        provider = NVIDIAProvider(model_config, provider_config)
        result = provider.validate_credentials()

        assert result.valid is True
        assert result.provider == "NVIDIA NIM"


def test_validate_credentials_missing_api_key(model_config):
    """Test that empty API key is rejected by Pydantic validation."""
    from pydantic import ValidationError as PydanticValidationError

    # NVIDIAConfig requires api_key with min_length=1
    # Empty string should raise Pydantic ValidationError at config creation
    with pytest.raises(
        PydanticValidationError, match="String should have at least 1 character"
    ):
        NVIDIAConfig(api_key="")


def test_validate_credentials_invalid_key_format(model_config):
    """Test credential validation warns about invalid key format."""
    provider_config_bad_format = NVIDIAConfig(api_key="not-a-nvidia-key")

    with (
        patch("codereview.providers.nvidia.ChatNVIDIA"),
        patch("codereview.providers.nvidia.httpx.Client") as mock_client,
    ):
        mock_response = Mock()
        mock_response.status_code = 401
        mock_client_instance = Mock()
        mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = Mock(return_value=False)
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance

        provider = NVIDIAProvider(model_config, provider_config_bad_format)
        result = provider.validate_credentials()

        # Should still be valid but with warning
        assert result.valid is True
        assert any("nvapi-" in w for w in result.warnings)


def test_validate_credentials_connection_error(model_config, provider_config):
    """Test credential validation handles connection errors gracefully."""
    with (
        patch("codereview.providers.nvidia.ChatNVIDIA"),
        patch("codereview.providers.nvidia.httpx.Client") as mock_client,
    ):
        mock_client_instance = Mock()
        mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = Mock(return_value=False)
        mock_client_instance.get.side_effect = Exception("Connection failed")
        mock_client.return_value = mock_client_instance

        provider = NVIDIAProvider(model_config, provider_config)
        result = provider.validate_credentials()

        # Should still be valid but with warning
        assert result.valid is True
        assert any("Connection test failed" in w for w in result.warnings)
