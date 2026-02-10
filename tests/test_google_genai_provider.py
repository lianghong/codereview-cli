"""Tests for Google Generative AI (Gemini) provider."""

from unittest.mock import Mock, patch

import pytest

from codereview.config.models import (
    GoogleGenAIConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport, ReviewMetrics
from codereview.providers.google_genai import GoogleGenAIProvider


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        id="test-gemini",
        name="Test Gemini",
        aliases=["gemini-test"],
        full_id="gemini-3-pro-preview",
        pricing=PricingConfig(
            input_per_million=2.0,
            output_per_million=12.0,
        ),
        inference_params=InferenceParams(
            temperature=0.1,
            top_p=0.95,
            max_output_tokens=65536,
        ),
    )


@pytest.fixture
def provider_config():
    """Create test provider configuration."""
    return GoogleGenAIConfig(
        api_key="test-google-api-key-12345",
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


def test_google_genai_provider_initialization(model_config, provider_config):
    """Test GoogleGenAIProvider can be instantiated."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)
        assert provider is not None
        assert provider.temperature == 0.1


def test_google_genai_provider_default_temperature(provider_config):
    """Test default temperature when not specified in model config."""
    model_config_no_temp = ModelConfig(
        id="test-model",
        name="Test Model",
        aliases=[],
        full_id="gemini-3-pro-preview",
        pricing=PricingConfig(
            input_per_million=2.0,
            output_per_million=12.0,
        ),
    )

    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config_no_temp, provider_config)
        assert provider.temperature == 0.15  # Default


def test_google_genai_provider_custom_temperature(model_config, provider_config):
    """Test temperature override."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config, temperature=0.7)
        assert provider.temperature == 0.7


def test_google_genai_provider_invalid_temperature(model_config, provider_config):
    """Test invalid temperature raises ValueError."""
    with patch("codereview.providers.google_genai.ChatGoogleGenerativeAI"):
        with pytest.raises(ValueError, match="Temperature must be between"):
            GoogleGenAIProvider(model_config, provider_config, temperature=2.5)


def test_google_genai_provider_missing_full_id(provider_config):
    """Test missing full_id raises ValueError."""
    model_config_no_full_id = ModelConfig(
        id="test-model",
        name="Test Model",
        aliases=[],
        pricing=PricingConfig(
            input_per_million=2.0,
            output_per_million=12.0,
        ),
    )

    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        with pytest.raises(ValueError, match="missing required full_id"):
            GoogleGenAIProvider(model_config_no_full_id, provider_config)


def test_google_genai_model_params(model_config, provider_config):
    """Test that model parameters are passed correctly to ChatGoogleGenerativeAI."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        GoogleGenAIProvider(model_config, provider_config)

        # Verify ChatGoogleGenerativeAI was called with correct params
        mock_google.assert_called_once()
        call_kwargs = mock_google.call_args[1]
        assert call_kwargs["model"] == "gemini-3-pro-preview"
        assert call_kwargs["google_api_key"] == "test-google-api-key-12345"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_output_tokens"] == 65536
        assert call_kwargs["top_p"] == 0.95

        # Verify structured output method
        mock_instance.with_structured_output.assert_called_once_with(
            CodeReviewReport, method="json_schema"
        )


def test_analyze_batch(model_config, provider_config, mock_report):
    """Test analyze_batch returns CodeReviewReport."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)

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


def test_token_tracking_usage_metadata(model_config, provider_config, mock_report):
    """Test token tracking from usage_metadata attribute."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        # Setup mock with usage_metadata
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
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
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        # Initial state
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0

        # After analyze
        provider.analyze_batch(1, 1, {"test.py": "code"})
        assert provider.total_input_tokens == 200
        assert provider.total_output_tokens == 100


def test_token_tracking_response_metadata_fallback(
    model_config, provider_config, mock_report
):
    """Test token tracking falls back to response_metadata."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        # No usage_metadata, but has response_metadata
        mock_report_with_metadata.usage_metadata = None
        mock_report_with_metadata.response_metadata = {
            "usage_metadata": {
                "prompt_token_count": 150,
                "candidates_token_count": 75,
            }
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
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        provider.analyze_batch(1, 1, {"test.py": "code"})
        assert provider.total_input_tokens == 150
        assert provider.total_output_tokens == 75


def test_cost_estimation(model_config, provider_config, mock_report):
    """Test cost estimation with Gemini pricing."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.usage_metadata = {
            "input_tokens": 100000,
            "output_tokens": 50000,
        }

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        provider.analyze_batch(1, 1, {"test.py": "code"})

        cost = provider.estimate_cost()

        # 100000 tokens * $2/million = $0.20
        # 50000 tokens * $12/million = $0.60
        assert cost["input_tokens"] == 100000
        assert cost["output_tokens"] == 50000
        assert cost["input_cost"] == pytest.approx(0.2)
        assert cost["output_cost"] == pytest.approx(0.6)
        assert cost["total_cost"] == pytest.approx(0.8)


def test_reset_state(model_config, provider_config, mock_report):
    """Test reset_state clears token counters."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

        provider.analyze_batch(1, 1, {"test.py": "code"})
        assert provider.total_input_tokens > 0

        provider.reset_state()
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0


def test_get_model_display_name(model_config, provider_config):
    """Test get_model_display_name returns model name."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)
        assert provider.get_model_display_name() == "Test Gemini"


def test_get_pricing(model_config, provider_config):
    """Test get_pricing returns pricing info."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)
        pricing = provider.get_pricing()
        assert pricing["input_price_per_million"] == 2.0
        assert pricing["output_price_per_million"] == 12.0


def test_retry_logic_on_resource_exhausted(model_config, provider_config, mock_report):
    """Test exponential backoff retry on ResourceExhausted (429)."""
    with (
        patch(
            "codereview.providers.google_genai.ChatGoogleGenerativeAI"
        ) as mock_google,
        patch("time.sleep") as mock_sleep,
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)

        # Use actual ResourceExhausted exception for isinstance() matching
        from google.api_core.exceptions import ResourceExhausted

        resource_exhausted = ResourceExhausted("429 Resource exhausted")

        provider.chain = Mock()
        provider.chain.invoke.side_effect = [
            resource_exhausted,
            resource_exhausted,
            mock_report,
        ]

        result = provider.analyze_batch(1, 1, {"test.py": "code"})

        assert result == mock_report
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)  # 2 * 2^0
        mock_sleep.assert_any_call(4)  # 2 * 2^1


def test_retry_logic_on_service_unavailable(model_config, provider_config, mock_report):
    """Test retry on ServiceUnavailable (503)."""
    with (
        patch(
            "codereview.providers.google_genai.ChatGoogleGenerativeAI"
        ) as mock_google,
        patch("time.sleep") as mock_sleep,
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)

        # Use actual ServiceUnavailable exception for isinstance() matching
        from google.api_core.exceptions import ServiceUnavailable

        service_unavailable = ServiceUnavailable("503 Service unavailable")

        provider.chain = Mock()
        provider.chain.invoke.side_effect = [
            service_unavailable,
            mock_report,
        ]

        result = provider.analyze_batch(1, 1, {"test.py": "code"})

        assert result == mock_report
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(2)  # 2 * 2^0


def test_retry_exhausted_raises_error(model_config, provider_config):
    """Test that error is raised when all retries are exhausted."""
    with (
        patch(
            "codereview.providers.google_genai.ChatGoogleGenerativeAI"
        ) as mock_google,
        patch("time.sleep"),
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)

        # Use actual ResourceExhausted exception for isinstance() matching
        from google.api_core.exceptions import ResourceExhausted

        resource_exhausted = ResourceExhausted("429 Resource exhausted")

        provider.chain = Mock()
        provider.chain.invoke.side_effect = resource_exhausted

        with pytest.raises(Exception, match="Resource exhausted"):
            provider.analyze_batch(1, 1, {"test.py": "code"}, max_retries=3)


def test_non_retryable_error_not_retried(model_config, provider_config):
    """Test that non-retryable errors are not retried."""
    with (
        patch(
            "codereview.providers.google_genai.ChatGoogleGenerativeAI"
        ) as mock_google,
        patch("time.sleep") as mock_sleep,
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)

        # A generic error (not retryable)
        generic_error = RuntimeError("Something went wrong")

        provider.chain = Mock()
        provider.chain.invoke.side_effect = generic_error

        with pytest.raises(RuntimeError, match="Something went wrong"):
            provider.analyze_batch(1, 1, {"test.py": "code"})

        mock_sleep.assert_not_called()


def test_validate_credentials_valid(model_config, provider_config):
    """Test credential validation with valid config."""
    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        provider = GoogleGenAIProvider(model_config, provider_config)
        result = provider.validate_credentials()

        assert result.valid is True
        assert result.provider == "Google GenAI"


def test_validate_credentials_missing_api_key(model_config):
    """Test that empty API key is rejected by Pydantic validation."""
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(
        PydanticValidationError, match="String should have at least 1 character"
    ):
        GoogleGenAIConfig(api_key="")


def test_validate_credentials_missing_model_id(provider_config):
    """Test credential validation with missing model full_id."""
    model_config_no_full_id = ModelConfig(
        id="test-model",
        name="Test Model",
        aliases=[],
        pricing=PricingConfig(
            input_per_million=2.0,
            output_per_million=12.0,
        ),
    )

    with patch(
        "codereview.providers.google_genai.ChatGoogleGenerativeAI"
    ) as mock_google:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_google.return_value = mock_instance

        # Provider creation will fail because of missing full_id
        # But we can test validate_credentials by creating with a full_id first
        # and then modifying
        model_config_with_id = ModelConfig(
            id="test-model",
            name="Test Model",
            aliases=[],
            full_id="test-model-id",
            pricing=PricingConfig(
                input_per_million=2.0,
                output_per_million=12.0,
            ),
        )

        provider = GoogleGenAIProvider(model_config_with_id, provider_config)

        # Override model_config to simulate missing full_id for validation
        provider.model_config = model_config_no_full_id
        result = provider.validate_credentials()

        assert result.valid is False
        assert any("full_id" in err for err in result.errors)
