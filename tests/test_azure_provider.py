from unittest.mock import Mock, patch

import pytest
from openai import RateLimitError

from codereview.config.models import (
    AzureOpenAIConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport
from codereview.providers.azure_openai import AzureOpenAIProvider


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        id="test-gpt",
        name="Test GPT",
        aliases=["gpt-test"],
        deployment_name="gpt-test-deployment",
        pricing=PricingConfig(
            input_per_million=2.0,
            output_per_million=10.0,
        ),
        inference_params=InferenceParams(
            default_temperature=0.0,
            default_top_p=0.95,
            max_output_tokens=4096,
        ),
    )


@pytest.fixture
def provider_config():
    """Create test provider configuration."""
    return AzureOpenAIConfig(
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        api_version="2024-01-01",
    )


@pytest.fixture
def mock_report():
    """Create mock CodeReviewReport."""
    return CodeReviewReport(
        summary="Test analysis",
        metrics={"files_analyzed": 1, "issues_found": 0, "critical_issues": 0},
        issues=[],
        system_design_insights="No major design issues",
        recommendations=["Keep up the good work"],
        improvement_suggestions=[],
    )


def test_azure_provider_initialization(model_config, provider_config):
    """Test AzureOpenAIProvider can be instantiated."""
    with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
        provider = AzureOpenAIProvider(model_config, provider_config)
        assert provider is not None
        assert provider.temperature == 0.0


def test_azure_provider_custom_temperature(model_config, provider_config):
    """Test temperature override."""
    with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
        provider = AzureOpenAIProvider(model_config, provider_config, temperature=0.5)
        assert provider.temperature == 0.5


def test_analyze_batch(model_config, provider_config, mock_report):
    """Test analyze_batch returns CodeReviewReport."""
    with patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure:
        # Setup mock
        mock_instance = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = mock_report
        mock_instance.with_structured_output.return_value = mock_structured
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)
        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"test.py": "print('hello')"},
        )

        assert isinstance(result, CodeReviewReport)
        assert result.summary == "Test analysis"


def test_token_tracking(model_config, provider_config, mock_report):
    """Test token tracking."""
    with patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure:
        # Setup mock with usage metadata (Azure uses different field names)
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "token_usage": {"prompt_tokens": 150, "completion_tokens": 75}
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
        mock_structured = Mock()
        mock_structured.invoke.return_value = mock_report_with_metadata
        mock_instance.with_structured_output.return_value = mock_structured
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)

        # Initial state
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0

        # After analyze
        provider.analyze_batch(1, 1, {"test.py": "code"})
        assert provider.total_input_tokens == 150
        assert provider.total_output_tokens == 75


def test_cost_estimation(model_config, provider_config, mock_report):
    """Test cost estimation."""
    with patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "token_usage": {"prompt_tokens": 100000, "completion_tokens": 50000}
        }

        mock_instance = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = mock_report_with_metadata
        mock_instance.with_structured_output.return_value = mock_structured
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)
        provider.analyze_batch(1, 1, {"test.py": "code"})

        cost = provider.estimate_cost()

        # 100000 tokens * $2/million = $0.20
        # 50000 tokens * $10/million = $0.50
        assert cost["input_tokens"] == 100000
        assert cost["output_tokens"] == 50000
        assert cost["input_cost"] == 0.2
        assert cost["output_cost"] == 0.5
        assert cost["total_cost"] == 0.7


def test_reset_state(model_config, provider_config, mock_report):
    """Test reset_state clears token counters."""
    with patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }

        mock_instance = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = mock_report_with_metadata
        mock_instance.with_structured_output.return_value = mock_structured
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)
        provider.analyze_batch(1, 1, {"test.py": "code"})

        assert provider.total_input_tokens > 0

        provider.reset_state()
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0


def test_get_model_display_name(model_config, provider_config):
    """Test get_model_display_name returns model name."""
    with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
        provider = AzureOpenAIProvider(model_config, provider_config)
        assert provider.get_model_display_name() == "Test GPT"


def test_retry_logic_on_rate_limit(model_config, provider_config, mock_report):
    """Test exponential backoff retry on rate limit."""
    with (
        patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure,
        patch("time.sleep") as mock_sleep,
    ):

        mock_instance = Mock()
        mock_structured = Mock()

        # Create proper RateLimitError with required arguments
        mock_response = Mock()
        mock_response.status_code = 429
        rate_limit_error = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        # First two calls raise RateLimitError, third succeeds
        mock_structured.invoke.side_effect = [
            rate_limit_error,
            rate_limit_error,
            mock_report,
        ]

        mock_instance.with_structured_output.return_value = mock_structured
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)
        result = provider.analyze_batch(1, 1, {"test.py": "code"})

        # Should succeed after 2 retries
        assert result == mock_report

        # Verify exponential backoff: 1s, 2s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)  # 2^0
        mock_sleep.assert_any_call(2)  # 2^1
