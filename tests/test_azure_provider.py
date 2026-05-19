from unittest.mock import Mock, patch

import pytest
from openai import RateLimitError

from codereview.config.models import (
    AzureOpenAIConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport, ReviewMetrics
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
        # Field names match InferenceParams: `temperature`/`top_p`,
        # not the YAML-side `default_temperature`/`default_top_p`.
        # Older versions of this fixture used the wrong kwargs and were
        # silently ignored by Pydantic, leaking a None temperature.
        inference_params=InferenceParams(
            temperature=0.0,
            top_p=0.95,
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
        metrics=ReviewMetrics(files_analyzed=1, total_issues=0, critical_issues=0),
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
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)

        # Mock the chain's invoke method (chain = prompt | model)
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
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)

        # Mock the chain's invoke method
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

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
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)

        # Mock the chain's invoke method
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

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
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)

        # Mock the chain's invoke method
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report_with_metadata

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
    """Test retry on rate limit respects Azure Retry-After header."""
    with (
        patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure,
        patch("time.sleep") as mock_sleep,
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)

        # Create RateLimitError with Retry-After header (as Azure sends)
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "15"}
        rate_limit_error = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
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

        # Verify backoff uses Retry-After value (15s) for both retries
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(15.0)


def test_retry_logic_fallback_backoff(model_config, provider_config, mock_report):
    """Test retry fallback backoff when no Retry-After header is present."""
    with (
        patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure,
        patch("time.sleep") as mock_sleep,
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)

        # Create RateLimitError without Retry-After header
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {}
        rate_limit_error = RateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        provider.chain = Mock()
        provider.chain.invoke.side_effect = [
            rate_limit_error,
            mock_report,
        ]

        result = provider.analyze_batch(1, 1, {"test.py": "code"})

        assert result == mock_report
        # Fallback: 10.0 * 2^0 = 10s for first retry
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(10.0)


def test_responses_api_enabled(provider_config):
    """Test that use_responses_api=True is passed to AzureChatOpenAI.

    When use_responses_api is enabled, temperature and top_p should NOT be passed
    as these parameters are not supported by the Responses API.
    """
    model_config_with_responses_api = ModelConfig(
        id="gpt-5.4",
        name="GPT-5.4",
        aliases=["gpt"],
        deployment_name="gpt-5.4",
        pricing=PricingConfig(
            input_per_million=2.50,
            output_per_million=15.00,
        ),
        # Reasoning model: no temperature/top_p — same shape as the
        # real gpt-5.4 entry in models.yaml.
        inference_params=InferenceParams(
            max_output_tokens=128000,
        ),
        use_responses_api=True,  # Enable Responses API
    )

    with patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        AzureOpenAIProvider(model_config_with_responses_api, provider_config)

        # Verify use_responses_api=True was passed to AzureChatOpenAI
        mock_azure.assert_called_once()
        call_kwargs = mock_azure.call_args[1]
        assert call_kwargs.get("use_responses_api") is True
        # Verify temperature and top_p are NOT passed (not supported by Responses API)
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs


def test_responses_api_disabled_by_default(model_config, provider_config):
    """Test that use_responses_api is not set when not specified in config.

    When use_responses_api is not enabled, temperature should be passed normally.
    """
    # model_config fixture has use_responses_api=None (default)
    assert model_config.use_responses_api is None

    with patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        AzureOpenAIProvider(model_config, provider_config)

        # Verify use_responses_api was NOT passed to AzureChatOpenAI
        mock_azure.assert_called_once()
        call_kwargs = mock_azure.call_args[1]
        assert "use_responses_api" not in call_kwargs
        # Verify temperature IS passed when not using Responses API
        assert "temperature" in call_kwargs


def test_supports_tool_use_false_uses_prompt_parsing(provider_config):
    """Models with supports_tool_use=False (e.g. DeepSeek-V4-Pro on Azure)
    must skip with_structured_output and parse JSON via prompt instructions.

    Regression: previously the Azure provider always called
    with_structured_output, which crashes against models that don't expose
    tool calling on Foundry.
    """
    no_tool_use_config = ModelConfig(
        id="deepseek-v4-pro-azure",
        name="DeepSeek-V4-Pro (Azure)",
        aliases=["dsv4-azure"],
        deployment_name="DeepSeek-V4-Pro",
        pricing=PricingConfig(input_per_million=1.74, output_per_million=3.48),
        inference_params=InferenceParams(max_output_tokens=32000),
        supports_tool_use=False,
    )

    with patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure:
        mock_instance = Mock()
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(no_tool_use_config, provider_config)

        # The model returned by _create_model is the raw chat model — never
        # wrapped via with_structured_output for tool-use-less deployments.
        mock_instance.with_structured_output.assert_not_called()
        assert provider._use_prompt_parsing is True

        # Reasoning model: temperature must NOT be passed.
        call_kwargs = mock_azure.call_args[1]
        assert "temperature" not in call_kwargs

        # Regression: Foundry's SGLang backend rejects null `model` field
        # in the body with HTTP 400. Provider must populate the field with
        # the deployment name so the JSON serializes to a real string.
        assert call_kwargs["model"] == "DeepSeek-V4-Pro"


def test_supports_tool_use_true_uses_structured_output(model_config, provider_config):
    """The default model_config sets supports_tool_use=True, so the provider
    keeps the existing with_structured_output path."""
    with patch("codereview.providers.azure_openai.AzureChatOpenAI") as mock_azure:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_azure.return_value = mock_instance

        provider = AzureOpenAIProvider(model_config, provider_config)

        mock_instance.with_structured_output.assert_called_once()
        assert provider._use_prompt_parsing is False
