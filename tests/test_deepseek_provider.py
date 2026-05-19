"""Tests for the DeepSeek provider (langchain-deepseek)."""

from unittest.mock import Mock, patch

import pytest
from openai import RateLimitError

from codereview.config.models import (
    DeepSeekConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport, ReviewMetrics
from codereview.providers.deepseek import DeepSeekProvider


@pytest.fixture
def model_config():
    return ModelConfig(
        id="deepseek-v4-pro",
        full_id="deepseek-v4-pro",
        name="DeepSeek-V4-Pro",
        aliases=["dsv4-pro"],
        pricing=PricingConfig(input_per_million=1.74, output_per_million=3.48),
        inference_params=InferenceParams(
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=16384,
        ),
    )


@pytest.fixture
def flash_model_config():
    return ModelConfig(
        id="deepseek-v4-flash",
        full_id="deepseek-v4-flash",
        name="DeepSeek-V4-Flash",
        aliases=["dsv4-flash"],
        pricing=PricingConfig(input_per_million=0.14, output_per_million=0.28),
        inference_params=InferenceParams(
            temperature=0.3,
            max_output_tokens=16384,
        ),
    )


@pytest.fixture
def provider_config():
    return DeepSeekConfig(
        api_key="test-deepseek-key-1234567890abcdef",
        api_base="https://api.deepseek.com",
        request_timeout=300,
    )


@pytest.fixture
def mock_report():
    return CodeReviewReport(
        summary="DeepSeek test analysis",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=0, critical_issues=0),
        issues=[],
        system_design_insights="Looks fine",
        recommendations=["Ship it"],
        improvement_suggestions=[],
    )


def test_deepseek_provider_initialization(model_config, provider_config):
    with patch("codereview.providers.deepseek.ChatDeepSeek"):
        provider = DeepSeekProvider(model_config, provider_config)
        assert provider is not None
        assert provider.temperature == 0.3
        assert provider.top_p == 0.95
        assert provider.max_tokens == 16384


def test_deepseek_uses_chatdeepseek_with_api_base(model_config, provider_config):
    """DeepSeek integrates via langchain-deepseek's ChatDeepSeek class."""
    with patch("codereview.providers.deepseek.ChatDeepSeek") as mock_ds:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ds.return_value = mock_instance

        DeepSeekProvider(model_config, provider_config)

        mock_ds.assert_called_once()
        kwargs = mock_ds.call_args.kwargs

        # Wire-level model name from full_id
        assert kwargs["model"] == "deepseek-v4-pro"
        # ChatDeepSeek uses `api_base` (not `base_url`)
        assert kwargs["api_base"] == "https://api.deepseek.com"
        # api_key wrapped in SecretStr to keep it out of repr/logs
        assert (
            kwargs["api_key"].get_secret_value() == "test-deepseek-key-1234567890abcdef"
        )
        # Inference params forwarded
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.95
        assert kwargs["max_tokens"] == 16384

        # Tool-calling structured output is used (V4-Pro supports it)
        mock_instance.with_structured_output.assert_called_once_with(
            CodeReviewReport, include_raw=True
        )


def test_deepseek_disables_thinking_by_default(model_config, provider_config):
    """V4-Pro routes to deepseek-reasoner server-side; reasoner mode rejects
    tool_choice="auto" (which langchain forces for with_structured_output).

    The provider must send thinking={"type": "disabled"} via extra_body so
    the request becomes a plain chat completion that accepts tool calls.
    """
    with patch("codereview.providers.deepseek.ChatDeepSeek") as mock_ds:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ds.return_value = mock_instance

        DeepSeekProvider(model_config, provider_config)

        kwargs = mock_ds.call_args.kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_thinking_override_enables_reasoner(provider_config):
    """Explicit `thinking: enabled` in inference_params turns on reasoning.

    Note: this combination requires *not* using structured output, since
    reasoner mode rejects forced tool_choice. Operators who set this are
    on their own re: schema validation.
    """
    cfg = ModelConfig(
        id="deepseek-v4-pro",
        full_id="deepseek-v4-pro",
        name="DeepSeek-V4-Pro (thinking)",
        aliases=[],
        pricing=PricingConfig(input_per_million=1.74, output_per_million=3.48),
        inference_params=InferenceParams(
            temperature=0.3,
            max_output_tokens=16384,
            thinking="enabled",
        ),
    )
    with patch("codereview.providers.deepseek.ChatDeepSeek") as mock_ds:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ds.return_value = mock_instance

        DeepSeekProvider(cfg, provider_config)
        kwargs = mock_ds.call_args.kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_deepseek_v4_flash_uses_correct_model_id(flash_model_config, provider_config):
    """V4-Flash routes to its own wire-level model name."""
    with patch("codereview.providers.deepseek.ChatDeepSeek") as mock_ds:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ds.return_value = mock_instance

        DeepSeekProvider(flash_model_config, provider_config)
        assert mock_ds.call_args.kwargs["model"] == "deepseek-v4-flash"


def test_deepseek_falls_back_to_id_when_full_id_missing(provider_config):
    """When full_id is absent the bare id is used as wire model name."""
    config_no_full_id = ModelConfig(
        id="deepseek-coder",
        full_id=None,
        name="DeepSeek Coder",
        aliases=[],
        pricing=PricingConfig(input_per_million=0.5, output_per_million=2.0),
    )

    with patch("codereview.providers.deepseek.ChatDeepSeek") as mock_ds:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ds.return_value = mock_instance

        DeepSeekProvider(config_no_full_id, provider_config)
        assert mock_ds.call_args.kwargs["model"] == "deepseek-coder"


def test_deepseek_analyze_batch(model_config, provider_config, mock_report):
    with patch("codereview.providers.deepseek.ChatDeepSeek") as mock_ds:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ds.return_value = mock_instance

        provider = DeepSeekProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report

        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"x.py": "print('hi')"},
        )

        assert isinstance(result, CodeReviewReport)
        assert result.summary == "DeepSeek test analysis"


def test_deepseek_token_tracking(model_config, provider_config, mock_report):
    """DeepSeek mirrors OpenAI's response_metadata.token_usage shape."""
    with patch("codereview.providers.deepseek.ChatDeepSeek") as mock_ds:
        report_with_metadata = Mock(spec=CodeReviewReport)
        report_with_metadata.response_metadata = {
            "token_usage": {"prompt_tokens": 250, "completion_tokens": 90}
        }
        report_with_metadata.model_dump_json.return_value = "{}"
        for attr in (
            "summary",
            "metrics",
            "issues",
            "system_design_insights",
            "recommendations",
            "improvement_suggestions",
        ):
            setattr(report_with_metadata, attr, getattr(mock_report, attr))

        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ds.return_value = mock_instance

        provider = DeepSeekProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = report_with_metadata

        provider.analyze_batch(1, 1, {"x.py": "code"})
        assert provider.total_input_tokens == 250
        assert provider.total_output_tokens == 90


def test_deepseek_retry_on_rate_limit(model_config, provider_config, mock_report):
    """RateLimitError is retryable with Retry-After honored."""
    with (
        patch("codereview.providers.deepseek.ChatDeepSeek") as mock_ds,
        patch("time.sleep") as mock_sleep,
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ds.return_value = mock_instance

        provider = DeepSeekProvider(model_config, provider_config)

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "5"}
        rate_err = RateLimitError(
            "rate limited",
            response=mock_response,
            body={"error": {"message": "rate limited"}},
        )

        provider.chain = Mock()
        provider.chain.invoke.side_effect = [rate_err, mock_report]

        result = provider.analyze_batch(1, 1, {"x.py": "code"})
        assert result == mock_report
        mock_sleep.assert_called_with(5.0)


def test_deepseek_validate_credentials_empty_key():
    """ZAIConfig-style: empty api_key fails Pydantic validation outright."""
    with pytest.raises(Exception):
        DeepSeekConfig(api_key="")


def test_deepseek_validate_credentials_placeholder(model_config):
    config = DeepSeekConfig(api_key="your-deepseek-api-key-here")
    with patch("codereview.providers.deepseek.ChatDeepSeek"):
        provider = DeepSeekProvider(model_config, config)
        result = provider.validate_credentials()
        assert result.valid is False


def test_deepseek_validate_credentials_happy_path(model_config, provider_config):
    with patch("codereview.providers.deepseek.ChatDeepSeek"):
        provider = DeepSeekProvider(model_config, provider_config)
        result = provider.validate_credentials()
        assert result.valid is True


def test_deepseek_validate_rejects_non_https_base(model_config):
    config = DeepSeekConfig(api_key="test-key", api_base="http://insecure.example.com")
    with patch("codereview.providers.deepseek.ChatDeepSeek"):
        provider = DeepSeekProvider(model_config, config)
        result = provider.validate_credentials()
        assert result.valid is False
