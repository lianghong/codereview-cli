"""Tests for the Moonshot (Kimi) provider via langchain-moonshot."""

from unittest.mock import Mock, patch

import pytest
from openai import RateLimitError

from codereview.config.models import (
    InferenceParams,
    ModelConfig,
    MoonshotConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport, ReviewMetrics
from codereview.providers.moonshot import MoonshotProvider


@pytest.fixture
def model_config():
    """Tool-use-capable Moonshot model fixture (hypothetical future Kimi).

    The real kimi-k2.6 model uses ``supports_tool_use=False`` (see
    ``models.yaml``); this fixture exercises the tool-calling path
    explicitly so both branches of ``_create_model`` stay covered.
    """
    return ModelConfig(
        id="kimi-tooluse",
        full_id="kimi-tooluse",
        name="Kimi (tool-use)",
        aliases=["kimi"],
        pricing=PricingConfig(input_per_million=0.60, output_per_million=2.50),
        inference_params=InferenceParams(
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=16384,
        ),
        supports_tool_use=True,
    )


@pytest.fixture
def k26_model_config():
    """Real kimi-k2.6 config: prompt-based JSON parsing path."""
    return ModelConfig(
        id="kimi-k2.6",
        full_id="kimi-k2.6",
        name="Kimi K2.6 (Moonshot)",
        aliases=["kimi"],
        pricing=PricingConfig(input_per_million=0.60, output_per_million=2.50),
        inference_params=InferenceParams(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=16384,
        ),
        supports_tool_use=False,
    )


@pytest.fixture
def provider_config():
    return MoonshotConfig(
        api_key="test-kimi-key-1234567890abcdef",
        base_url="https://api.moonshot.cn/v1",
        request_timeout=300,
    )


@pytest.fixture
def mock_report():
    return CodeReviewReport(
        summary="Moonshot test analysis",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=0, critical_issues=0),
        issues=[],
        system_design_insights="Looks fine",
        recommendations=["Ship it"],
        improvement_suggestions=[],
    )


def test_moonshot_provider_initialization(model_config, provider_config):
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        provider = MoonshotProvider(model_config, provider_config)
        assert provider is not None
        assert provider.temperature == 0.3
        assert provider.top_p == 0.95
        assert provider.max_tokens == 16384


def test_moonshot_uses_chatmoonshot_with_base_url(model_config, provider_config):
    """Moonshot integrates via langchain-moonshot's ChatMoonshot class."""
    with patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ms.return_value = mock_instance

        MoonshotProvider(model_config, provider_config)

        mock_ms.assert_called_once()
        kwargs = mock_ms.call_args.kwargs

        # Wire-level model name from full_id
        assert kwargs["model"] == "kimi-tooluse"
        # ChatMoonshot accepts base_url (alias of api_base)
        assert kwargs["base_url"] == "https://api.moonshot.cn/v1"
        # api_key wrapped in SecretStr to keep it out of repr/logs
        assert kwargs["api_key"].get_secret_value() == "test-kimi-key-1234567890abcdef"
        # Inference params forwarded
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.95
        assert kwargs["max_tokens"] == 16384

        # Tool-calling structured output is used (K2.6 supports it via
        # BaseChatOpenAI inheritance).
        mock_instance.with_structured_output.assert_called_once_with(
            CodeReviewReport, include_raw=True
        )


def test_moonshot_k26_uses_prompt_parsing(k26_model_config, provider_config):
    """kimi-k2.6 (supports_tool_use=False) skips tool-calling structured
    output and uses PydanticOutputParser instead — Moonshot's server
    rejects tool_choice='specified' while thinking mode is enabled."""
    with patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms:
        mock_instance = Mock()
        mock_ms.return_value = mock_instance

        provider = MoonshotProvider(k26_model_config, provider_config)

        # No tool-calling structured output should have been requested.
        mock_instance.with_structured_output.assert_not_called()
        assert provider._use_prompt_parsing is True
        # Chain ends with the PydanticOutputParser so the model's text
        # response is converted into a CodeReviewReport.
        assert provider.chain.last is provider._output_parser


def test_moonshot_falls_back_to_id_when_full_id_missing(provider_config):
    """When full_id is absent the bare id is used as wire model name."""
    config_no_full_id = ModelConfig(
        id="kimi-k2.5",
        full_id=None,
        name="Kimi K2.5",
        aliases=[],
        pricing=PricingConfig(input_per_million=0.5, output_per_million=2.0),
    )

    with patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ms.return_value = mock_instance

        MoonshotProvider(config_no_full_id, provider_config)
        assert mock_ms.call_args.kwargs["model"] == "kimi-k2.5"


def test_moonshot_analyze_batch(model_config, provider_config, mock_report):
    with patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ms.return_value = mock_instance

        provider = MoonshotProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report

        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"x.py": "print('hi')"},
        )

        assert isinstance(result, CodeReviewReport)
        assert result.summary == "Moonshot test analysis"


def test_moonshot_token_tracking(model_config, provider_config, mock_report):
    """Moonshot mirrors OpenAI's response_metadata.token_usage shape."""
    with patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms:
        report_with_metadata = Mock(spec=CodeReviewReport)
        report_with_metadata.response_metadata = {
            "token_usage": {"prompt_tokens": 180, "completion_tokens": 70}
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
        mock_ms.return_value = mock_instance

        provider = MoonshotProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = report_with_metadata

        provider.analyze_batch(1, 1, {"x.py": "code"})
        assert provider.total_input_tokens == 180
        assert provider.total_output_tokens == 70


def test_moonshot_retry_on_rate_limit(model_config, provider_config, mock_report):
    """RateLimitError is retryable with Retry-After honored."""
    with (
        patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms,
        patch("time.sleep") as mock_sleep,
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_ms.return_value = mock_instance

        provider = MoonshotProvider(model_config, provider_config)

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "4"}
        rate_err = RateLimitError(
            "rate limited",
            response=mock_response,
            body={"error": {"message": "rate limited"}},
        )

        provider.chain = Mock()
        provider.chain.invoke.side_effect = [rate_err, mock_report]

        result = provider.analyze_batch(1, 1, {"x.py": "code"})
        assert result == mock_report
        mock_sleep.assert_called_with(4.0)


def test_moonshot_validate_credentials_empty_key_rejected():
    """Empty api_key fails Pydantic validation outright."""
    with pytest.raises(Exception):
        MoonshotConfig(api_key="")


def test_moonshot_validate_credentials_placeholder(model_config):
    config = MoonshotConfig(api_key="your-kimi-api-key-here")
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        provider = MoonshotProvider(model_config, config)
        result = provider.validate_credentials()
        assert result.valid is False


def test_moonshot_validate_credentials_happy_path(model_config, provider_config):
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        provider = MoonshotProvider(model_config, provider_config)
        result = provider.validate_credentials()
        assert result.valid is True


def test_moonshot_validate_rejects_non_https_base(model_config):
    config = MoonshotConfig(api_key="test-key", base_url="http://insecure.example.com")
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        provider = MoonshotProvider(model_config, config)
        result = provider.validate_credentials()
        assert result.valid is False
