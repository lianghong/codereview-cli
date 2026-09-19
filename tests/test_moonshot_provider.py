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

    The real kimi-k3 model uses ``supports_tool_use=False`` (see
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
def k3_model_config():
    """Real kimi-k3 config: prompt-based JSON parsing path.

    Mirrors the shipped entry: no temperature/top_p (K3 fixes both
    server-side), ``reasoning_effort`` pinned down from the card's ``max``
    default, and the 32K output budget.
    """
    return ModelConfig(
        id="kimi-k3",
        full_id="kimi-k3",
        name="Kimi K3 (Moonshot)",
        aliases=["kimi"],
        pricing=PricingConfig(input_per_million=3.00, output_per_million=15.00),
        inference_params=InferenceParams(
            reasoning_effort="high",
            max_output_tokens=32768,
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


def test_moonshot_k3_uses_prompt_parsing(k3_model_config, provider_config):
    """kimi-k3 (supports_tool_use=False) skips tool-calling structured
    output and uses PydanticOutputParser instead — Moonshot's server
    rejects tool_choice='specified' while thinking mode is enabled, and K3
    cannot turn thinking off, so the HTTP 400 is unconditional."""
    with patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms:
        mock_instance = Mock()
        mock_ms.return_value = mock_instance

        provider = MoonshotProvider(k3_model_config, provider_config)

        # No tool-calling structured output should have been requested.
        mock_instance.with_structured_output.assert_not_called()
        assert provider._use_prompt_parsing is True
        # Chain ends with the PydanticOutputParser so the model's text
        # response is converted into a CodeReviewReport.
        assert provider.chain.last is provider._output_parser


def test_reasoning_effort_reaches_the_client(k3_model_config, provider_config):
    """``inference_params.reasoning_effort`` must be forwarded to ChatMoonshot.

    Parsed onto ``InferenceParams`` is not the same as sent. K3's card defaults
    to ``max`` effort and Moonshot bills reasoning inside ``completion_tokens``
    at the output rate, so an unforwarded ``high`` is a silent cost regression
    *and* eats the output budget the review report needs — the same
    invisible-knob shape that made ``NVIDIAConfig.polling_timeout`` dead config.
    """
    with patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms:
        mock_ms.return_value = Mock()

        MoonshotProvider(k3_model_config, provider_config)

        kwargs = mock_ms.call_args.kwargs
        assert kwargs["reasoning_effort"] == "high"
        # K3 fixes temperature/top_p server-side, so neither may be sent.
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert kwargs["max_tokens"] == 32768


def test_reasoning_effort_is_omitted_when_unset(model_config, provider_config):
    """No ``reasoning_effort`` in the YAML means the kwarg is not sent at all.

    Passing ``None`` through would override the server-side default with a
    value the model never asked for; and ``inference_params`` is Optional, so
    the resolution has to survive a config that carries none.
    """
    bare = ModelConfig(
        id="kimi-bare",
        full_id="kimi-bare",
        name="Kimi (no inference params)",
        aliases=[],
        pricing=PricingConfig(input_per_million=1.0, output_per_million=2.0),
    )

    for config in (model_config, bare):
        with patch("codereview.providers.moonshot.ChatMoonshot") as mock_ms:
            mock_ms.return_value = Mock()

            MoonshotProvider(config, provider_config)

            assert "reasoning_effort" not in mock_ms.call_args.kwargs


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


@pytest.mark.parametrize(
    "placeholder",
    [
        "your-kimi-api-key-here",
        "your-moonshot-key",  # the exact string README documents
        "placeholder",
        "  Your-Moonshot-Key  ",  # whitespace + case must still be rejected
    ],
)
def test_moonshot_validate_credentials_placeholder(model_config, placeholder):
    config = MoonshotConfig(api_key=placeholder)
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        provider = MoonshotProvider(model_config, config)
        result = provider.validate_credentials()
        assert result.valid is False


def test_moonshot_validate_credentials_happy_path(model_config, provider_config):
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        provider = MoonshotProvider(model_config, provider_config)
        result = provider.validate_credentials()
        assert result.valid is True


def test_moonshot_non_https_base_fails_closed_at_construction(model_config):
    """A cleartext base_url must fail closed when the client is built, before
    any network call — KIMI_API_KEY can never reach an http:// endpoint."""
    config = MoonshotConfig(api_key="test-key", base_url="http://insecure.example.com")
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        with pytest.raises(ValueError, match="must use HTTPS"):
            MoonshotProvider(model_config, config)
