"""Tests for the OpenAI-on-Bedrock provider (OpenAI-compatible endpoint)."""

from unittest.mock import Mock, patch

import pytest
from openai import RateLimitError

from codereview.config.models import (
    BedrockOpenAIConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport, ReviewMetrics
from codereview.providers.bedrock_openai import BedrockOpenAIProvider


@pytest.fixture
def gpt55_model_config():
    """The real GPT-5.5-on-Bedrock registry shape.

    Reasoning model: no temperature/top_p, Responses API, and
    supports_tool_use=False (adaptive thinking returns reasoning-only
    responses that break the forced tool_choice, so we parse JSON via prompt).
    """
    return ModelConfig(
        id="gpt5.5-bedrock",
        full_id="openai.gpt-5.5",
        name="GPT-5.5 (Bedrock)",
        aliases=["gpt-bedrock"],
        pricing=PricingConfig(input_per_million=2.50, output_per_million=15.00),
        inference_params=InferenceParams(max_output_tokens=128000),
        use_responses_api=True,
        context_window=400000,
        supports_tool_use=False,
    )


@pytest.fixture
def tooluse_model_config():
    """A hypothetical tool-use-capable model on the same endpoint.

    Exercises the with_structured_output path so both branches stay covered
    if a future model on this endpoint can tool-call while reasoning.
    """
    return ModelConfig(
        id="oai-tooluse",
        full_id="openai.tooluse",
        name="OAI tool-use (Bedrock)",
        aliases=[],
        pricing=PricingConfig(input_per_million=1.0, output_per_million=2.0),
        inference_params=InferenceParams(temperature=0.3, max_output_tokens=16384),
        supports_tool_use=True,
    )


@pytest.fixture
def provider_config():
    return BedrockOpenAIConfig(
        api_key="test-bedrock-key-1234567890abcdef",
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
        request_timeout=300,
    )


@pytest.fixture
def mock_report():
    return CodeReviewReport(
        summary="Bedrock OpenAI test analysis",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=0, critical_issues=0),
        issues=[],
        system_design_insights="Looks fine",
        recommendations=["Ship it"],
        improvement_suggestions=[],
    )


def test_provider_initialization(gpt55_model_config, provider_config):
    with patch("codereview.providers.bedrock_openai.ChatOpenAI"):
        provider = BedrockOpenAIProvider(gpt55_model_config, provider_config)
        assert provider is not None
        # Reasoning model: no temperature/top_p.
        assert provider.temperature is None
        assert provider.top_p is None
        assert provider.max_tokens == 128000


def test_gpt55_uses_prompt_parsing_and_responses_api(
    gpt55_model_config, provider_config
):
    """Regression (field failure): GPT-5.5 must NOT use tool-calling structured
    output. Its adaptive server-side thinking returns reasoning-only responses
    with no `parsed` field, breaking the forced tool_choice that
    .with_structured_output() sets ("Structured Output response does not have a
    'parsed' field"). It must route through PydanticOutputParser instead, and
    enable the Responses API.
    """
    with patch("codereview.providers.bedrock_openai.ChatOpenAI") as mock_openai:
        mock_instance = Mock()
        mock_openai.return_value = mock_instance

        provider = BedrockOpenAIProvider(gpt55_model_config, provider_config)

        # No tool-calling structured output requested.
        mock_instance.with_structured_output.assert_not_called()
        assert provider._use_prompt_parsing is True
        # Chain ends with the PydanticOutputParser.
        assert provider.chain.last is provider._output_parser

        kwargs = mock_openai.call_args.kwargs
        # Responses API enabled; temperature/top_p NOT forwarded (reasoning).
        assert kwargs["use_responses_api"] is True
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        # Wire-level model id from full_id; bearer key wrapped in SecretStr.
        assert kwargs["model"] == "openai.gpt-5.5"
        assert (
            kwargs["api_key"].get_secret_value() == "test-bedrock-key-1234567890abcdef"
        )
        assert (
            kwargs["base_url"]
            == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
        )


def test_tooluse_model_uses_structured_output(tooluse_model_config, provider_config):
    """A supports_tool_use=True model keeps tool-calling structured output."""
    with patch("codereview.providers.bedrock_openai.ChatOpenAI") as mock_openai:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_openai.return_value = mock_instance

        provider = BedrockOpenAIProvider(tooluse_model_config, provider_config)

        assert provider._use_prompt_parsing is False
        mock_instance.with_structured_output.assert_called_once_with(
            CodeReviewReport, include_raw=True
        )
        # Non-Responses-API path forwards temperature.
        kwargs = mock_openai.call_args.kwargs
        assert "use_responses_api" not in kwargs
        assert kwargs["temperature"] == 0.3


def test_falls_back_to_id_when_full_id_missing(provider_config):
    """When full_id is absent, the wire-level model is the bare id."""
    config_no_full_id = ModelConfig(
        id="openai.bare-id",
        full_id=None,
        name="Bare id",
        aliases=[],
        pricing=PricingConfig(input_per_million=0.5, output_per_million=2.0),
        supports_tool_use=False,
    )
    with patch("codereview.providers.bedrock_openai.ChatOpenAI") as mock_openai:
        mock_openai.return_value = Mock()
        BedrockOpenAIProvider(config_no_full_id, provider_config)
        assert mock_openai.call_args.kwargs["model"] == "openai.bare-id"


def test_analyze_batch(gpt55_model_config, provider_config, mock_report):
    with patch("codereview.providers.bedrock_openai.ChatOpenAI"):
        provider = BedrockOpenAIProvider(gpt55_model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report

        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"x.py": "print('hi')"},
        )

        assert isinstance(result, CodeReviewReport)
        assert result.summary == "Bedrock OpenAI test analysis"


def test_retry_on_rate_limit(gpt55_model_config, provider_config, mock_report):
    """RateLimitError is retryable with Retry-After honored."""
    with (
        patch("codereview.providers.bedrock_openai.ChatOpenAI"),
        patch("time.sleep") as mock_sleep,
    ):
        provider = BedrockOpenAIProvider(gpt55_model_config, provider_config)

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "3"}
        rate_err = RateLimitError(
            "rate limited",
            response=mock_response,
            body={"error": {"message": "rate limited"}},
        )

        provider.chain = Mock()
        provider.chain.invoke.side_effect = [rate_err, mock_report]

        result = provider.analyze_batch(1, 1, {"x.py": "code"})
        assert result == mock_report
        mock_sleep.assert_called_with(3.0)


def test_validate_credentials_rejects_empty_key():
    """BedrockOpenAIConfig requires a non-empty api_key (min_length=1)."""
    with pytest.raises(Exception):
        BedrockOpenAIConfig(
            api_key="",
            base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
        )


def test_validate_credentials_placeholder(gpt55_model_config):
    """A literal placeholder key is rejected by validate_credentials."""
    config = BedrockOpenAIConfig(
        api_key="placeholder",
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
    )
    with patch("codereview.providers.bedrock_openai.ChatOpenAI"):
        provider = BedrockOpenAIProvider(gpt55_model_config, config)
        result = provider.validate_credentials()
        assert result.valid is False


def test_validate_credentials_happy_path(gpt55_model_config, provider_config):
    with patch("codereview.providers.bedrock_openai.ChatOpenAI"):
        provider = BedrockOpenAIProvider(gpt55_model_config, provider_config)
        result = provider.validate_credentials()
        assert result.valid is True


def test_non_https_base_fails_closed_at_construction(gpt55_model_config):
    """A cleartext base_url must fail closed when the client is built, before
    any network call — not just when validate_credentials is invoked. This is
    the stronger guarantee: the bearer key can never reach an http:// endpoint
    even if a caller skips validation.
    """
    config = BedrockOpenAIConfig(
        api_key="test-bedrock-key-1234567890abcdef",
        base_url="http://insecure.example.com/openai/v1",
    )
    with patch("codereview.providers.bedrock_openai.ChatOpenAI"):
        with pytest.raises(ValueError, match="must use HTTPS"):
            BedrockOpenAIProvider(gpt55_model_config, config)


# --- Per-model Region override -------------------------------------------
#
# `base_url` is a PROVIDER-level setting but bedrock-mantle's Regions are
# per-model and non-overlapping: GPT-6 Astra is served from us-west-2 only,
# GPT-5.6 Sol is In-Region us-east-1/us-east-2 only. One OPENAI_BASE_URL
# therefore cannot reach both — pointing at the wrong Region returns a 404
# naming the model id ("The model 'openai.gpt-6-astra' does not exist")
# rather than falling back, and it does so on every batch of the run.
#
# A model entry may carry `region:` to rewrite the Region component of the
# configured URL. The Region, not the whole URL, is what varies per the model
# card, and deriving from the configured value (rather than a hardcoded host
# template) keeps OPENAI_BASE_URL authoritative for scheme, host and path.


@pytest.fixture
def mantle_provider_config():
    """The real bedrock-mantle endpoint shape, pointed at Sol's Region."""
    return BedrockOpenAIConfig(
        api_key="test-bedrock-key-1234567890abcdef",
        base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        request_timeout=300,
    )


def _astra_config(**overrides):
    """The GPT-6 Astra registry shape: us-west-2-only on bedrock-mantle."""
    params = {
        "id": "gpt6-astra-bedrock",
        "full_id": "openai.gpt-6-astra",
        "name": "GPT-6 Astra (Bedrock)",
        "aliases": ["gpt6"],
        "pricing": PricingConfig(input_per_million=11.0, output_per_million=55.0),
        "inference_params": InferenceParams(max_output_tokens=128000),
        "use_responses_api": True,
        "supports_tool_use": False,
        "region": "us-west-2",
    }
    params.update(overrides)
    return ModelConfig(**params)


def _built_base_url(model_config, provider_config):
    """The base_url the ChatOpenAI client was actually constructed with."""
    with patch("codereview.providers.bedrock_openai.ChatOpenAI") as mock_openai:
        mock_openai.return_value = Mock()
        BedrockOpenAIProvider(model_config, provider_config)
        return mock_openai.call_args.kwargs["base_url"]


def test_model_region_rewrites_the_endpoint_region(mantle_provider_config):
    """Regression (field failure, 2026-09-13): every batch of an Astra run
    404'd with "The model 'openai.gpt-6-astra' does not exist" because
    OPENAI_BASE_URL named us-east-2 (Sol's Region). A model that declares its
    Region must be reached there regardless of what the env var says.
    """
    url = _built_base_url(_astra_config(), mantle_provider_config)
    assert url == "https://bedrock-mantle.us-west-2.api.aws/openai/v1"


def test_no_model_region_leaves_the_configured_url_untouched(mantle_provider_config):
    """The override is opt-in: an entry without `region` must behave exactly as
    before, so adding this feature can't move any other model's endpoint.
    """
    url = _built_base_url(_astra_config(region=None), mantle_provider_config)
    assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"


def test_region_already_correct_is_a_no_op(mantle_provider_config):
    """Rewriting to the Region already in the URL must be idempotent."""
    url = _built_base_url(_astra_config(region="us-east-2"), mantle_provider_config)
    assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"


def test_region_override_passes_through_a_host_with_no_region_label():
    """Fail SOFT, not closed. A self-hosted or otherwise non-AWS endpoint has no
    Region component to swap; refusing to run would break a working custom
    endpoint over a cosmetic mismatch, so the configured URL is used as-is.
    """
    config = BedrockOpenAIConfig(
        api_key="test-bedrock-key-1234567890abcdef",
        base_url="https://gateway.internal.example.com/openai/v1",
    )
    url = _built_base_url(_astra_config(), config)
    assert url == "https://gateway.internal.example.com/openai/v1"


def test_region_override_does_not_bypass_the_https_gate():
    """The cleartext gate must run on the URL actually used, so a per-model
    Region can never become a way to smuggle the bearer key over http://.
    """
    config = BedrockOpenAIConfig(
        api_key="test-bedrock-key-1234567890abcdef",
        base_url="http://bedrock-mantle.us-east-2.api.aws/openai/v1",
    )
    with patch("codereview.providers.bedrock_openai.ChatOpenAI"):
        with pytest.raises(ValueError, match="must use HTTPS"):
            BedrockOpenAIProvider(_astra_config(), config)
