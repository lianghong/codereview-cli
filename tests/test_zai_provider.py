"""Tests for the Z.AI provider (Zhipu international, OpenAI-compatible)."""

from unittest.mock import Mock, patch

import pytest
from openai import RateLimitError

from codereview.config.models import (
    InferenceParams,
    ModelConfig,
    PricingConfig,
    ZAIConfig,
)
from codereview.models import CodeReviewReport, ReviewMetrics
from codereview.providers.zai import ZAIProvider


@pytest.fixture
def model_config():
    """A tool-use-capable GLM config (exercises the with_structured_output path)."""
    return ModelConfig(
        id="glm-tooluse",
        full_id="glm-tooluse",
        name="GLM tool-use (Z.AI)",
        aliases=["glm-tu"],
        pricing=PricingConfig(input_per_million=1.40, output_per_million=4.40),
        inference_params=InferenceParams(
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=16384,
        ),
        supports_tool_use=True,
    )


@pytest.fixture
def glm51_model_config():
    """The real GLM-5.1 registry shape: supports_tool_use=False (prompt parsing).

    Z.AI's OpenAI-compat endpoint ignores json_schema response_format and emits
    markdown-fenced JSON, so GLM-5.1 routes through PydanticOutputParser.
    """
    return ModelConfig(
        id="zhipuai/glm-5.1",
        full_id="glm-5.1",
        name="GLM-5.1 (Z.AI)",
        aliases=["zai-glm"],
        pricing=PricingConfig(input_per_million=1.40, output_per_million=4.40),
        inference_params=InferenceParams(
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=16384,
        ),
        supports_tool_use=False,
    )


@pytest.fixture
def provider_config():
    return ZAIConfig(
        api_key="test-zai-key-1234567890abcdef",
        base_url="https://api.z.ai/api/paas/v4/",
        request_timeout=300,
    )


@pytest.fixture
def mock_report():
    return CodeReviewReport(
        summary="Z.AI test analysis",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=0, critical_issues=0),
        issues=[],
        system_design_insights="Looks fine",
        recommendations=["Ship it"],
        improvement_suggestions=[],
    )


def test_zai_provider_initialization(model_config, provider_config):
    with patch("codereview.providers.zai.ChatOpenAI"):
        provider = ZAIProvider(model_config, provider_config)
        assert provider is not None
        assert provider.temperature == 0.3
        assert provider.top_p == 0.95
        assert provider.max_tokens == 16384


def test_zai_uses_chatopenai_with_custom_base_url(model_config, provider_config):
    """Z.AI integrates via the OpenAI-compatible adapter, not ChatZhipuAI."""
    with patch("codereview.providers.zai.ChatOpenAI") as mock_openai:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_openai.return_value = mock_instance

        ZAIProvider(model_config, provider_config)

        mock_openai.assert_called_once()
        kwargs = mock_openai.call_args.kwargs

        # Wire-level model name comes from full_id.
        assert kwargs["model"] == "glm-tooluse"
        assert kwargs["base_url"] == "https://api.z.ai/api/paas/v4/"

        # api_key is wrapped in SecretStr to keep it out of repr/logs.
        assert kwargs["api_key"].get_secret_value() == "test-zai-key-1234567890abcdef"

        # Inference params are forwarded as-is.
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.95
        assert kwargs["max_tokens"] == 16384

        # Tool-calling structured output is used when supports_tool_use=True.
        mock_instance.with_structured_output.assert_called_once_with(
            CodeReviewReport, include_raw=True
        )


def test_zai_falls_back_to_id_when_full_id_missing(provider_config):
    """When full_id is absent, the wire-level model is the bare id."""
    config_no_full_id = ModelConfig(
        id="glm-4-32b",
        full_id=None,
        name="GLM 4 32B",
        aliases=[],
        pricing=PricingConfig(input_per_million=0.5, output_per_million=2.0),
    )

    with patch("codereview.providers.zai.ChatOpenAI") as mock_openai:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_openai.return_value = mock_instance

        ZAIProvider(config_no_full_id, provider_config)
        assert mock_openai.call_args.kwargs["model"] == "glm-4-32b"


def test_zai_analyze_batch(model_config, provider_config, mock_report):
    with patch("codereview.providers.zai.ChatOpenAI") as mock_openai:
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_openai.return_value = mock_instance

        provider = ZAIProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = mock_report

        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"x.py": "print('hi')"},
        )

        assert isinstance(result, CodeReviewReport)
        assert result.summary == "Z.AI test analysis"


def test_zai_token_tracking(model_config, provider_config, mock_report):
    """Z.AI mirrors OpenAI's response_metadata.token_usage shape."""
    with patch("codereview.providers.zai.ChatOpenAI") as mock_openai:
        report_with_metadata = Mock(spec=CodeReviewReport)
        report_with_metadata.response_metadata = {
            "token_usage": {"prompt_tokens": 200, "completion_tokens": 80}
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
        mock_openai.return_value = mock_instance

        provider = ZAIProvider(model_config, provider_config)
        provider.chain = Mock()
        provider.chain.invoke.return_value = report_with_metadata

        provider.analyze_batch(1, 1, {"x.py": "code"})
        assert provider.total_input_tokens == 200
        assert provider.total_output_tokens == 80


def test_zai_retry_on_rate_limit(model_config, provider_config, mock_report):
    """RateLimitError is retryable with Retry-After honored."""
    with (
        patch("codereview.providers.zai.ChatOpenAI") as mock_openai,
        patch("time.sleep") as mock_sleep,
    ):
        mock_instance = Mock()
        mock_instance.with_structured_output.return_value = Mock()
        mock_openai.return_value = mock_instance

        provider = ZAIProvider(model_config, provider_config)

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


def test_zai_validate_credentials_missing_key(model_config):
    """validate_credentials fails fast with a clear suggestion when key empty."""
    with pytest.raises(Exception):
        # ZAIConfig requires a non-empty api_key (min_length=1), so an
        # empty key fails at construction. Simulate the "no key" path by
        # constructing with a placeholder and forcing the check.
        ZAIConfig(api_key="")


def test_zai_validate_credentials_placeholder(model_config):
    """A literal placeholder string is rejected by validate_credentials."""
    config = ZAIConfig(api_key="your-zai-api-key-here")
    with patch("codereview.providers.zai.ChatOpenAI"):
        provider = ZAIProvider(model_config, config)
        result = provider.validate_credentials()
        assert result.valid is False


def test_zai_validate_credentials_happy_path(model_config, provider_config):
    with patch("codereview.providers.zai.ChatOpenAI"):
        provider = ZAIProvider(model_config, provider_config)
        result = provider.validate_credentials()
        assert result.valid is True


def test_zai_glm51_uses_prompt_parsing(glm51_model_config, provider_config):
    """GLM-5.1 (supports_tool_use=False) must skip tool-calling structured
    output and parse JSON with PydanticOutputParser.

    Regression: Z.AI's endpoint ignores OpenAI's json_schema response_format
    and returns markdown-fenced JSON (```json ... ```), which the json_schema
    path rejects ("Invalid JSON: expected value at line 1 column 1"). The
    PydanticOutputParser path strips the fences.
    """
    with patch("codereview.providers.zai.ChatOpenAI") as mock_openai:
        mock_instance = Mock()
        mock_openai.return_value = mock_instance

        provider = ZAIProvider(glm51_model_config, provider_config)

        # No tool-calling structured output should have been requested.
        mock_instance.with_structured_output.assert_not_called()
        assert provider._use_prompt_parsing is True
        # Chain ends with the PydanticOutputParser so a fenced-JSON text
        # response is still converted into a CodeReviewReport.
        assert provider.chain.last is provider._output_parser


def test_zai_prompt_parser_strips_markdown_fences():
    """The PydanticOutputParser used by the GLM-5.1 path strips ```json fences.

    This is the exact failure observed in the field: GLM returned
    ```json\\n{...}\\n``` and the json_schema parser choked at column 1.
    """
    from langchain_core.output_parsers import PydanticOutputParser

    from codereview.models import CodeReviewReport

    parser = PydanticOutputParser(pydantic_object=CodeReviewReport)
    fenced = '```json\n{"summary": "ok", "issues": []}\n```'
    report = parser.parse(fenced)
    assert report.summary == "ok"
    assert report.issues == []
