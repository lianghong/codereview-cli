"""Tests for BedrockProvider."""

from unittest.mock import Mock, patch

import pytest

from codereview.config.models import (
    BedrockConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
)
from codereview.models import CodeReviewReport
from codereview.providers.bedrock import BedrockProvider


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        id="test-opus",
        name="Test Opus",
        aliases=["opus-test"],
        full_id="test.opus.v1",
        pricing=PricingConfig(
            input_per_million=5.0,
            output_per_million=25.0,
        ),
        inference_params=InferenceParams(
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            max_output_tokens=8192,
        ),
    )


@pytest.fixture
def provider_config():
    """Create test provider configuration."""
    return BedrockConfig(region="us-west-2")


@pytest.fixture
def mock_report():
    """Create mock CodeReviewReport."""
    return CodeReviewReport(
        summary="Test analysis",
        metrics={"files": 1},
        issues=[],
        system_design_insights="No concerns",
        recommendations=[],
        improvement_suggestions=[],
    )


def test_bedrock_provider_initialization(model_config, provider_config):
    """Test BedrockProvider can be instantiated."""
    with patch("codereview.providers.bedrock.ChatBedrockConverse"):
        provider = BedrockProvider(model_config, provider_config)
        assert provider is not None
        assert provider.temperature == 0.1


def test_bedrock_provider_custom_temperature(model_config, provider_config):
    """Test temperature override."""
    with patch("codereview.providers.bedrock.ChatBedrockConverse"):
        provider = BedrockProvider(model_config, provider_config, temperature=0.5)
        assert provider.temperature == 0.5


def test_analyze_batch(model_config, provider_config, mock_report):
    """Test analyze_batch returns CodeReviewReport."""
    with patch("codereview.providers.bedrock.ChatBedrockConverse") as mock_bedrock:
        # Setup mock
        mock_instance = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = mock_report
        mock_instance.with_structured_output.return_value = mock_structured
        mock_bedrock.return_value = mock_instance

        provider = BedrockProvider(model_config, provider_config)
        result = provider.analyze_batch(
            batch_number=1,
            total_batches=1,
            files_content={"test.py": "print('hello')"},
        )

        assert isinstance(result, CodeReviewReport)
        assert result.summary == "Test analysis"


def test_token_tracking(model_config, provider_config, mock_report):
    """Test token tracking."""
    with patch("codereview.providers.bedrock.ChatBedrockConverse") as mock_bedrock:
        # Setup mock with usage metadata
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "usage": {"input_tokens": 100, "output_tokens": 50}
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
        mock_bedrock.return_value = mock_instance

        provider = BedrockProvider(model_config, provider_config)

        # Initial state
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0

        # After analyze
        provider.analyze_batch(1, 1, {"test.py": "code"})
        assert provider.total_input_tokens == 100
        assert provider.total_output_tokens == 50


def test_cost_estimation(model_config, provider_config, mock_report):
    """Test cost estimation."""
    with patch("codereview.providers.bedrock.ChatBedrockConverse") as mock_bedrock:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "usage": {"input_tokens": 100000, "output_tokens": 50000}
        }

        mock_instance = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = mock_report_with_metadata
        mock_instance.with_structured_output.return_value = mock_structured
        mock_bedrock.return_value = mock_instance

        provider = BedrockProvider(model_config, provider_config)
        provider.analyze_batch(1, 1, {"test.py": "code"})

        cost = provider.estimate_cost()

        # 100000 tokens * $5/million = $0.50
        # 50000 tokens * $25/million = $1.25
        assert cost["input_tokens"] == 100000
        assert cost["output_tokens"] == 50000
        assert cost["input_cost"] == 0.5
        assert cost["output_cost"] == 1.25
        assert cost["total_cost"] == 1.75


def test_reset_state(model_config, provider_config, mock_report):
    """Test reset_state clears token counters."""
    with patch("codereview.providers.bedrock.ChatBedrockConverse") as mock_bedrock:
        mock_report_with_metadata = Mock(spec=CodeReviewReport)
        mock_report_with_metadata.response_metadata = {
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }

        mock_instance = Mock()
        mock_structured = Mock()
        mock_structured.invoke.return_value = mock_report_with_metadata
        mock_instance.with_structured_output.return_value = mock_structured
        mock_bedrock.return_value = mock_instance

        provider = BedrockProvider(model_config, provider_config)
        provider.analyze_batch(1, 1, {"test.py": "code"})

        assert provider.total_input_tokens > 0

        provider.reset_state()
        assert provider.total_input_tokens == 0
        assert provider.total_output_tokens == 0


def test_get_model_display_name(model_config, provider_config):
    """Test get_model_display_name returns model name."""
    with patch("codereview.providers.bedrock.ChatBedrockConverse"):
        provider = BedrockProvider(model_config, provider_config)
        assert provider.get_model_display_name() == "Test Opus"


def test_retry_logic_on_throttling(model_config, provider_config, mock_report):
    """Test exponential backoff retry logic for throttling errors."""
    with patch("codereview.providers.bedrock.ChatBedrockConverse") as mock_bedrock:
        with patch("time.sleep") as mock_sleep:
            # Setup mock to fail twice with throttling, then succeed
            mock_instance = Mock()
            mock_structured = Mock()

            from botocore.exceptions import ClientError

            throttle_error = ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
                "InvokeModel",
            )

            mock_structured.invoke.side_effect = [
                throttle_error,
                throttle_error,
                mock_report,
            ]
            mock_instance.with_structured_output.return_value = mock_structured
            mock_bedrock.return_value = mock_instance

            provider = BedrockProvider(model_config, provider_config)
            result = provider.analyze_batch(1, 1, {"test.py": "code"})

            # Should succeed after retries
            assert isinstance(result, CodeReviewReport)

            # Verify exponential backoff: 2^0=1s, 2^1=2s
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)
            mock_sleep.assert_any_call(2)
