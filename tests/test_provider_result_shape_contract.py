"""Contract tests: every two-mode provider produces a documented result shape.

``_execute_with_retry`` (providers/base.py) accepts exactly two result shapes
from a provider's chain:

1. an ``include_raw=True`` dict ``{"raw": ..., "parsed": CodeReviewReport}`` —
   produced by ``with_structured_output(CodeReviewReport, include_raw=True)`` on
   the tool-calling path, and
2. a bare ``CodeReviewReport`` — produced by appending a ``PydanticOutputParser``
   to the chain on the prompt-parsing path.

Providers that honor ``supports_tool_use`` must produce shape (1) when it's
True and shape (2) when it's False. These tests assert that structural contract
for each such provider in BOTH modes, so a provider whose chain wiring drifts
(e.g. forgets ``include_raw=True``, or doesn't append the parser) is caught —
without depending on a live model call.
"""

from unittest.mock import MagicMock, patch

import pytest

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    BedrockOpenAIConfig,
    ModelConfig,
    MoonshotConfig,
    NVIDIAConfig,
    PricingConfig,
    ZAIConfig,
)


def _model_config(supports_tool_use: bool) -> ModelConfig:
    return ModelConfig(
        id="contract-model",
        full_id="contract-model",
        name="Contract Model",
        aliases=[],
        pricing=PricingConfig(input_per_million=1.0, output_per_million=2.0),
        supports_tool_use=supports_tool_use,
    )


def _build_bedrock(model_config):
    from codereview.providers.bedrock import BedrockProvider

    return (
        "codereview.providers.bedrock.ChatBedrockConverse",
        lambda: BedrockProvider(model_config, BedrockConfig(region="us-west-2")),
    )


def _build_azure(model_config):
    from codereview.providers.azure_openai import AzureOpenAIProvider

    cfg = AzureOpenAIConfig(
        endpoint="https://test.openai.azure.com",
        api_key="test-key-12345678901234567890",
        api_version="2024-01-01",
    )
    # Azure routes the wire model via deployment_name.
    mc = model_config.model_copy(update={"deployment_name": "contract-deployment"})
    return (
        "codereview.providers.azure_openai.AzureChatOpenAI",
        lambda: AzureOpenAIProvider(mc, cfg),
    )


def _build_nvidia(model_config):
    from codereview.providers.nvidia import NVIDIAProvider

    cfg = NVIDIAConfig(api_key="nvapi-test-1234567890abcdef")
    return (
        "codereview.providers.nvidia.ChatNVIDIA",
        lambda: NVIDIAProvider(model_config, cfg),
    )


def _build_moonshot(model_config):
    from codereview.providers.moonshot import MoonshotProvider

    cfg = MoonshotConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.moonshot.ChatMoonshot",
        lambda: MoonshotProvider(model_config, cfg),
    )


def _build_zai(model_config):
    from codereview.providers.zai import ZAIProvider

    cfg = ZAIConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.zai.ChatOpenAI",
        lambda: ZAIProvider(model_config, cfg),
    )


def _build_bedrock_openai(model_config):
    from codereview.providers.bedrock_openai import BedrockOpenAIProvider

    cfg = BedrockOpenAIConfig(
        api_key="test-key-1234567890abcdef",
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
    )
    return (
        "codereview.providers.bedrock_openai.ChatOpenAI",
        lambda: BedrockOpenAIProvider(model_config, cfg),
    )


# Every provider that branches on supports_tool_use. NVIDIA's client patch
# target plus the five others.
_TWO_MODE_BUILDERS = {
    "bedrock": _build_bedrock,
    "azure_openai": _build_azure,
    "nvidia": _build_nvidia,
    "moonshot": _build_moonshot,
    "zai": _build_zai,
    "bedrock_openai": _build_bedrock_openai,
}


@pytest.mark.parametrize("provider_key", sorted(_TWO_MODE_BUILDERS))
def test_tool_use_mode_requests_include_raw_structured_output(provider_key):
    """supports_tool_use=True → chain uses with_structured_output(include_raw=True).

    That call is what yields the documented dict shape
    {"raw": ..., "parsed": CodeReviewReport} consumed by _execute_with_retry.
    """
    from codereview.models import CodeReviewReport

    model_config = _model_config(supports_tool_use=True)
    patch_target, build = _TWO_MODE_BUILDERS[provider_key](model_config)

    with patch(patch_target) as mock_client:
        instance = MagicMock()
        instance.with_structured_output.return_value = MagicMock()
        mock_client.return_value = instance

        provider = build()

        assert provider._use_prompt_parsing is False, (
            f"{provider_key}: expected tool-use mode"
        )
        instance.with_structured_output.assert_called_once_with(
            CodeReviewReport, include_raw=True
        )


@pytest.mark.parametrize("provider_key", sorted(_TWO_MODE_BUILDERS))
def test_prompt_parse_mode_appends_pydantic_parser(provider_key):
    """supports_tool_use=False → chain ends with the PydanticOutputParser.

    That parser is what yields the documented bare-CodeReviewReport shape.
    """
    model_config = _model_config(supports_tool_use=False)
    patch_target, build = _TWO_MODE_BUILDERS[provider_key](model_config)

    with patch(patch_target) as mock_client:
        instance = MagicMock()
        mock_client.return_value = instance

        provider = build()

        assert provider._use_prompt_parsing is True, (
            f"{provider_key}: expected prompt-parsing mode"
        )
        # No tool-calling structured output on this path.
        instance.with_structured_output.assert_not_called()
        # The chain's final runnable is the provider's PydanticOutputParser, so
        # a model text response is coerced into a CodeReviewReport.
        assert provider.chain.last is provider._output_parser
