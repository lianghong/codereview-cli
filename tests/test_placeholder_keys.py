"""Placeholder API-key rejection: shared helper + per-provider contract.

CLAUDE.md contract: the placeholder set must include the exact strings the
README tells users to export — matched case-insensitively after .strip() —
so a copied-and-not-replaced placeholder fails fast at --validate instead of
401'ing on the first real call.

These tests lock that contract for every provider that validates an API key,
via the shared helper in mixins.py (single source of truth for the generic
placeholders) plus each provider's README-documented string.
"""

import os
from unittest.mock import patch

import pytest

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockOpenAIConfig,
    DeepSeekConfig,
    GoogleGenAIConfig,
    ModelConfig,
    MoonshotConfig,
    NVIDIAConfig,
    PricingConfig,
    ZAIConfig,
)
from codereview.providers.mixins import is_placeholder_api_key

# ---------------------------------------------------------------------------
# Shared helper unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "placeholder",
        "your-api-key-here",
        "your-api-key",
        "  PLACEHOLDER  ",  # strip + case-insensitive
        "Your-API-Key-Here",
    ],
)
def test_generic_placeholders_rejected(key):
    assert is_placeholder_api_key(key)


@pytest.mark.parametrize(
    "key,extra",
    [
        ("your-deepseek-key", ("your-deepseek-key", "your-deepseek-api-key-here")),
        ("  Your-Moonshot-Key ", ("your-moonshot-key",)),
    ],
)
def test_provider_specific_placeholders_rejected(key, extra):
    assert is_placeholder_api_key(key, extra)


def test_real_key_accepted():
    assert not is_placeholder_api_key("sk-abc123def456ghi789jkl012")
    assert not is_placeholder_api_key(
        "nvapi-x9y8z7w6v5u4t3s2r1q0", ("nvapi-your-key-here",)
    )


# ---------------------------------------------------------------------------
# Per-provider contract: the EXACT string the README documents must hard-fail
# --validate. README export lines:
#   AZURE_OPENAI_API_KEY="your-api-key"        (README.md:132)
#   NVIDIA_API_KEY="nvapi-your-key-here"       (README.md:182)
#   GOOGLE_API_KEY="your-api-key-here"         (README.md:240)
#   DEEPSEEK_API_KEY="your-deepseek-key"       (README.md:265)
#   ZAI_API_KEY="your-zai-key"                 (README.md:289)
#   KIMI_API_KEY="your-moonshot-key"           (README.md:315)
# ---------------------------------------------------------------------------


def _model_config(**overrides):
    defaults = dict(
        id="test",
        full_id="vendor/test-model",
        name="Test Model",
        pricing=PricingConfig(input_per_million=1.0, output_per_million=5.0),
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


@pytest.fixture(autouse=True)
def _skip_connection_tests():
    os.environ["CODEREVIEW_SKIP_CONNECTION_TEST"] = "1"
    yield
    os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)


def _assert_placeholder_fails(provider):
    result = provider.validate_credentials()
    assert result.valid is False, (
        "README-documented placeholder key must hard-fail --validate "
        f"(provider={result.provider})"
    )


@pytest.mark.parametrize("key", ["your-zai-key", "  Your-ZAI-Key  "])
def test_zai_rejects_readme_placeholder(key):
    from codereview.providers.zai import ZAIProvider

    config = ZAIConfig(api_key=key)
    with patch("codereview.providers.zai.ChatOpenAI"):
        _assert_placeholder_fails(ZAIProvider(_model_config(), config))


@pytest.mark.parametrize("key", ["your-api-key", "  Your-API-Key  "])
def test_azure_rejects_readme_placeholder(key):
    from codereview.providers.azure_openai import AzureOpenAIProvider

    config = AzureOpenAIConfig(
        endpoint="https://test.openai.azure.com",
        api_key=key,
        api_version="2024-01-01",
    )
    with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
        _assert_placeholder_fails(
            AzureOpenAIProvider(_model_config(deployment_name="test-deploy"), config)
        )


@pytest.mark.parametrize("key", ["nvapi-your-key-here", "  NVAPI-Your-Key-Here "])
def test_nvidia_rejects_readme_placeholder(key):
    from codereview.providers.nvidia import NVIDIAProvider

    config = NVIDIAConfig(api_key=key)
    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        _assert_placeholder_fails(NVIDIAProvider(_model_config(), config))


@pytest.mark.parametrize("key", ["your-api-key-here", " Your-API-Key-Here "])
def test_google_rejects_readme_placeholder(key):
    from codereview.providers.google_genai import GoogleGenAIProvider

    config = GoogleGenAIConfig(api_key=key)
    with patch("codereview.providers.google_genai.ChatGoogleGenerativeAI"):
        _assert_placeholder_fails(GoogleGenAIProvider(_model_config(), config))


@pytest.mark.parametrize("key", ["your-deepseek-key", " Your-DeepSeek-Key "])
def test_deepseek_rejects_readme_placeholder(key):
    from codereview.providers.deepseek import DeepSeekProvider

    config = DeepSeekConfig(api_key=key)
    with patch("codereview.providers.deepseek.ChatDeepSeek"):
        _assert_placeholder_fails(DeepSeekProvider(_model_config(), config))


@pytest.mark.parametrize("key", ["your-moonshot-key", " Your-Moonshot-Key "])
def test_moonshot_rejects_readme_placeholder(key):
    from codereview.providers.moonshot import MoonshotProvider

    config = MoonshotConfig(api_key=key)
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        _assert_placeholder_fails(MoonshotProvider(_model_config(), config))


@pytest.mark.parametrize(
    "key", ["your-bedrock-api-key-here", " Your-Bedrock-API-Key-Here "]
)
def test_bedrock_openai_rejects_placeholder(key):
    from codereview.providers.bedrock_openai import BedrockOpenAIProvider

    config = BedrockOpenAIConfig(
        api_key=key,
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
    )
    with patch("codereview.providers.bedrock_openai.ChatOpenAI"):
        _assert_placeholder_fails(BedrockOpenAIProvider(_model_config(), config))
