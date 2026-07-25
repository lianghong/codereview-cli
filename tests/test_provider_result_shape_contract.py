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


# ---------------------------------------------------------------------------
# Contract: a cleartext endpoint must fail closed at client construction
#
# require_https (providers/mixins.py) is called from _create_model, which runs
# in __init__ — deliberately stronger than checking inside
# validate_credentials, because a caller that skips validation still cannot
# send an API key / bearer token to an http:// endpoint (CWE-319).
#
# Parametrized over every provider whose config carries a URL so a NEW provider
# that forgets the call fails here, rather than relying on someone remembering
# to add a per-provider test. Bedrock's SigV4 path takes no URL and is absent.
# ---------------------------------------------------------------------------


def _cleartext_azure(model_config):
    from codereview.providers.azure_openai import AzureOpenAIProvider

    # endpoint is a Pydantic HttpUrl, which accepts http:// — require_https is
    # the only thing standing between AZURE_OPENAI_API_KEY and cleartext.
    cfg = AzureOpenAIConfig(
        endpoint="http://insecure.openai.azure.com",
        api_key="test-key-12345678901234567890",
        api_version="2024-01-01",
    )
    mc = model_config.model_copy(update={"deployment_name": "contract-deployment"})
    return (
        "codereview.providers.azure_openai.AzureChatOpenAI",
        lambda: AzureOpenAIProvider(mc, cfg),
    )


def _cleartext_nvidia(model_config):
    from codereview.providers.nvidia import NVIDIAProvider

    cfg = NVIDIAConfig(
        api_key="nvapi-test-1234567890abcdef",
        base_url="http://insecure-nim.example.com/v1",
    )
    return (
        "codereview.providers.nvidia.ChatNVIDIA",
        lambda: NVIDIAProvider(model_config, cfg),
    )


def _cleartext_moonshot(model_config):
    from codereview.providers.moonshot import MoonshotProvider

    cfg = MoonshotConfig(
        api_key="test-key-1234567890abcdef",
        base_url="http://insecure.moonshot.cn/v1",
    )
    return (
        "codereview.providers.moonshot.ChatMoonshot",
        lambda: MoonshotProvider(model_config, cfg),
    )


def _cleartext_zai(model_config):
    from codereview.providers.zai import ZAIProvider

    cfg = ZAIConfig(
        api_key="test-key-1234567890abcdef",
        base_url="http://insecure.z.ai/api/paas/v4/",
    )
    return (
        "codereview.providers.zai.ChatOpenAI",
        lambda: ZAIProvider(model_config, cfg),
    )


def _cleartext_bedrock_openai(model_config):
    from codereview.providers.bedrock_openai import BedrockOpenAIProvider

    cfg = BedrockOpenAIConfig(
        api_key="test-key-1234567890abcdef",
        base_url="http://insecure.example.com/openai/v1",
    )
    return (
        "codereview.providers.bedrock_openai.ChatOpenAI",
        lambda: BedrockOpenAIProvider(model_config, cfg),
    )


def _cleartext_deepseek(model_config):
    from codereview.config.models import DeepSeekConfig
    from codereview.providers.deepseek import DeepSeekProvider

    cfg = DeepSeekConfig(
        api_key="test-key-1234567890abcdef",
        api_base="http://insecure.deepseek.com",
    )
    return (
        "codereview.providers.deepseek.ChatDeepSeek",
        lambda: DeepSeekProvider(model_config, cfg),
    )


# Every provider whose config carries an endpoint/base URL. Keep in sync with
# the require_https call sites in codereview/providers/*.py.
_CLEARTEXT_BUILDERS = {
    "azure_openai": _cleartext_azure,
    "bedrock_openai": _cleartext_bedrock_openai,
    "deepseek": _cleartext_deepseek,
    "moonshot": _cleartext_moonshot,
    "nvidia": _cleartext_nvidia,
    "zai": _cleartext_zai,
}


@pytest.mark.parametrize("provider_key", sorted(_CLEARTEXT_BUILDERS))
def test_cleartext_endpoint_fails_closed_at_construction(provider_key):
    """An http:// endpoint raises before any client is built or call is made."""
    model_config = _model_config(supports_tool_use=True)
    patch_target, build = _CLEARTEXT_BUILDERS[provider_key](model_config)

    with patch(patch_target) as mock_client:
        with pytest.raises(ValueError, match="must use HTTPS"):
            build()

        # Fail *closed*: the credential never reached a client instance.
        mock_client.assert_not_called()


def test_every_url_taking_provider_is_covered_by_the_cleartext_contract():
    """No provider may call require_https without appearing above.

    Guards the registry itself: a new provider that wires a base_url gets a
    cleartext test by construction, instead of depending on someone noticing.
    """
    import re
    from pathlib import Path

    providers_dir = Path(__file__).resolve().parent.parent / "codereview" / "providers"
    callers = {
        path.stem
        for path in providers_dir.glob("*.py")
        if path.stem not in {"__init__", "base", "mixins", "factory"}
        and re.search(r"^\s+.*require_https\(", path.read_text(), re.MULTILINE)
    }
    assert callers, "no require_https call sites found; the scan is broken"

    missing = sorted(callers - set(_CLEARTEXT_BUILDERS))
    assert not missing, (
        f"provider(s) {missing} call require_https but have no cleartext test. "
        "Add a _cleartext_<provider> builder to _CLEARTEXT_BUILDERS."
    )


# ---------------------------------------------------------------------------
# Contract: max_retries=None means "the provider decides"
#
# CodeAnalyzer.analyze_batch used to default max_retries to a hardcoded 3 and
# forward it unconditionally, so every provider's own default was dead code and
# NVIDIAConfig.max_retries was unreachable config — a Bedrock-tuned retry count
# was silently applied to NVIDIA NIM's gateway 504s and Azure's quota windows.
#
# These tests pin the resolution contract end to end: None reaches each provider
# and resolves to *that provider's* number, while an explicit value still wins.
# Parametrized over every provider so a new one that hardcodes a default in its
# signature (rather than resolving through _resolve_max_retries) fails here.
# ---------------------------------------------------------------------------


def _build_google(model_config):
    from codereview.config.models import GoogleGenAIConfig
    from codereview.providers.google_genai import GoogleGenAIProvider

    cfg = GoogleGenAIConfig(api_key="test-google-api-key-12345")
    return (
        "codereview.providers.google_genai.ChatGoogleGenerativeAI",
        lambda: GoogleGenAIProvider(model_config, cfg),
    )


def _build_deepseek(model_config):
    from codereview.config.models import DeepSeekConfig
    from codereview.providers.deepseek import DeepSeekProvider

    cfg = DeepSeekConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.deepseek.ChatDeepSeek",
        lambda: DeepSeekProvider(model_config, cfg),
    )


# provider key -> (builder, retries expected when the caller passes None).
# Bedrock's 3 is deliberate (throttling clears in a couple of attempts); the
# rest use 5. NVIDIA's comes from NVIDIAConfig.max_retries, which the analyzer's
# old hardcoded 3 made unreachable.
_RETRY_DEFAULTS = {
    "azure_openai": (_build_azure, 5),
    "bedrock": (_build_bedrock, 3),
    "bedrock_openai": (_build_bedrock_openai, 5),
    "deepseek": (_build_deepseek, 5),
    "google_genai": (_build_google, 5),
    "moonshot": (_build_moonshot, 5),
    "nvidia": (_build_nvidia, 5),
    "zai": (_build_zai, 5),
}


def _capture_retry_config(provider_key, max_retries):
    """Run analyze_batch and return the RetryConfig the provider built."""
    model_config = _model_config(supports_tool_use=True)
    patch_target, build = _RETRY_DEFAULTS[provider_key][0](model_config)

    with patch(patch_target) as mock_client:
        instance = MagicMock()
        instance.with_structured_output.return_value = MagicMock()
        mock_client.return_value = instance

        provider = build()

        captured = {}

        def fake_execute(chain_input, retry_config, batch_context):
            captured["retry_config"] = retry_config
            from codereview.models import CodeReviewReport

            return CodeReviewReport(summary="ok", issues=[])

        with patch.object(provider, "_execute_with_retry", fake_execute):
            provider.analyze_batch(1, 1, {"a.py": "x=1"}, max_retries=max_retries)

    return captured["retry_config"]


@pytest.mark.parametrize("provider_key", sorted(_RETRY_DEFAULTS))
def test_none_max_retries_uses_the_providers_own_default(provider_key):
    """max_retries=None must resolve to the provider's own retry count."""
    expected = _RETRY_DEFAULTS[provider_key][1]
    retry_config = _capture_retry_config(provider_key, None)

    assert retry_config.max_retries == expected, (
        f"{provider_key}: max_retries=None resolved to "
        f"{retry_config.max_retries}, expected its own default of {expected}. "
        "Resolve None through _resolve_max_retries instead of hardcoding a "
        "default in the signature."
    )


@pytest.mark.parametrize("provider_key", sorted(_RETRY_DEFAULTS))
def test_explicit_max_retries_overrides_the_providers_default(provider_key):
    """An explicit caller value still wins over the provider's default."""
    retry_config = _capture_retry_config(provider_key, 1)

    assert retry_config.max_retries == 1, (
        f"{provider_key}: explicit max_retries=1 was not honoured"
    )


def test_nvidia_config_max_retries_is_live_config():
    """NVIDIAConfig.max_retries must reach RetryConfig, not just sit in config.

    It was dead for as long as CodeAnalyzer forwarded a hardcoded 3. A
    non-default value proves the config field is actually read.
    """
    model_config = _model_config(supports_tool_use=True)

    from codereview.providers.nvidia import NVIDIAProvider

    cfg = NVIDIAConfig(api_key="nvapi-test-1234567890abcdef", max_retries=9)

    with patch("codereview.providers.nvidia.ChatNVIDIA") as mock_client:
        instance = MagicMock()
        instance.with_structured_output.return_value = MagicMock()
        mock_client.return_value = instance

        provider = NVIDIAProvider(model_config, cfg)
        captured = {}

        def fake_execute(chain_input, retry_config, batch_context):
            captured["retry_config"] = retry_config
            from codereview.models import CodeReviewReport

            return CodeReviewReport(summary="ok", issues=[])

        with patch.object(provider, "_execute_with_retry", fake_execute):
            provider.analyze_batch(1, 1, {"a.py": "x=1"})

    assert captured["retry_config"].max_retries == 9


def test_analyzer_defers_the_retry_decision_to_the_provider():
    """CodeAnalyzer must forward None, not a retry count of its own.

    The regression this guards: analyzer.py hardcoded `max_retries: int = 3`
    and always passed it, so every provider default and NVIDIAConfig.max_retries
    were unreachable.
    """
    import inspect

    from codereview.analyzer import CodeAnalyzer

    default = (
        inspect.signature(CodeAnalyzer.analyze_batch).parameters["max_retries"].default
    )
    assert default is None, (
        "CodeAnalyzer.analyze_batch must default max_retries to None so the "
        f"provider resolves it; got {default!r}."
    )


def test_every_provider_analyze_batch_defaults_max_retries_to_none():
    """No provider may hardcode a max_retries default in its signature.

    A concrete default there overrides the base contract for every caller that
    doesn't pass one — which is how the analyzer's hardcoded 3 went unnoticed.
    """
    import importlib
    import inspect
    from pathlib import Path

    from codereview.providers.base import ModelProvider

    providers_dir = Path(__file__).resolve().parent.parent / "codereview" / "providers"
    offenders = []
    checked = []
    for path in sorted(providers_dir.glob("*.py")):
        if path.stem in {"__init__", "base", "mixins", "factory"}:
            continue
        module = importlib.import_module(f"codereview.providers.{path.stem}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                not issubclass(obj, ModelProvider)
                or obj is ModelProvider
                or obj.__module__ != module.__name__
            ):
                continue
            default = (
                inspect.signature(obj.analyze_batch).parameters["max_retries"].default
            )
            checked.append(obj.__name__)
            if default is not None:
                offenders.append(f"{obj.__name__}={default!r}")

    assert checked, "no provider classes found; the scan is broken"
    assert not offenders, (
        f"provider(s) hardcode a max_retries default: {offenders}. Default to "
        "None and resolve via _resolve_max_retries so the provider's own "
        "number applies without overriding an explicit caller value."
    )


# ---------------------------------------------------------------------------
# Contract: validate_credentials agrees with the constructor
#
# require_https (enforced in _create_model, from __init__) and each provider's
# validate_credentials HTTPS check are two spellings of the same question, and
# they had drifted: require_https lowercases the URL, the providers used a bare
# startswith("https://"). So "HTTPS://host" built a working provider that
# --validate then hard-failed — a pre-flight check calling a good config broken.
# Both now route through is_https_url.
#
# The presence check had the mirror-image bug: `not api_key` is False for
# "   ", which passes Pydantic's min_length=1, so a whitespace-only key
# reported every check as PASSING and deferred the failure to a 401.
# ---------------------------------------------------------------------------


def _validating_provider(provider_key, *, api_key, url=None):
    """Construct a provider for --validate, optionally overriding its URL.

    Reuses the cleartext registry's model config; the client class is patched so
    nothing reaches a network. Returns the constructed provider.
    """
    from codereview.config.models import (
        AzureOpenAIConfig,
        BedrockOpenAIConfig,
        DeepSeekConfig,
        MoonshotConfig,
        ZAIConfig,
    )

    model_config = _model_config(supports_tool_use=True)

    if provider_key == "zai":
        from codereview.providers.zai import ZAIProvider

        cfg = ZAIConfig(api_key=api_key, **({"base_url": url} if url else {}))
        target, build = (
            "codereview.providers.zai.ChatOpenAI",
            lambda: ZAIProvider(model_config, cfg),
        )
    elif provider_key == "moonshot":
        from codereview.providers.moonshot import MoonshotProvider

        cfg = MoonshotConfig(api_key=api_key, **({"base_url": url} if url else {}))
        target, build = (
            "codereview.providers.moonshot.ChatMoonshot",
            lambda: MoonshotProvider(model_config, cfg),
        )
    elif provider_key == "deepseek":
        from codereview.providers.deepseek import DeepSeekProvider

        cfg = DeepSeekConfig(api_key=api_key, **({"api_base": url} if url else {}))
        target, build = (
            "codereview.providers.deepseek.ChatDeepSeek",
            lambda: DeepSeekProvider(model_config, cfg),
        )
    elif provider_key == "bedrock_openai":
        from codereview.providers.bedrock_openai import BedrockOpenAIProvider

        cfg = BedrockOpenAIConfig(
            api_key=api_key,
            base_url=url or "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        )
        target, build = (
            "codereview.providers.bedrock_openai.ChatOpenAI",
            lambda: BedrockOpenAIProvider(model_config, cfg),
        )
    elif provider_key == "azure_openai":
        from codereview.providers.azure_openai import AzureOpenAIProvider

        cfg = AzureOpenAIConfig(
            endpoint=url or "https://test.openai.azure.com",
            api_key=api_key,
            api_version="2024-01-01",
        )
        mc = model_config.model_copy(update={"deployment_name": "contract-deployment"})
        target, build = (
            "codereview.providers.azure_openai.AzureChatOpenAI",
            lambda: AzureOpenAIProvider(mc, cfg),
        )
    else:  # pragma: no cover — parametrization keeps this unreachable
        raise AssertionError(f"unknown provider key {provider_key!r}")

    with patch(target):
        return build()


# Providers whose config carries a URL that validate_credentials checks for
# HTTPS, with the uppercase-scheme spelling of their default endpoint. Azure is
# absent: its endpoint is a Pydantic HttpUrl, which normalizes the scheme to
# lowercase before validate_credentials ever sees it.
_UPPERCASE_SCHEME_URLS = {
    "zai": "HTTPS://api.z.ai/api/paas/v4/",
    "moonshot": "HTTPS://api.moonshot.cn/v1",
    "deepseek": "HTTPS://api.deepseek.com",
    "bedrock_openai": "HTTPS://bedrock-mantle.us-east-1.api.aws/openai/v1",
}

_GOOD_KEY = "test-key-1234567890abcdef"


@pytest.mark.parametrize("provider_key", sorted(_UPPERCASE_SCHEME_URLS))
def test_validate_accepts_any_url_the_constructor_accepted(provider_key, monkeypatch):
    """An uppercase HTTPS scheme is valid; --validate must not call it broken.

    URL schemes are case-insensitive (RFC 3986 §3.1) and ``require_https``
    lowercases before testing, so the provider constructs fine and real calls
    succeed. A ``startswith("https://")`` in validate_credentials rejected it,
    making --validate report a hard failure for a working configuration.
    """
    monkeypatch.setenv("CODEREVIEW_SKIP_CONNECTION_TEST", "1")

    provider = _validating_provider(
        provider_key, api_key=_GOOD_KEY, url=_UPPERCASE_SCHEME_URLS[provider_key]
    )
    result = provider.validate_credentials()

    assert result.valid is True, (
        f"{provider_key}: --validate rejected an uppercase-scheme HTTPS URL that "
        f"_create_model accepted. Checks: {result.checks}"
    )


@pytest.mark.parametrize(
    "provider_key",
    ["azure_openai", "bedrock_openai", "deepseek", "moonshot", "zai"],
)
@pytest.mark.parametrize("blank", ["   ", "\t", "\n "])
def test_whitespace_only_api_key_is_not_a_credential(provider_key, blank, monkeypatch):
    """A whitespace-only key must hard-fail, not report every check passing.

    Pydantic's ``min_length=1`` accepts "   " and the loader's truthiness gate
    registers the provider, so before ``is_blank`` this reached
    validate_credentials and came back valid=True — the 401 landing on the
    first real API call instead.
    """
    monkeypatch.setenv("CODEREVIEW_SKIP_CONNECTION_TEST", "1")

    provider = _validating_provider(provider_key, api_key=blank)
    result = provider.validate_credentials()

    assert result.valid is False, (
        f"{provider_key}: whitespace-only API key passed --validate. "
        f"Checks: {result.checks}"
    )
    assert any("API Key" in name and not passed for name, passed, _ in result.checks), (
        f"{provider_key}: the failure must be attributed to the API Key check"
    )


def test_every_url_checking_provider_uses_the_shared_https_predicate():
    """No provider may re-implement the HTTPS test with a bare startswith.

    The drift this guards against is exactly what happened: two spellings of
    "is this HTTPS", one case-insensitive and one not, so the constructor and
    --validate disagreed about the same URL.
    """
    import re
    from pathlib import Path

    providers_dir = Path(__file__).resolve().parent.parent / "codereview" / "providers"
    offenders = []
    for path in sorted(providers_dir.glob("*.py")):
        # mixins.py is the one permitted site: it *defines* is_https_url.
        if path.stem in {"__init__", "base", "factory", "mixins"}:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'startswith\(\s*["\']https?://', line):
                offenders.append(f"{path.name}:{lineno}")

    assert not offenders, (
        f"hardcoded scheme test(s) at {offenders}. Use is_https_url from "
        "mixins.py so validate_credentials and require_https can't disagree."
    )
