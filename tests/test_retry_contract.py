"""Contract tests: every provider classifies transient failures consistently.

``_is_retryable_error`` is the single decision that separates "wait and try
again" from "throw this batch away", and it is invisible when wrong: a
misclassified throttle looks exactly like a lost batch. Two provider
classifiers were dead code against their *installed* clients before these tests
existed, both for the same reason — they tested ``isinstance`` against an
exception type the client never raises:

- **NVIDIA** matched ``httpx.HTTPStatusError``, but
  ``langchain-nvidia-ai-endpoints`` runs on ``requests`` and its
  ``_NVIDIASyncClient._try_raise`` *discards* the typed error, re-raising
  ``Exception("[504] Gateway Timeout\\n…")`` (its own source carries a
  ``# todo: raise as an HTTPError``). Every NIM gateway 504 — the exact failure
  NVIDIA raises ``max_retries`` for — aborted on attempt 1.
- **Google** matched ``google.api_core.exceptions.ResourceExhausted`` /
  ``ServiceUnavailable``, but ``langchain-google-genai`` 4.x raises
  ``google.genai.errors.ClientError`` / ``ServerError``. ``api_core`` is still
  installed transitively, so the import succeeded and nothing looked wrong.

The lesson these tests encode: **build the error the way the real client builds
it.** A hand-constructed exception of a type the SDK no longer raises passes a
test and fails in production, which is precisely what happened — the previous
Google tests constructed ``ResourceExhausted`` directly and passed for as long
as the classifier was dead.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests

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
from codereview.providers.base import RetryConfig


def _model_config() -> ModelConfig:
    return ModelConfig(
        id="retry-contract-model",
        full_id="retry-contract-model",
        name="Retry Contract Model",
        aliases=[],
        pricing=PricingConfig(input_per_million=1.0, output_per_million=2.0),
    )


# ---------------------------------------------------------------------------
# Provider construction (client patched; nothing reaches a network)
# ---------------------------------------------------------------------------


def _build(provider_key):
    """Return a constructed provider with its LangChain client patched out."""
    model_config = _model_config()

    if provider_key == "nvidia":
        from codereview.providers.nvidia import NVIDIAProvider

        target = "codereview.providers.nvidia.ChatNVIDIA"
        make = lambda: NVIDIAProvider(  # noqa: E731
            model_config, NVIDIAConfig(api_key="nvapi-test-1234567890abcdef")
        )
    elif provider_key == "google_genai":
        from codereview.providers.google_genai import GoogleGenAIProvider

        target = "codereview.providers.google_genai.ChatGoogleGenerativeAI"
        make = lambda: GoogleGenAIProvider(  # noqa: E731
            model_config, GoogleGenAIConfig(api_key="test-google-api-key-12345")
        )
    elif provider_key == "azure_openai":
        from codereview.providers.azure_openai import AzureOpenAIProvider

        target = "codereview.providers.azure_openai.AzureChatOpenAI"
        cfg = AzureOpenAIConfig(
            endpoint="https://test.openai.azure.com",
            api_key="test-key-12345678901234567890",
            api_version="2024-01-01",
        )
        mc = model_config.model_copy(update={"deployment_name": "retry-deployment"})
        make = lambda: AzureOpenAIProvider(mc, cfg)  # noqa: E731
    elif provider_key == "deepseek":
        from codereview.providers.deepseek import DeepSeekProvider

        target = "codereview.providers.deepseek.ChatDeepSeek"
        make = lambda: DeepSeekProvider(  # noqa: E731
            model_config, DeepSeekConfig(api_key="test-key-1234567890abcdef")
        )
    elif provider_key == "moonshot":
        from codereview.providers.moonshot import MoonshotProvider

        target = "codereview.providers.moonshot.ChatMoonshot"
        make = lambda: MoonshotProvider(  # noqa: E731
            model_config, MoonshotConfig(api_key="test-key-1234567890abcdef")
        )
    elif provider_key == "zai":
        from codereview.providers.zai import ZAIProvider

        target = "codereview.providers.zai.ChatOpenAI"
        make = lambda: ZAIProvider(  # noqa: E731
            model_config, ZAIConfig(api_key="test-key-1234567890abcdef")
        )
    elif provider_key == "bedrock_openai":
        from codereview.providers.bedrock_openai import BedrockOpenAIProvider

        target = "codereview.providers.bedrock_openai.ChatOpenAI"
        cfg = BedrockOpenAIConfig(
            api_key="test-key-1234567890abcdef",
            base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        )
        make = lambda: BedrockOpenAIProvider(model_config, cfg)  # noqa: E731
    elif provider_key == "bedrock":
        from codereview.config.models import BedrockConfig
        from codereview.providers.bedrock import BedrockProvider

        target = "codereview.providers.bedrock.ChatBedrockConverse"
        make = lambda: BedrockProvider(  # noqa: E731
            model_config, BedrockConfig(region="us-west-2")
        )
    else:  # pragma: no cover — parametrization keeps this unreachable
        raise AssertionError(f"unknown provider key {provider_key!r}")

    with patch(target) as mock_client:
        instance = MagicMock()
        instance.with_structured_output.return_value = MagicMock()
        mock_client.return_value = instance
        return make()


# ---------------------------------------------------------------------------
# Error builders — each constructs the error THE WAY ITS REAL CLIENT DOES
# ---------------------------------------------------------------------------


def _nim_error(status: int):
    """Raise-and-capture a NIM error through the real client's own raiser.

    Routing through ``_NVIDIASyncClient._try_raise`` rather than hand-building
    the exception is the whole point: it is the code that converts the typed
    ``requests.HTTPError`` into a bare ``Exception`` whose only record of the
    status is the ``[504] …`` message prefix.
    """
    from langchain_nvidia_ai_endpoints._common import _NVIDIASyncClient

    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({"title": f"status {status}"}).encode()
    response.headers["Content-Type"] = "application/json"
    response.url = "https://integrate.api.nvidia.com/v1/chat/completions"

    client = _NVIDIASyncClient.__new__(_NVIDIASyncClient)
    try:
        _NVIDIASyncClient._try_raise(client, response)
    except Exception as exc:  # noqa: BLE001 — capturing is the point
        return exc
    raise AssertionError(f"NIM client did not raise for status {status}")


def _google_error(status: int):
    """Build the google-genai error type the SDK raises for ``status``."""
    import google.genai.errors as ge

    cls = ge.ClientError if 400 <= status < 500 else ge.ServerError
    return cls(
        status,
        {"error": {"code": status, "status": "TEST", "message": f"status {status}"}},
    )


def _openai_error(status: int):
    """Build the openai-client error every OpenAI-compatible provider sees."""
    from openai import APIStatusError, RateLimitError

    response = httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
        json={"error": {"message": f"status {status}"}},
    )
    cls = RateLimitError if status == 429 else APIStatusError
    return cls(f"status {status}", response=response, body=None)


def _bedrock_error(status: int):
    """Build the botocore ClientError Bedrock raises for ``status``."""
    from botocore.exceptions import ClientError

    codes = {
        429: "ThrottlingException",
        500: "InternalServerException",
        502: "InternalServerException",
        503: "ServiceUnavailableException",
        504: "ModelTimeoutException",
        400: "ValidationException",
        403: "AccessDeniedException",
        404: "ResourceNotFoundException",
    }
    return ClientError(
        {
            "Error": {"Code": codes[status], "Message": f"status {status}"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "Converse",
    )


# provider key -> (error builder, {status: expected_retryable})
#
# 429 (throttle) and 503 (unavailable) MUST be retryable everywhere — those are
# the failures that made retry logic necessary. 400/403/404 must NOT be: a bad
# request, key, or model won't heal, and retrying only makes the failure slower.
#
# 500/502/504 differ by provider on purpose, and the differences are recorded
# rather than normalized: NVIDIA treats a bare 500 as non-retryable (a NIM 500 is
# usually a malformed request the gateway rejected), while its 502/504 gateway
# errors are the common transient case. Bedrock accepts 5xx wholesale.
_RETRY_MATRIX = {
    "nvidia": (
        _nim_error,
        {
            429: True,
            502: True,
            503: True,
            504: True,
            500: False,
            400: False,
            404: False,
        },
    ),
    "google_genai": (
        _google_error,
        {
            429: True,
            500: True,
            503: True,
            504: True,
            400: False,
            403: False,
            404: False,
        },
    ),
    "azure_openai": (
        _openai_error,
        {429: True, 500: True, 502: True, 503: True, 504: True, 400: False, 404: False},
    ),
    "deepseek": (
        _openai_error,
        {429: True, 500: True, 503: True, 504: True, 400: False, 404: False},
    ),
    "moonshot": (
        _openai_error,
        {429: True, 500: True, 503: True, 504: True, 400: False, 404: False},
    ),
    "zai": (
        _openai_error,
        {429: True, 500: True, 503: True, 504: True, 400: False, 404: False},
    ),
    "bedrock_openai": (
        _openai_error,
        {429: True, 500: True, 503: True, 504: True, 400: False, 404: False},
    ),
    "bedrock": (
        _bedrock_error,
        {
            429: True,
            500: True,
            503: True,
            504: True,
            400: False,
            403: False,
            404: False,
        },
    ),
}


def _status_cases():
    for provider_key, (_, expectations) in sorted(_RETRY_MATRIX.items()):
        for status, expected in sorted(expectations.items()):
            yield pytest.param(
                provider_key, status, expected, id=f"{provider_key}-{status}"
            )


@pytest.mark.parametrize(("provider_key", "status", "expected"), list(_status_cases()))
def test_http_status_retryability_matches_the_contract(provider_key, status, expected):
    """Each provider classifies each HTTP status as the matrix documents.

    The errors are built the way the provider's own client builds them, so a
    classifier that tests for an exception type the installed client no longer
    raises fails here instead of silently retrying nothing.
    """
    build_error, _ = _RETRY_MATRIX[provider_key]
    provider = _build(provider_key)
    error = build_error(status)

    assert provider._is_retryable_error(error) is expected, (
        f"{provider_key}: HTTP {status} classified as "
        f"retryable={not expected}, expected {expected}. Error was "
        f"{type(error).__module__}.{type(error).__name__}: {str(error)[:120]!r}"
    )


@pytest.mark.parametrize("provider_key", sorted(_RETRY_MATRIX))
def test_throttling_is_retryable_for_every_provider(provider_key):
    """No provider may treat a 429 as fatal — the baseline retry case."""
    build_error, _ = _RETRY_MATRIX[provider_key]
    provider = _build(provider_key)

    assert provider._is_retryable_error(build_error(429)) is True, (
        f"{provider_key}: a 429 must be retryable"
    )


@pytest.mark.parametrize("provider_key", sorted(_RETRY_MATRIX))
def test_client_errors_are_not_retryable_for_every_provider(provider_key):
    """A 400 must never be retried: the request itself is wrong."""
    build_error, _ = _RETRY_MATRIX[provider_key]
    provider = _build(provider_key)

    assert provider._is_retryable_error(build_error(400)) is False, (
        f"{provider_key}: a 400 must not be retried"
    )


# ---------------------------------------------------------------------------
# Transport failures: no HTTP status exists yet
# ---------------------------------------------------------------------------

# provider key -> the transport exceptions its client actually raises. The
# OpenAI-compatible providers wrap these in APIConnectionError/APITimeoutError;
# NVIDIA (requests) and Google (httpx + requests) surface them raw.
_TRANSPORT_ERRORS = {
    "nvidia": [
        requests.exceptions.ConnectionError("dns failure"),
        requests.exceptions.ReadTimeout("read timed out"),
    ],
    "google_genai": [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("read timed out"),
        requests.exceptions.ConnectionError("dns failure"),
    ],
}


def _openai_transport_errors():
    from openai import APIConnectionError, APITimeoutError

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return [APIConnectionError(request=request), APITimeoutError(request=request)]


for _key in ("azure_openai", "deepseek", "moonshot", "zai", "bedrock_openai"):
    _TRANSPORT_ERRORS[_key] = _openai_transport_errors()


def _transport_cases():
    for provider_key, errors in sorted(_TRANSPORT_ERRORS.items()):
        for error in errors:
            yield pytest.param(
                provider_key,
                error,
                id=f"{provider_key}-{type(error).__name__}",
            )


@pytest.mark.parametrize(("provider_key", "error"), list(_transport_cases()))
def test_transport_failures_are_retryable(provider_key, error):
    """A connection that never completed carries no status — still retry it.

    The request was never processed, so retrying is safe *and* necessary: a DNS
    blip or read timeout otherwise discards a whole batch (and the tokens
    already spent on it) on attempt 1.
    """
    provider = _build(provider_key)

    assert provider._is_retryable_error(error) is True, (
        f"{provider_key}: {type(error).__name__} must be retryable"
    )


@pytest.mark.parametrize("provider_key", sorted(_RETRY_MATRIX))
def test_configuration_errors_are_never_retryable(provider_key):
    """A programming/config error must abort, not burn every retry slot."""
    provider = _build(provider_key)

    for error in (ValueError("bad model id"), KeyError("missing"), TypeError("nope")):
        assert provider._is_retryable_error(error) is False, (
            f"{provider_key}: {type(error).__name__} must not be retryable"
        )


# ---------------------------------------------------------------------------
# Backoff: bounded, non-negative, and monotonic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_key", sorted(_RETRY_MATRIX))
def test_backoff_is_bounded_and_never_negative(provider_key):
    """Backoff must stay within [0, max_wait] for every attempt.

    ``time.sleep`` raises on a negative value, so an unclamped backoff turns a
    retryable failure into an unrelated crash — and an unbounded one stalls the
    run. Checked past the cap to catch a missing ``min()``.
    """
    build_error, expectations = _RETRY_MATRIX[provider_key]
    provider = _build(provider_key)
    error = build_error(429)
    config = RetryConfig(max_retries=5, base_wait=2.0, max_wait=60.0)

    for attempt in range(12):
        wait = provider._calculate_backoff(error, attempt, config)
        assert 0.0 <= wait <= config.max_wait, (
            f"{provider_key}: attempt {attempt} produced backoff {wait}, "
            f"outside [0, {config.max_wait}]"
        )


@pytest.mark.parametrize("provider_key", sorted(_RETRY_MATRIX))
def test_backoff_grows_until_it_saturates(provider_key):
    """Backoff must increase with attempts (a flat curve isn't backoff).

    Guards against a classifier drifting so that the status is no longer read
    and every attempt returns the same base wait.
    """
    build_error, _ = _RETRY_MATRIX[provider_key]
    provider = _build(provider_key)
    error = build_error(429)
    config = RetryConfig(max_retries=5, base_wait=2.0, max_wait=60.0)

    waits = [provider._calculate_backoff(error, a, config) for a in range(4)]
    assert waits[1] > waits[0], (
        f"{provider_key}: backoff did not grow between attempts: {waits}"
    )
    assert all(b >= a for a, b in zip(waits, waits[1:], strict=False)), (
        f"{provider_key}: backoff decreased across attempts: {waits}"
    )


def test_every_provider_is_covered_by_the_retry_matrix():
    """A new provider must appear above, not silently skip retry classification.

    Reflects over the provider package so the matrix can't lapse behind the
    registry — the failure mode being guarded is a provider whose
    ``_is_retryable_error`` nobody ever exercised.
    """
    import importlib
    import inspect
    from pathlib import Path

    from codereview.providers.base import ModelProvider

    providers_dir = Path(__file__).resolve().parent.parent / "codereview" / "providers"
    found = set()
    for path in sorted(providers_dir.glob("*.py")):
        if path.stem in {"__init__", "base", "mixins", "factory"}:
            continue
        module = importlib.import_module(f"codereview.providers.{path.stem}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, ModelProvider)
                and obj is not ModelProvider
                and obj.__module__ == module.__name__
            ):
                found.add(path.stem)

    assert found, "no provider modules found; the scan is broken"
    missing = sorted(found - set(_RETRY_MATRIX))
    assert not missing, (
        f"provider module(s) {missing} have no retry-contract coverage. Add an "
        "entry to _RETRY_MATRIX with the statuses that client actually raises."
    )


def test_no_provider_classifies_by_an_exception_type_its_client_cannot_raise():
    """Every classifier must accept an error its *installed* client produces.

    The meta-guard for the two dead classifiers this file was written for. It
    doesn't inspect source: it feeds each provider the 429 its real client
    builds and requires a True. A classifier keyed on a type the client no
    longer raises cannot pass, however plausible the code reads.
    """
    dead = []
    for provider_key, (build_error, _) in sorted(_RETRY_MATRIX.items()):
        provider = _build(provider_key)
        if not provider._is_retryable_error(build_error(429)):
            dead.append(provider_key)

    assert not dead, (
        f"provider(s) {dead} do not recognize the throttling error their own "
        "client raises — the classifier is dead code. Classify on the status "
        "the installed client reports, not on a legacy exception class."
    )
