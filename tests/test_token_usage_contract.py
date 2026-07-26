"""Contract tests: every provider reports the token counts it was billed for.

``_extract_token_usage`` is the only place a real vendor count enters the run.
When it returns ``(0, 0)``, ``base.py`` silently substitutes an *estimate*
(``bytes // 3 + 50`` / tiktoken) — and an estimate cannot see reasoning tokens
at all, so the report looks confident and reads low. The symptom is a cost
figure nobody can distinguish from a correct one.

That is exactly what happened. ``extract_openai_token_usage`` read
``response_metadata["token_usage"]`` (``prompt_tokens``/``completion_tokens``),
which **only** langchain-openai's Chat Completions converter populates. On the
Responses API path — Azure ``gpt-5.4`` / ``gpt-5.4-pro``,
``use_responses_api: true``, tool-use structured output, i.e. the path where
real counts *were* available — the helper returned ``(0, 0)`` for every
response: 40,000 in / 9,000 out billed (8,500 of them reasoning) recorded as
6,211 / 145, $0.2350 of spend reported as $0.0177.

Every per-provider token test that existed hand-built an ``AIMessage`` with a
``response_metadata["token_usage"]`` dict, so all of them passed for exactly as
long as the extractor was wrong. The rule these tests encode is the same one
``test_retry_contract.py`` encodes for errors: **build the response the way the
real client builds it.** Each case below drives the response payload through
the installed client's own converter (``BaseChatOpenAI._generate`` →
``_create_chat_result`` / ``_construct_lc_result_from_responses_api``,
``langchain_aws``'s ``_parse_response``, ``ChatNVIDIA._process_generate_response``,
``langchain_google_genai``'s ``_response_to_result``) and then asserts the
provider's extractor reads the vendor's numbers back out.
"""

import copy
import importlib
import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    BedrockOpenAIConfig,
    DeepSeekConfig,
    GoogleGenAIConfig,
    ModelConfig,
    MoonshotConfig,
    NVIDIAConfig,
    PricingConfig,
    ZAIConfig,
)
from codereview.providers.base import ModelProvider

# Distinctive values so a match cannot be a coincidence (and so an estimate,
# which lands nowhere near these, can't be mistaken for a real count).
IN_TOKENS = 1234
OUT_TOKENS = 567
REASONING_TOKENS = 400


def _model_config(**overrides) -> ModelConfig:
    fields = {
        "id": "usage-contract-model",
        "full_id": "usage-contract-model",
        "name": "Usage Contract Model",
        "aliases": [],
        "pricing": PricingConfig(input_per_million=1.0, output_per_million=2.0),
        "supports_tool_use": True,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


# ---------------------------------------------------------------------------
# Response payloads — the wire shapes each vendor actually returns
# ---------------------------------------------------------------------------

_CHAT_COMPLETION_PAYLOAD = {
    "id": "chatcmpl-usage-contract",
    "object": "chat.completion",
    "created": 1,
    "model": "usage-contract-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }
    ],
    "usage": {
        "prompt_tokens": IN_TOKENS,
        "completion_tokens": OUT_TOKENS,
        "total_tokens": IN_TOKENS + OUT_TOKENS,
    },
}

# The Responses API spells usage differently *and* is the only path that reports
# reasoning tokens — the ones an estimate can never recover.
_RESPONSES_PAYLOAD = {
    "id": "resp-usage-contract",
    "object": "response",
    "created_at": 1,
    "model": "usage-contract-model",
    "status": "completed",
    "output": [
        {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": IN_TOKENS,
        "output_tokens": OUT_TOKENS,
        "total_tokens": IN_TOKENS + OUT_TOKENS,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": REASONING_TOKENS},
    },
}

_CONVERSE_PAYLOAD = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    # Converse is camelCase; usage_metadata's snake_case is LangChain's
    # normalization, and langchain_aws *pops* this key before response_metadata
    # is built — see BedrockProvider._extract_token_usage.
    "usage": {
        "inputTokens": IN_TOKENS,
        "outputTokens": OUT_TOKENS,
        "totalTokens": IN_TOKENS + OUT_TOKENS,
    },
    "metrics": {"latencyMs": 10},
    "ResponseMetadata": {"RequestId": "req-1", "HTTPStatusCode": 200},
}

_GEMINI_PAYLOAD = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finish_reason": "STOP",
            "index": 0,
        }
    ],
    "usage_metadata": {
        "prompt_token_count": IN_TOKENS,
        "candidates_token_count": OUT_TOKENS,
        "total_token_count": IN_TOKENS + OUT_TOKENS,
    },
}


# ---------------------------------------------------------------------------
# Message builders — each routes its payload through the real client
# ---------------------------------------------------------------------------


def _invoke_chat_completions(llm):
    """Drive ``BaseChatOpenAI._generate``'s Chat Completions branch.

    Patching ``llm.client.with_raw_response`` rather than returning an
    ``AIMessage`` directly is what makes this a contract test: the full
    ``invoke`` path is where langchain-core merges ``llm_output`` into
    ``response_metadata``, so ``token_usage`` only exists at all because the
    real code put it there.
    """
    from openai.types.chat import ChatCompletion

    raw = Mock()
    raw.parse.return_value = ChatCompletion.model_validate(_CHAT_COMPLETION_PAYLOAD)
    raw.headers = {}

    with patch.object(llm.client, "with_raw_response", create=True) as with_raw:
        with_raw.create.return_value = raw
        return llm.invoke("review this")


def _invoke_responses_api(llm):
    """Drive ``BaseChatOpenAI._generate``'s Responses API branch.

    This branch returns ``_construct_lc_result_from_responses_api(...)``
    directly, bypassing ``_create_chat_result`` — which is why
    ``response_metadata["token_usage"]`` is absent here and reading it was a
    silent zero.
    """
    from openai.types.responses import Response

    raw = Mock()
    raw.parse.return_value = Response.model_validate(_RESPONSES_PAYLOAD)
    raw.headers = {}

    with patch.object(
        llm.root_client.responses, "with_raw_response", create=True
    ) as with_raw:
        with_raw.create.return_value = raw
        with_raw.parse.return_value = raw
        return llm.invoke("review this")


def _openai_client_message(llm, *, responses_api: bool):
    return (
        _invoke_responses_api(llm) if responses_api else _invoke_chat_completions(llm)
    )


def _message_azure(*, responses_api):
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_endpoint="https://test.openai.azure.com",
        api_key="test-key-12345678901234567890",
        api_version="2024-01-01",
        azure_deployment="usage-contract-deployment",
        model="usage-contract-deployment",
        use_responses_api=responses_api or None,
    )
    return _openai_client_message(llm, responses_api=responses_api)


def _message_openai_compat(base_url, *, responses_api=False):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key="test-key-1234567890abcdef",
        model="usage-contract-model",
        base_url=base_url,
        use_responses_api=responses_api or None,
    )
    return _openai_client_message(llm, responses_api=responses_api)


def _message_deepseek():
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(api_key="test-key-1234567890abcdef", model="deepseek-chat")
    return _invoke_chat_completions(llm)


def _message_moonshot():
    from langchain_moonshot import ChatMoonshot

    llm = ChatMoonshot(api_key="test-key-1234567890abcdef", model="kimi-k2.6")
    return _invoke_chat_completions(llm)


def _message_bedrock():
    """Route the Converse payload through langchain_aws's own parser.

    Deep-copied because ``_parse_response`` *mutates* what it is handed — it
    pops ``output`` and ``usage`` off the dict (the reason ``usage`` never
    reaches ``response_metadata``, per BedrockProvider._extract_token_usage), so
    a shared payload would be consumed by the first caller.
    """
    from langchain_aws.chat_models.bedrock_converse import _parse_response

    return _parse_response(copy.deepcopy(_CONVERSE_PAYLOAD))


def _message_nvidia():
    """Drive ``ChatNVIDIA._generate`` with a real ``requests.Response``.

    NIM's client calls ``response.json()`` in ``_process_response`` and raises
    "Received ill-formed response" for anything without it, so a dict mock
    cannot reach the converter — the same reason ``test_retry_contract.py``'s
    ``_nim_error`` builds a real ``requests.Response``.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(_CHAT_COMPLETION_PAYLOAD).encode()
    response.headers["Content-Type"] = "application/json"
    response.url = "https://integrate.api.nvidia.com/v1/chat/completions"

    llm = ChatNVIDIA(
        api_key="nvapi-test-1234567890abcdef", model="moonshotai/kimi-k2.6"
    )
    with patch.object(type(llm._client), "get_req", return_value=response):
        return llm.invoke("review this")


def _message_google():
    from google.genai.types import GenerateContentResponse
    from langchain_google_genai.chat_models import _response_to_result

    response = GenerateContentResponse.model_validate(_GEMINI_PAYLOAD)
    return _response_to_result(response).generations[0].message


# ---------------------------------------------------------------------------
# Provider builders (client patched; nothing reaches a network)
# ---------------------------------------------------------------------------


def _provider_azure(model_config):
    from codereview.providers.azure_openai import AzureOpenAIProvider

    cfg = AzureOpenAIConfig(
        endpoint="https://test.openai.azure.com",
        api_key="test-key-12345678901234567890",
        api_version="2024-01-01",
    )
    mc = model_config.model_copy(
        update={"deployment_name": "usage-contract-deployment"}
    )
    return (
        "codereview.providers.azure_openai.AzureChatOpenAI",
        lambda: AzureOpenAIProvider(mc, cfg),
    )


def _provider_bedrock(model_config):
    from codereview.providers.bedrock import BedrockProvider

    return (
        "codereview.providers.bedrock.ChatBedrockConverse",
        lambda: BedrockProvider(model_config, BedrockConfig(region="us-west-2")),
    )


def _provider_bedrock_openai(model_config):
    from codereview.providers.bedrock_openai import BedrockOpenAIProvider

    cfg = BedrockOpenAIConfig(
        api_key="test-key-1234567890abcdef",
        base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    )
    return (
        "codereview.providers.bedrock_openai.ChatOpenAI",
        lambda: BedrockOpenAIProvider(model_config, cfg),
    )


def _provider_deepseek(model_config):
    from codereview.providers.deepseek import DeepSeekProvider

    cfg = DeepSeekConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.deepseek.ChatDeepSeek",
        lambda: DeepSeekProvider(model_config, cfg),
    )


def _provider_google(model_config):
    from codereview.providers.google_genai import GoogleGenAIProvider

    cfg = GoogleGenAIConfig(api_key="test-google-api-key-12345")
    return (
        "codereview.providers.google_genai.ChatGoogleGenerativeAI",
        lambda: GoogleGenAIProvider(model_config, cfg),
    )


def _provider_moonshot(model_config):
    from codereview.providers.moonshot import MoonshotProvider

    cfg = MoonshotConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.moonshot.ChatMoonshot",
        lambda: MoonshotProvider(model_config, cfg),
    )


def _provider_nvidia(model_config):
    from codereview.providers.nvidia import NVIDIAProvider

    cfg = NVIDIAConfig(api_key="nvapi-test-1234567890abcdef")
    return (
        "codereview.providers.nvidia.ChatNVIDIA",
        lambda: NVIDIAProvider(model_config, cfg),
    )


def _provider_zai(model_config):
    from codereview.providers.zai import ZAIProvider

    cfg = ZAIConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.zai.ChatOpenAI",
        lambda: ZAIProvider(model_config, cfg),
    )


def _build_provider(provider_key, model_config=None):
    """Construct a provider with its LangChain client patched out."""
    model_config = model_config or _model_config()
    patch_target, build = _USAGE_MATRIX[provider_key][0](model_config)

    with patch(patch_target) as mock_client:
        instance = MagicMock()
        instance.with_structured_output.return_value = MagicMock()
        mock_client.return_value = instance
        return build()


# provider key -> (provider builder, message builders keyed by wire API).
#
# Every provider must read its own client's usage. The ``responses`` entries are
# the regression this file was written for: only the two providers that can set
# ``use_responses_api`` have one, and both read ``(0, 0)`` before the fix.
_USAGE_MATRIX = {
    "azure_openai": (
        _provider_azure,
        {
            "chat_completions": lambda: _message_azure(responses_api=False),
            "responses": lambda: _message_azure(responses_api=True),
        },
    ),
    "bedrock": (_provider_bedrock, {"converse": _message_bedrock}),
    "bedrock_openai": (
        _provider_bedrock_openai,
        {
            "chat_completions": lambda: _message_openai_compat(
                "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
            ),
            "responses": lambda: _message_openai_compat(
                "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
                responses_api=True,
            ),
        },
    ),
    "deepseek": (_provider_deepseek, {"chat_completions": _message_deepseek}),
    "google_genai": (_provider_google, {"generate_content": _message_google}),
    "moonshot": (_provider_moonshot, {"chat_completions": _message_moonshot}),
    "nvidia": (_provider_nvidia, {"chat_completions": _message_nvidia}),
    "zai": (
        _provider_zai,
        {
            "chat_completions": lambda: _message_openai_compat(
                "https://api.z.ai/api/paas/v4/"
            )
        },
    ),
}


def _usage_cases():
    for provider_key, (_, message_builders) in sorted(_USAGE_MATRIX.items()):
        for api_name in sorted(message_builders):
            yield pytest.param(provider_key, api_name, id=f"{provider_key}-{api_name}")


@pytest.mark.parametrize(("provider_key", "api_name"), list(_usage_cases()))
def test_extractor_reads_the_vendors_own_counts(provider_key, api_name):
    """Each provider's extractor returns the counts its real client reported.

    Built through the installed client's converter, so an extractor keyed on a
    field that client doesn't populate fails here instead of quietly degrading
    the whole run to estimates.
    """
    provider = _build_provider(provider_key)
    message = _USAGE_MATRIX[provider_key][1][api_name]()

    assert provider._extract_token_usage(message) == (IN_TOKENS, OUT_TOKENS), (
        f"{provider_key} ({api_name}): extractor did not read the vendor's "
        f"counts. usage_metadata={getattr(message, 'usage_metadata', None)!r} "
        f"response_metadata keys="
        f"{sorted(getattr(message, 'response_metadata', {}) or {})}"
    )


@pytest.mark.parametrize(("provider_key", "api_name"), list(_usage_cases()))
def test_extractor_never_silently_returns_zero(provider_key, api_name):
    """A ``(0, 0)`` return is indistinguishable from "no usage was reported".

    ``base.py`` treats a zero as "estimate it instead", so this is the shape of
    the failure rather than an exception: the cost figure stays plausible and
    reads low by exactly the reasoning tokens.
    """
    provider = _build_provider(provider_key)
    message = _USAGE_MATRIX[provider_key][1][api_name]()
    input_tokens, output_tokens = provider._extract_token_usage(message)

    assert input_tokens > 0 and output_tokens > 0, (
        f"{provider_key} ({api_name}): extractor returned "
        f"({input_tokens}, {output_tokens}); base.py will substitute an "
        "estimate, which cannot see reasoning tokens."
    )


def test_responses_api_usage_survives_the_full_provider_path():
    """End to end: a Responses-API batch bills what the vendor charged.

    The Azure ``gpt-5.4`` shape — ``use_responses_api`` + tool-use structured
    output, where ``include_raw`` makes the raw ``AIMessage`` available and the
    real counts *were* there to be read. Asserted on the provider's own totals
    (not just the extractor) because the under-report was only visible after
    ``_track_tokens``: the reported cost was 13x low.
    """
    from codereview.models import CodeReviewReport

    model_config = _model_config(
        deployment_name="usage-contract-deployment", use_responses_api=True
    )
    patch_target, build = _provider_azure(model_config)

    with patch(patch_target) as mock_client:
        instance = MagicMock()
        instance.with_structured_output.return_value = MagicMock()
        mock_client.return_value = instance
        provider = build()

    raw = _message_azure(responses_api=True)
    parsed = CodeReviewReport(summary="ok", issues=[])

    with patch.object(
        provider, "_invoke_chain", return_value={"raw": raw, "parsed": parsed}
    ):
        from codereview.providers.base import RetryConfig

        provider._execute_with_retry(
            {"code_content": "x = 1"},
            RetryConfig(max_retries=1, base_wait=0.0, max_wait=0.0),
            "batch 1/1",
        )

    assert (provider._total_input_tokens, provider._total_output_tokens) == (
        IN_TOKENS,
        OUT_TOKENS,
    ), (
        "Responses-API usage did not reach the token totals; the run would "
        "report an estimate that cannot see the "
        f"{REASONING_TOKENS} reasoning tokens the vendor billed."
    )


def test_every_provider_is_covered_by_the_usage_matrix():
    """A new provider must appear above, not silently skip usage coverage.

    Reflects over the provider package rather than trusting the matrix, because
    the failure being guarded is precisely a ``_extract_token_usage`` nobody
    exercised against a real client response — the state every OpenAI-compatible
    provider was already in.
    """
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
    missing = sorted(found - set(_USAGE_MATRIX))
    assert not missing, (
        f"provider module(s) {missing} have no token-usage coverage. Add an "
        "entry to _USAGE_MATRIX whose message builder routes a response "
        "through that client's own converter."
    )


def test_every_responses_api_capable_provider_is_covered_on_that_path():
    """``use_responses_api`` support implies a ``responses`` matrix entry.

    The bug lived on exactly this path for exactly this reason: the providers
    that can select the Responses API were only ever tested on Chat Completions,
    where ``response_metadata["token_usage"]`` happens to exist.
    """
    providers_dir = Path(__file__).resolve().parent.parent / "codereview" / "providers"
    capable = {
        path.stem
        for path in sorted(providers_dir.glob("*.py"))
        if path.stem not in {"__init__", "base", "mixins", "factory"}
        and "use_responses_api" in path.read_text()
    }
    assert capable, "no use_responses_api call sites found; the scan is broken"

    missing = sorted(
        key
        for key in capable
        if "responses" not in _USAGE_MATRIX.get(key, (None, {}))[1]
    )
    assert not missing, (
        f"provider(s) {missing} can set use_responses_api but are only covered "
        "on Chat Completions. Add a 'responses' message builder — that path "
        "does not populate response_metadata['token_usage']."
    )
