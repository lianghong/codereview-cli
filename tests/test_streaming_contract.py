"""Contract tests: ``--stream`` must only cost what it actually buys.

Streaming touches three things that have to agree, and each of them was wrong
in a different way:

1. **Which handler wants tokens.** Providers passed ``streaming=bool(callbacks)``,
   but ``--verbose`` also installs a callback — ``ProgressCallbackHandler``,
   which does not override ``on_llm_new_token`` and therefore cannot observe a
   single streamed token. So every ``--verbose`` run on an OpenAI-compatible
   provider was switched onto the streaming wire path for nothing.
2. **Usage on the streaming path.** ``stream_options={"include_usage": True}``
   is only sent when ``stream_usage`` is set, and langchain-openai auto-enables
   it *only* when no ``base_url`` is configured — we always configure one. A
   real OpenAI-compatible server then sends no usage chunk at all, so
   ``usage_metadata`` is ``None``, ``extract_openai_token_usage`` returns
   ``(0, 0)``, and ``base.py`` silently substitutes its byte-heuristic estimate
   (which cannot see reasoning tokens).
3. **Who can stream at all.** ``--stream`` drops the run to one worker. On
   Bedrock, NVIDIA and Google no token ever reaches a callback, so the flag was
   buying a 3-5x slowdown for output that cannot appear.
"""

from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import pytest
from rich.console import Console

from codereview.callbacks import ProgressCallbackHandler, StreamingCallbackHandler
from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockOpenAIConfig,
    DeepSeekConfig,
    ModelConfig,
    MoonshotConfig,
    PricingConfig,
    ZAIConfig,
)
from codereview.providers.mixins import openai_stream_params, wants_token_streaming


@pytest.fixture
def sample_code_dir(tmp_path):
    """Two files, so ``batch_size=1`` yields two parallelizable batches."""
    for i in range(2):
        (tmp_path / f"test{i}.py").write_text(f"def hello{i}():\n    return 'world'\n")
    return tmp_path


# ---------------------------------------------------------------------------
# wants_token_streaming: "does any handler consume tokens?"
# ---------------------------------------------------------------------------


def test_no_callbacks_do_not_want_streaming():
    assert wants_token_streaming(None) is False
    assert wants_token_streaming([]) is False


def test_progress_handler_does_not_want_streaming():
    """The --verbose handler shows a spinner; it never sees a token.

    This is the regression: ``bool(self.callbacks)`` was True here, so
    ``--verbose`` alone put every OpenAI-compatible provider on the streaming
    wire path to feed a handler with no ``on_llm_new_token``.
    """
    assert wants_token_streaming([ProgressCallbackHandler(console=Console())]) is False


def test_streaming_handler_wants_streaming():
    assert wants_token_streaming([StreamingCallbackHandler(console=Console())]) is True


def test_a_token_consuming_handler_among_others_is_enough():
    """Any one consumer turns streaming on — the check is over-, not all-."""
    handlers = [
        ProgressCallbackHandler(console=Console()),
        StreamingCallbackHandler(console=Console()),
    ]
    assert wants_token_streaming(handlers) is True


def test_detection_is_by_override_not_by_class_identity():
    """A third-party handler that overrides on_llm_new_token counts too.

    ``mixins.py`` must not import the concrete handler classes (it is imported
    by every provider, and ``codereview.callbacks`` pulls in Rich), so the test
    is an override check — which also means it generalizes.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class Consumer(BaseCallbackHandler):
        def on_llm_new_token(self, token, **kwargs):  # noqa: ANN001, ANN003
            pass

    class Bystander(BaseCallbackHandler):
        pass

    assert wants_token_streaming([Consumer()]) is True
    assert wants_token_streaming([Bystander()]) is False


# ---------------------------------------------------------------------------
# openai_stream_params: streaming and stream_usage travel together
# ---------------------------------------------------------------------------


def test_stream_usage_is_never_requested_without_streaming():
    """``stream_usage`` is inert off the streaming path — don't set it there.

    ``_stream`` is the only place langchain-openai converts it into
    ``stream_options``; a non-streaming request payload omits it entirely.
    """
    params = openai_stream_params(None)
    assert params == {"streaming": False}


def test_stream_usage_accompanies_streaming():
    """Turning streaming on without it loses the billed token counts."""
    params = openai_stream_params([StreamingCallbackHandler(console=Console())])
    assert params == {"streaming": True, "stream_usage": True}


# ---------------------------------------------------------------------------
# Per-provider: the kwargs that actually reach the vendor client
# ---------------------------------------------------------------------------


def _model_config(**overrides) -> ModelConfig:
    base = {
        "id": "stream-model",
        "full_id": "stream-model",
        "name": "Stream Model",
        "aliases": [],
        "pricing": PricingConfig(input_per_million=1.0, output_per_million=2.0),
    }
    return ModelConfig(**{**base, **overrides})


def _build_azure(callbacks):
    from codereview.providers.azure_openai import AzureOpenAIProvider

    cfg = AzureOpenAIConfig(
        endpoint="https://test.openai.azure.com",
        api_key="test-key-12345678901234567890",
        api_version="2024-01-01",
    )
    mc = _model_config(deployment_name="stream-deployment")
    return (
        "codereview.providers.azure_openai.AzureChatOpenAI",
        lambda: AzureOpenAIProvider(mc, cfg, callbacks=callbacks),
    )


def _build_bedrock_openai(callbacks):
    from codereview.providers.bedrock_openai import BedrockOpenAIProvider

    cfg = BedrockOpenAIConfig(
        api_key="test-key-1234567890abcdef",
        base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    )
    return (
        "codereview.providers.bedrock_openai.ChatOpenAI",
        lambda: BedrockOpenAIProvider(_model_config(), cfg, callbacks=callbacks),
    )


def _build_zai(callbacks):
    from codereview.providers.zai import ZAIProvider

    cfg = ZAIConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.zai.ChatOpenAI",
        lambda: ZAIProvider(_model_config(), cfg, callbacks=callbacks),
    )


def _build_deepseek(callbacks):
    from codereview.providers.deepseek import DeepSeekProvider

    cfg = DeepSeekConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.deepseek.ChatDeepSeek",
        lambda: DeepSeekProvider(_model_config(), cfg, callbacks=callbacks),
    )


def _build_moonshot(callbacks):
    from codereview.providers.moonshot import MoonshotProvider

    cfg = MoonshotConfig(api_key="test-key-1234567890abcdef")
    return (
        "codereview.providers.moonshot.ChatMoonshot",
        lambda: MoonshotProvider(_model_config(), cfg, callbacks=callbacks),
    )


# Every provider that builds an OpenAI-compatible client, i.e. every provider
# whose _create_model must go through openai_stream_params.
_OPENAI_COMPAT_BUILDERS = {
    "azure_openai": _build_azure,
    "bedrock_openai": _build_bedrock_openai,
    "deepseek": _build_deepseek,
    "moonshot": _build_moonshot,
    "zai": _build_zai,
}


def _client_kwargs(provider_key, callbacks):
    patch_target, build = _OPENAI_COMPAT_BUILDERS[provider_key](callbacks)
    with patch(patch_target) as mock_client:
        mock_client.return_value = MagicMock()
        build()
        return mock_client.call_args.kwargs


@pytest.mark.parametrize("provider_key", sorted(_OPENAI_COMPAT_BUILDERS))
def test_no_callbacks_means_no_streaming(provider_key):
    kwargs = _client_kwargs(provider_key, None)
    assert kwargs["streaming"] is False
    assert "stream_usage" not in kwargs


@pytest.mark.parametrize("provider_key", sorted(_OPENAI_COMPAT_BUILDERS))
def test_progress_callback_does_not_enable_streaming(provider_key):
    """--verbose must not move the provider onto the streaming wire path."""
    kwargs = _client_kwargs(provider_key, [ProgressCallbackHandler(console=Console())])
    assert kwargs["streaming"] is False, (
        f"{provider_key}: --verbose enabled streaming for a handler that "
        "cannot consume tokens"
    )


@pytest.mark.parametrize("provider_key", sorted(_OPENAI_COMPAT_BUILDERS))
def test_streaming_always_requests_usage(provider_key):
    """Streaming without ``stream_usage`` silently loses the billed counts."""
    kwargs = _client_kwargs(provider_key, [StreamingCallbackHandler(console=Console())])
    assert kwargs["streaming"] is True
    assert kwargs["stream_usage"] is True, (
        f"{provider_key}: streaming enabled without stream_usage — the final "
        "usage chunk is never sent, so token counts fall back to the estimate"
    )


def test_streaming_kwargs_are_the_ones_the_real_client_accepts():
    """Drive a real ChatOpenAI with the streaming kwargs and check the effect.

    The per-provider tests above assert on what we *pass*; this one asserts the
    kwargs actually make the client stream, so a rename upstream (or a kwarg the
    client quietly ignores) is caught rather than passing forever.
    """
    from langchain_openai import ChatOpenAI

    params = openai_stream_params([StreamingCallbackHandler(console=Console())])
    client = ChatOpenAI(
        model="stream-model",
        api_key="sk-test-not-a-real-key",
        base_url="https://example.invalid/v1",
        **params,
    )
    assert client._should_stream(async_api=False) is True
    assert client.stream_usage is True

    quiet = ChatOpenAI(
        model="stream-model",
        api_key="sk-test-not-a-real-key",
        base_url="https://example.invalid/v1",
        **openai_stream_params(None),
    )
    assert quiet._should_stream(async_api=False) is False


# ---------------------------------------------------------------------------
# supports_token_streaming: answered from the class, before construction
# ---------------------------------------------------------------------------

# Providers where no token ever reaches a callback, with the reason. Written out
# rather than derived so flipping one is a deliberate, reviewed edit:
#   bedrock      — _create_model passes disable_streaming=True (and the
#                  read_timeout: 1800 overrides depend on the non-streaming
#                  Converse path)
#   nvidia       — ChatNVIDIA has no `streaming` model field at all
#   google_genai — could stream, but method="json_schema" structured output
#                  through the streaming path is unproven live
_NON_STREAMING_PROVIDERS = {"bedrock", "nvidia", "google_genai"}


def _provider_classes():
    from importlib import import_module

    from codereview.providers.factory import _PROVIDER_REGISTRY

    return {
        name: getattr(import_module(entry.module), entry.class_name)
        for name, entry in _PROVIDER_REGISTRY.items()
    }


def test_every_provider_answers_streaming_support_without_an_instance():
    """The CLI must be able to ask before it has credentials or a provider.

    Attaching a streaming handler and choosing a worker count are one decision
    (concurrent Rich ``Live`` displays corrupt terminal state — see
    callbacks.py), and both feed the provider constructor, so the answer has to
    come from the class.
    """
    for name, provider_class in _provider_classes().items():
        answer = provider_class.supports_token_streaming()
        assert isinstance(answer, bool), f"{name}: expected a bool, got {answer!r}"


def test_the_non_streaming_provider_set_is_exactly_the_documented_one():
    actual = {
        name
        for name, cls in _provider_classes().items()
        if not cls.supports_token_streaming()
    }
    assert actual == _NON_STREAMING_PROVIDERS, (
        "the set of providers that cannot stream changed. Flipping one to True "
        "needs a live run proving tokens reach a callback; flipping one to "
        "False needs the reason recorded on the override."
    )


def test_factory_answers_streaming_support_without_building_a_client():
    """Asking must not construct the vendor client (nor need credentials).

    ``--stream`` has no business demanding credentials earlier than the run
    itself does.
    """
    from codereview.providers.factory import ProviderFactory

    with patch("codereview.providers.bedrock.ChatBedrockConverse") as mock_client:
        assert ProviderFactory().supports_token_streaming("opus5") is False
        mock_client.assert_not_called()


def test_factory_defers_an_unresolvable_model_to_create_provider():
    """A bad --model must fail with the resolver's message, not here."""
    from codereview.providers.factory import ProviderFactory

    assert ProviderFactory().supports_token_streaming("no-such-model-xyz") is True


# ---------------------------------------------------------------------------
# CLI: --stream keeps concurrency when it would buy nothing
# ---------------------------------------------------------------------------


def _cli_mocks(code_dir, file_count=2):
    """run_review with the analyzer/scanner mocked but a REAL ProviderFactory.

    The factory is left real on purpose: ``supports_token_streaming`` is exactly
    what's under test, and a MagicMock's truthy return would make these tests
    pass vacuously. It needs no credentials — the answer comes from the class.
    """
    from contextlib import ExitStack

    from codereview.models import CodeReviewReport, ReviewMetrics

    stack = ExitStack()
    mock_analyzer_cls = stack.enter_context(patch("codereview.cli.CodeAnalyzer"))
    mock_scanner_cls = stack.enter_context(patch("codereview.cli.FileScanner"))

    mock_provider = Mock()
    mock_provider.total_input_tokens = 100
    mock_provider.total_output_tokens = 50
    mock_provider.get_pricing.return_value = {
        "input_price_per_million": 5.0,
        "output_price_per_million": 25.0,
    }

    mock_analyzer = Mock()
    mock_analyzer.provider = mock_provider
    mock_analyzer.analyze_batch.return_value = CodeReviewReport(
        summary="Test",
        metrics=ReviewMetrics(files_analyzed=1),
        issues=[],
        system_design_insights="No issues",
        recommendations=[],
        improvement_suggestions=[],
    )
    mock_analyzer.skipped_files = []
    mock_analyzer_cls.return_value = mock_analyzer

    mock_scanner = Mock()
    mock_scanner.scan.return_value = [
        code_dir / f"test{i}.py" for i in range(file_count)
    ]
    mock_scanner.skipped_files = []
    mock_scanner_cls.return_value = mock_scanner

    return stack


def _run_and_capture_workers(code_dir, model, *, stream):
    """Run run_review and return (max_workers, console output).

    ``batch_size=1`` with two files guarantees two batches, so the worker count
    is decided by *streaming* rather than by there being nothing to parallelize.
    """
    from concurrent.futures import ThreadPoolExecutor

    from codereview.cli import run_review

    recorded: list[int | None] = []

    def spy(max_workers=None, **kwargs):
        recorded.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers, **kwargs)

    buffer = StringIO()
    with _cli_mocks(code_dir), patch("codereview.cli.ThreadPoolExecutor", new=spy):
        run_review(
            code_dir,
            console=Console(file=buffer, width=200),
            no_readme=True,
            model_name=model,
            batch_size=1,
            stream=stream,
        )

    assert recorded, "run_review never built a ThreadPoolExecutor"
    return recorded[-1], buffer.getvalue()


def test_stream_on_a_non_streaming_provider_keeps_parallel_batches(sample_code_dir):
    """--stream on Bedrock must not cost the 3-5x multi-batch speedup.

    No token reaches a callback there, so serializing the run bought nothing.
    """
    workers, _ = _run_and_capture_workers(sample_code_dir, "opus5", stream=True)
    assert workers > 1, (
        "--stream dropped to one worker on a provider that never streams a token"
    )


def test_stream_on_a_non_streaming_provider_says_so(sample_code_dir):
    """Ignoring a flag silently is worse than the slowdown it replaces."""
    _, output = _run_and_capture_workers(sample_code_dir, "opus5", stream=True)
    assert "--stream ignored" in output
    assert "Keeping parallel batches" in output


def test_stream_still_serializes_where_tokens_do_arrive(sample_code_dir):
    """The downgrade must be narrow: a streaming provider still goes sequential.

    Token-by-token output from concurrent batches interleaves incomprehensibly,
    so this half of the original behaviour has to survive.
    """
    workers, output = _run_and_capture_workers(sample_code_dir, "glm-5.2", stream=True)
    assert workers == 1
    assert "--stream ignored" not in output


def test_no_stream_flag_is_unaffected(sample_code_dir):
    """Without --stream nothing changes, and no notice is printed."""
    workers, output = _run_and_capture_workers(sample_code_dir, "opus5", stream=False)
    assert workers > 1
    assert "--stream ignored" not in output
