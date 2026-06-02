import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from codereview.models import CodeReviewReport, ReviewMetrics
from codereview.providers.base import ModelProvider
from codereview.providers.mixins import is_openai_retryable_error


def _api_status_error(status_code: int) -> APIStatusError:
    """Build an APIStatusError with a given HTTP status for retry tests."""
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError("boom", response=response, body=None)


def test_is_openai_retryable_error_retries_transient():
    """Rate limits, timeouts, connection errors, and 5xx are retryable."""
    req = httpx.Request("POST", "https://example.test/")
    assert is_openai_retryable_error(
        RateLimitError("rl", response=httpx.Response(429, request=req), body=None)
    )
    assert is_openai_retryable_error(APITimeoutError(request=req))
    assert is_openai_retryable_error(APIConnectionError(request=req))
    assert is_openai_retryable_error(_api_status_error(500))
    assert is_openai_retryable_error(_api_status_error(503))


def test_is_openai_retryable_error_skips_client_errors():
    """4xx (other than 429) and unrelated exceptions are NOT retryable."""
    assert not is_openai_retryable_error(_api_status_error(400))
    assert not is_openai_retryable_error(_api_status_error(401))
    assert not is_openai_retryable_error(_api_status_error(404))
    assert not is_openai_retryable_error(ValueError("nope"))


def _rate_limit_error_with_retry_after(value: str) -> RateLimitError:
    """A RateLimitError whose response carries a given Retry-After header."""
    req = httpx.Request("POST", "https://example.test/")
    response = httpx.Response(429, headers={"retry-after": value}, request=req)
    return RateLimitError("rl", response=response, body=None)


def test_parse_retry_after_reads_valid_header():
    """A valid Retry-After is returned, capped at max_wait."""
    from codereview.providers.mixins import parse_retry_after

    err = _rate_limit_error_with_retry_after("3")
    assert parse_retry_after(err, max_wait=60.0) == 3.0
    # Capped at max_wait.
    assert parse_retry_after(_rate_limit_error_with_retry_after("999"), 60.0) == 60.0


def test_parse_retry_after_rejects_negative():
    """Regression: a negative Retry-After must return None, not a negative
    sleep — time.sleep(-1) raises ValueError and would abort the retry."""
    from codereview.providers.mixins import parse_retry_after

    assert parse_retry_after(_rate_limit_error_with_retry_after("-1"), 60.0) is None


def test_parse_retry_after_rejects_non_numeric():
    """A non-numeric Retry-After falls back (None) rather than raising."""
    from codereview.providers.mixins import parse_retry_after

    assert parse_retry_after(_rate_limit_error_with_retry_after("soon"), 60.0) is None


def test_parse_retry_after_none_without_header():
    """No Retry-After header → None (caller uses exponential backoff)."""
    from codereview.providers.mixins import parse_retry_after

    req = httpx.Request("POST", "https://example.test/")
    err = RateLimitError("rl", response=httpx.Response(429, request=req), body=None)
    assert parse_retry_after(err, 60.0) is None
    # A non-rate-limit error is never parsed for Retry-After.
    assert parse_retry_after(ValueError("x"), 60.0) is None


# ---------------------------------------------------------------------------
# _resolve_temperature precedence
# ---------------------------------------------------------------------------


class _Cfg:
    """Minimal stand-in: _resolve_temperature only reads inference_params."""

    def __init__(self, inference_params):
        self.inference_params = inference_params


class _Params:
    def __init__(self, temperature):
        self.temperature = temperature


@pytest.mark.parametrize(
    "override, params, allow_none, provider_default, expected",
    [
        # CLI override wins over everything.
        (0.7, _Params(0.2), False, 0.3, 0.7),
        (0.7, None, True, 0.3, 0.7),
        # Explicit numeric in inference_params beats the provider default.
        (None, _Params(0.2), False, 0.3, 0.2),
        (None, _Params(0.2), True, 0.3, 0.2),
        # No inference_params at all → provider default.
        (None, None, False, 0.3, 0.3),
        (None, None, True, 0.3, 0.3),
        # temperature is None in params:
        #   allow_none=False → fall back to provider default,
        #   allow_none=True  → stay None (reasoning models opt out). This is the
        #   DOCUMENTED design: a reasoning model omits default_temperature in
        #   YAML (loader passes temperature=None) so allow_none returns None.
        (None, _Params(None), False, 0.3, 0.3),
        (None, _Params(None), True, 0.3, None),
    ],
)
def test_resolve_temperature_precedence(
    override, params, allow_none, provider_default, expected
):
    result = ModelProvider._resolve_temperature(
        override=override,
        model_config=_Cfg(params),
        provider_default=provider_default,
        allow_none=allow_none,
    )
    assert result == expected


@pytest.mark.parametrize("bad", [-0.1, 2.1, 5.0])
def test_resolve_temperature_rejects_out_of_range_override(bad):
    """An out-of-range CLI override raises before any provider is built."""
    with pytest.raises(ValueError, match="between 0.0 and 2.0"):
        ModelProvider._resolve_temperature(
            override=bad,
            model_config=_Cfg(None),
            provider_default=0.3,
            allow_none=True,
        )


class ConcreteProvider(ModelProvider):
    """Test implementation of ModelProvider."""

    def __init__(self):
        self._display_name = "Test Model"
        self._input_tokens = 0
        self._output_tokens = 0

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        # Simple test implementation
        self._input_tokens += 100
        self._output_tokens += 50
        return CodeReviewReport(
            summary="Test summary",
            metrics=ReviewMetrics(files_analyzed=len(files_content)),
            issues=[],
            system_design_insights="No design issues found",
            recommendations=[],
            improvement_suggestions=[],
        )

    def get_model_display_name(self) -> str:
        return self._display_name

    def get_pricing(self) -> dict[str, float]:
        # Test pricing: $1 per million tokens
        return {
            "input_price_per_million": 1.0,
            "output_price_per_million": 1.0,
        }

    @property
    def total_input_tokens(self) -> int:
        return self._input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._output_tokens

    def estimate_cost(self) -> dict[str, float]:
        # $1 per million tokens for test
        input_cost = (self._input_tokens / 1_000_000) * 1.0
        output_cost = (self._output_tokens / 1_000_000) * 1.0
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }


def test_concrete_provider_implementation():
    """Test that concrete provider can be instantiated."""
    provider = ConcreteProvider()
    assert provider is not None


def test_analyze_batch():
    """Test analyze_batch returns CodeReviewReport."""
    provider = ConcreteProvider()

    result = provider.analyze_batch(
        batch_number=1,
        total_batches=1,
        files_content={"test.py": "print('hello')"},
    )

    assert isinstance(result, CodeReviewReport)
    assert result.summary == "Test summary"
    assert result.metrics.files_analyzed == 1


def test_get_model_display_name():
    """Test get_model_display_name returns string."""
    provider = ConcreteProvider()
    assert provider.get_model_display_name() == "Test Model"


def test_token_tracking():
    """Test token tracking via properties."""
    provider = ConcreteProvider()

    # Initial state
    assert provider.total_input_tokens == 0
    assert provider.total_output_tokens == 0

    # After analyze_batch
    provider.analyze_batch(1, 1, {"test.py": "code"})
    assert provider.total_input_tokens == 100
    assert provider.total_output_tokens == 50


def test_estimate_cost():
    """Test cost estimation."""
    provider = ConcreteProvider()
    provider.analyze_batch(1, 1, {"test.py": "code"})

    cost = provider.estimate_cost()

    assert cost["input_tokens"] == 100
    assert cost["output_tokens"] == 50
    assert cost["input_cost"] > 0
    assert cost["output_cost"] > 0
    assert cost["total_cost"] == cost["input_cost"] + cost["output_cost"]


def test_reset_state_default():
    """Test default reset_state does nothing but doesn't error."""
    provider = ConcreteProvider()
    provider.reset_state()  # Should not raise


def test_cannot_instantiate_abstract_class():
    """Test ModelProvider cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ModelProvider()


class TestPrepareContextWithReadme:
    """Tests for _prepare_batch_context with project_context parameter."""

    def test_prepends_readme_to_batch_context(self):
        """Verify README appears before file contents with proper delimiters."""
        provider = ConcreteProvider()
        readme_content = "# My Project\n\nThis is a test project."
        files_content = {"test.py": "print('hello')"}

        result = provider._prepare_batch_context(
            batch_number=1,
            total_batches=2,
            files_content=files_content,
            project_context=readme_content,
        )

        # Verify PROJECT CONTEXT section appears before CODE REVIEW section
        assert "== PROJECT CONTEXT ==" in result
        assert "--- README.md ---" in result
        assert readme_content in result
        assert "--- END README ---" in result
        assert "== CODE REVIEW ==" in result

        # Verify ordering: PROJECT CONTEXT comes before file analysis
        project_context_pos = result.index("== PROJECT CONTEXT ==")
        code_review_pos = result.index("== CODE REVIEW ==")
        file_pos = result.index("File: test.py")

        assert project_context_pos < code_review_pos < file_pos

    def test_no_readme_section_when_none(self):
        """Verify no PROJECT CONTEXT section when project_context is None."""
        provider = ConcreteProvider()
        files_content = {"test.py": "print('hello')"}

        result = provider._prepare_batch_context(
            batch_number=1,
            total_batches=2,
            files_content=files_content,
            project_context=None,
        )

        # Verify no PROJECT CONTEXT section
        assert "== PROJECT CONTEXT ==" not in result
        assert "--- README.md ---" not in result
        assert "--- END README ---" not in result
        assert "== CODE REVIEW ==" not in result

        # Verify normal batch context is still present
        assert "Analyzing Batch 1/2" in result
        assert "File: test.py" in result


# ---------------------------------------------------------------------------
# _execute_with_retry contract: result-shape validation
# ---------------------------------------------------------------------------


class _StubChainProvider(ConcreteProvider):
    """ConcreteProvider variant that lets tests inject _invoke_chain results.

    The base ConcreteProvider's analyze_batch returns a fixed
    CodeReviewReport without going through _execute_with_retry, which
    bypasses the retry framework entirely. To exercise the result-shape
    branching in _execute_with_retry we need a provider where
    _invoke_chain is the seam under test.
    """

    def __init__(self, stub_result):
        super().__init__()
        self._stub_result = stub_result

    def _invoke_chain(self, chain_input):
        return self._stub_result


def test_execute_with_retry_rejects_string_result():
    """A provider that returns a plain string surfaces a clear ValueError.

    Locks in the High #1 fix from f40121a: without the isinstance check
    in _execute_with_retry, this case would AttributeError on
    `result.model_dump_json()` during token estimation, which masks the
    real contract violation.
    """
    from codereview.providers.base import RetryConfig

    provider = _StubChainProvider(stub_result="this is not a CodeReviewReport")
    cfg = RetryConfig(max_retries=0)

    with pytest.raises(ValueError, match="unexpected result type"):
        provider._execute_with_retry(
            chain_input={"system_prompt": "x", "batch_context": "y"},
            retry_config=cfg,
            batch_context="y",
        )


def test_execute_with_retry_rejects_list_result():
    """Non-dict, non-CodeReviewReport, non-None — all rejected the same way."""
    from codereview.providers.base import RetryConfig

    provider = _StubChainProvider(stub_result=[1, 2, 3])
    cfg = RetryConfig(max_retries=0)

    with pytest.raises(ValueError, match="unexpected result type"):
        provider._execute_with_retry(
            chain_input={"system_prompt": "x", "batch_context": "y"},
            retry_config=cfg,
            batch_context="y",
        )


def test_execute_with_retry_accepts_codereviewreport():
    """The contract's direct-CodeReviewReport shape is the success path."""
    from codereview.providers.base import RetryConfig

    expected = CodeReviewReport(
        summary="ok",
        metrics=ReviewMetrics(files_analyzed=1),
        issues=[],
        system_design_insights="No design issues found",
        recommendations=[],
        improvement_suggestions=[],
    )
    provider = _StubChainProvider(stub_result=expected)
    cfg = RetryConfig(max_retries=0)

    result = provider._execute_with_retry(
        chain_input={"system_prompt": "x", "batch_context": "y"},
        retry_config=cfg,
        batch_context="y",
    )
    assert result is expected


def test_execute_with_retry_rejects_none_result():
    """None remains its own error case (parse failure), distinct from
    'unexpected type', so the message points at the right cause."""
    from codereview.providers.base import RetryConfig

    provider = _StubChainProvider(stub_result=None)
    cfg = RetryConfig(max_retries=0)

    with pytest.raises(ValueError, match="None"):
        provider._execute_with_retry(
            chain_input={"system_prompt": "x", "batch_context": "y"},
            retry_config=cfg,
            batch_context="y",
        )


# ---------------------------------------------------------------------------
# Malformed include_raw=True result (parsed is None) is retried, not aborted
# ---------------------------------------------------------------------------


class _SequencedChainProvider(ConcreteProvider):
    """Returns a queued result per _invoke_chain call, counting invocations.

    Lets a test assert that a transient parse failure is *retried* rather than
    aborting the batch on the first attempt.
    """

    def __init__(self, results, enable_output_fixing=True):
        super().__init__()
        self._results = list(results)
        self.enable_output_fixing = enable_output_fixing
        self.invocations = 0

    def _invoke_chain(self, chain_input):
        self.invocations += 1
        item = self._results.pop(0)
        # An Exception in the queue is raised (simulates the chain itself
        # throwing, e.g. PydanticOutputParser raising OutputParserException);
        # any other item is returned as the chain result.
        if isinstance(item, Exception):
            raise item
        return item


def _malformed_raw():
    """An include_raw=True dict whose parsed is None (transient bad output)."""
    return {"raw": object(), "parsed": None, "parsing_error": "bad json"}


def _good_raw():
    report = CodeReviewReport(
        summary="ok",
        metrics=ReviewMetrics(files_analyzed=1),
        issues=[],
        system_design_insights="No design issues found",
        recommendations=[],
        improvement_suggestions=[],
    )
    return {"raw": object(), "parsed": report, "parsing_error": None}


def test_parsed_none_is_retried_when_output_fixing_enabled():
    """Regression: a malformed structured-output result (parsed is None) must
    be retried via the enable_output_fixing path, not aborted immediately.

    Previously the parsed-is-None branch raised a plain ValueError that fell
    into the generic `except Exception`, where ValueError is non-retryable, so
    the batch failed on attempt 1. The dedicated OutputParsingRetryError is now
    caught alongside ValidationError and retried.
    """
    from codereview.providers.base import RetryConfig

    provider = _SequencedChainProvider(
        results=[_malformed_raw(), _good_raw()],
        enable_output_fixing=True,
    )
    cfg = RetryConfig(max_retries=3, validation_retry_sleep=0.0)

    result = provider._execute_with_retry(
        chain_input={"system_prompt": "x", "batch_context": "y"},
        retry_config=cfg,
        batch_context="y",
    )

    assert isinstance(result, CodeReviewReport)
    assert provider.invocations == 2  # retried once, then succeeded


def test_parsed_none_raises_after_retries_exhausted():
    """When every attempt is malformed, the retryable error surfaces."""
    from codereview.providers.base import OutputParsingRetryError, RetryConfig

    provider = _SequencedChainProvider(
        results=[_malformed_raw(), _malformed_raw()],
        enable_output_fixing=True,
    )
    cfg = RetryConfig(max_retries=1, validation_retry_sleep=0.0)

    with pytest.raises(
        OutputParsingRetryError, match="Structured output parsing failed"
    ):
        provider._execute_with_retry(
            chain_input={"system_prompt": "x", "batch_context": "y"},
            retry_config=cfg,
            batch_context="y",
        )
    assert provider.invocations == 2  # initial + 1 retry


def test_parsed_none_not_retried_when_output_fixing_disabled():
    """With output fixing off, the malformed result raises on the first attempt."""
    from codereview.providers.base import OutputParsingRetryError, RetryConfig

    provider = _SequencedChainProvider(
        results=[_malformed_raw(), _good_raw()],
        enable_output_fixing=False,
    )
    cfg = RetryConfig(max_retries=3, validation_retry_sleep=0.0)

    with pytest.raises(OutputParsingRetryError):
        provider._execute_with_retry(
            chain_input={"system_prompt": "x", "batch_context": "y"},
            retry_config=cfg,
            batch_context="y",
        )
    assert provider.invocations == 1  # no retry


# ---------------------------------------------------------------------------
# Prompt-parsing path: OutputParserException (malformed JSON) is retried
# ---------------------------------------------------------------------------


def test_output_parser_exception_is_retried_when_output_fixing_enabled():
    """Regression (field failure): on the prompt-parsing path, a malformed-JSON
    response makes PydanticOutputParser raise OutputParserException.

    That exception is a ValueError subclass but NOT a ValidationError, so before
    the fix it fell into the generic `except Exception` (non-retryable) and
    aborted the batch on attempt 1 — observed with GPT-5.5 on Bedrock emitting
    invalid JSON on a think-heavy batch. It must now retry like other transient
    output failures.
    """
    from langchain_core.exceptions import OutputParserException

    from codereview.providers.base import RetryConfig

    provider = _SequencedChainProvider(
        results=[
            OutputParserException("Invalid json output: {"),
            CodeReviewReport(
                summary="ok",
                metrics=ReviewMetrics(files_analyzed=1),
                issues=[],
                system_design_insights="No design issues found",
                recommendations=[],
                improvement_suggestions=[],
            ),
        ],
        enable_output_fixing=True,
    )
    cfg = RetryConfig(max_retries=3, validation_retry_sleep=0.0)

    result = provider._execute_with_retry(
        chain_input={"system_prompt": "x", "batch_context": "y"},
        retry_config=cfg,
        batch_context="y",
    )

    assert isinstance(result, CodeReviewReport)
    assert provider.invocations == 2  # retried once, then succeeded


def test_output_parser_exception_raises_after_retries_exhausted():
    """When every attempt yields malformed JSON, the parser error surfaces."""
    from langchain_core.exceptions import OutputParserException

    from codereview.providers.base import RetryConfig

    provider = _SequencedChainProvider(
        results=[
            OutputParserException("Invalid json output: {"),
            OutputParserException("Invalid json output: {"),
        ],
        enable_output_fixing=True,
    )
    cfg = RetryConfig(max_retries=1, validation_retry_sleep=0.0)

    with pytest.raises(OutputParserException):
        provider._execute_with_retry(
            chain_input={"system_prompt": "x", "batch_context": "y"},
            retry_config=cfg,
            batch_context="y",
        )
    assert provider.invocations == 2  # initial + 1 retry


def test_output_parser_exception_not_retried_when_output_fixing_disabled():
    """With output fixing off, a parser error raises on the first attempt."""
    from langchain_core.exceptions import OutputParserException

    from codereview.providers.base import RetryConfig

    provider = _SequencedChainProvider(
        results=[OutputParserException("Invalid json output: {")],
        enable_output_fixing=False,
    )
    cfg = RetryConfig(max_retries=3, validation_retry_sleep=0.0)

    with pytest.raises(OutputParserException):
        provider._execute_with_retry(
            chain_input={"system_prompt": "x", "batch_context": "y"},
            retry_config=cfg,
            batch_context="y",
        )
    assert provider.invocations == 1  # no retry


# ---------------------------------------------------------------------------
# Tool-calling path: a schema-violating result (ValidationError) is retried
# ---------------------------------------------------------------------------


def _schema_validation_error() -> Exception:
    """A real Pydantic ValidationError from a schema-violating CodeReviewReport.

    Simulates with_structured_output coercing a malformed tool call into the
    schema and failing — the tool-calling counterpart to OutputParserException.
    """
    from pydantic import ValidationError

    try:
        CodeReviewReport(summary="x", issues="not-a-list", metrics={})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")  # pragma: no cover


def test_tool_calling_validation_error_is_retried():
    """A ValidationError from the tool-calling chain retries end-to-end.

    Completes the both-paths malformed-output coverage: OutputParserException
    covers the prompt-parsing path; this covers schema violations on the
    tool-calling (with_structured_output) path surfacing into the retry loop.
    """
    from codereview.providers.base import RetryConfig

    provider = _SequencedChainProvider(
        results=[_schema_validation_error(), _good_raw()],
        enable_output_fixing=True,
    )
    cfg = RetryConfig(max_retries=3, validation_retry_sleep=0.0)

    result = provider._execute_with_retry(
        chain_input={"system_prompt": "x", "batch_context": "y"},
        retry_config=cfg,
        batch_context="y",
    )

    assert isinstance(result, CodeReviewReport)
    assert provider.invocations == 2  # retried once, then succeeded


def test_tool_calling_validation_error_not_retried_when_fixing_disabled():
    """With output fixing off, the schema ValidationError raises immediately."""
    from pydantic import ValidationError

    from codereview.providers.base import RetryConfig

    provider = _SequencedChainProvider(
        results=[_schema_validation_error(), _good_raw()],
        enable_output_fixing=False,
    )
    cfg = RetryConfig(max_retries=3, validation_retry_sleep=0.0)

    with pytest.raises(ValidationError):
        provider._execute_with_retry(
            chain_input={"system_prompt": "x", "batch_context": "y"},
            retry_config=cfg,
            batch_context="y",
        )
    assert provider.invocations == 1  # no retry
