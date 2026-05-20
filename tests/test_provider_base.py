import pytest

from codereview.models import CodeReviewReport, ReviewMetrics
from codereview.providers.base import ModelProvider


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
