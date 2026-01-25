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
