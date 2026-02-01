"""Provider mixins for shared functionality."""

from codereview.config.models import ModelConfig


class TokenTrackingMixin:
    """Mixin providing token tracking and cost estimation.

    Provides standardized token counting, state management, and cost
    calculation for LLM providers.

    Requirements:
        Classes using this mixin must have:
        - self.model_config: ModelConfig with pricing info

    Usage:
        class MyProvider(TokenTrackingMixin, ModelProvider):
            def __init__(self, model_config, ...):
                self.model_config = model_config
                self._init_token_tracking()

            def analyze_batch(self, ...):
                ...
                self._track_tokens(input_tokens, output_tokens)
    """

    _total_input_tokens: int
    _total_output_tokens: int
    model_config: ModelConfig

    def _init_token_tracking(self) -> None:
        """Initialize token counters. Call in __init__."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _track_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add tokens to running totals.

        Args:
            input_tokens: Number of input tokens to add
            output_tokens: Number of output tokens to add
        """
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

    def reset_state(self) -> None:
        """Reset token counters for fresh run."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens used."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens used."""
        return self._total_output_tokens

    def estimate_cost(self) -> dict[str, float]:
        """Calculate cost from token usage.

        Returns:
            Dict with keys:
                - input_tokens: Total input tokens used
                - output_tokens: Total output tokens used
                - input_cost: Cost for input tokens in USD
                - output_cost: Cost for output tokens in USD
                - total_cost: Combined cost in USD
        """
        pricing = self.model_config.pricing

        input_cost = (self._total_input_tokens / 1_000_000) * pricing.input_per_million
        output_cost = (
            self._total_output_tokens / 1_000_000
        ) * pricing.output_per_million

        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }
