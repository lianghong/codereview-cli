"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod

from codereview.models import CodeReviewReport


class ModelProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """Analyze a batch of files.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            max_retries: Maximum number of retries for rate limiting

        Returns:
            CodeReviewReport with findings
        """
        pass

    @abstractmethod
    def get_model_display_name(self) -> str:
        """Get human-readable model name.

        Returns:
            Display name like "Claude Opus 4.5" or "GPT-5.2 Codex"
        """
        pass

    @abstractmethod
    def get_pricing(self) -> dict[str, float]:
        """Get pricing information for the model.

        Returns:
            Dictionary with keys: input_price_per_million, output_price_per_million
        """
        pass

    def reset_state(self) -> None:
        """Reset token counters and state for fresh run.

        Optional to override. Default implementation does nothing.
        """
        pass

    def estimate_cost(self) -> dict[str, float]:
        """Calculate cost from token usage.

        Returns:
            Dictionary with keys: input_tokens, output_tokens, input_cost,
            output_cost, total_cost

        Optional to override. Default returns zeros.
        """
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens used.

        Optional to override. Default returns 0.
        """
        return 0

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens used.

        Optional to override. Default returns 0.
        """
        return 0
