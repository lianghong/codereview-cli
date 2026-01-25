"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from codereview.models import CodeReviewReport


@dataclass
class ValidationResult:
    """Result of credential/configuration validation.

    Attributes:
        valid: Whether validation passed
        provider: Provider name (e.g., "AWS Bedrock", "Azure OpenAI")
        checks: List of individual check results
        errors: List of error messages
        warnings: List of warning messages
        suggestions: List of fix suggestions
    """

    valid: bool
    provider: str
    checks: list[tuple[str, bool, str]] = field(
        default_factory=list
    )  # (name, passed, message)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, message: str = "") -> None:
        """Add a check result."""
        self.checks.append((name, passed, message))
        if not passed and message:
            self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_suggestion(self, message: str) -> None:
        """Add a fix suggestion."""
        self.suggestions.append(message)


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

    def _prepare_batch_context(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
    ) -> str:
        """Prepare context string for LLM.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents

        Returns:
            Formatted context string with file contents and line numbers
        """
        lines = [
            f"Analyzing Batch {batch_number}/{total_batches}",
            f"Files in this batch: {len(files_content)}",
            "",
            "=" * 80,
            "",
        ]

        for file_path, content in files_content.items():
            lines.append(f"File: {file_path}")
            lines.append("-" * 80)

            # Add line numbers
            for i, line in enumerate(content.splitlines(), start=1):
                lines.append(f"{i:4d} | {line}")

            lines.append("")
            lines.append("=" * 80)
            lines.append("")

        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token heuristic).

        Args:
            text: Text to estimate token count for

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def validate_credentials(self) -> ValidationResult:
        """Validate credentials and configuration before making API calls.

        Performs lightweight checks to verify:
        - Required credentials are present
        - Configuration is valid
        - Provider is accessible (optional, may make lightweight API call)

        Returns:
            ValidationResult with check details and any errors/suggestions

        Optional to override. Default returns valid result.
        """
        return ValidationResult(valid=True, provider="Unknown")
