"""Abstract base class for LLM providers."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from codereview.models import CodeReviewReport


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior in provider batch analysis.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_wait: Base wait time in seconds for exponential backoff
        max_wait: Maximum wait time in seconds (backoff cap)
        validation_retry_sleep: Sleep time in seconds between validation error retries
    """

    max_retries: int = 3
    base_wait: float = 1.0
    max_wait: float = 60.0
    validation_retry_sleep: float = 1.0


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
        ...

    @abstractmethod
    def get_model_display_name(self) -> str:
        """Get human-readable model name.

        Returns:
            Display name like "Claude Opus 4.6" or "GPT-5.2 Codex"
        """
        ...

    @abstractmethod
    def get_pricing(self) -> dict[str, float]:
        """Get pricing information for the model.

        Returns:
            Dictionary with keys: input_price_per_million, output_price_per_million
        """
        ...

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
        project_context: str | None = None,
    ) -> str:
        """Prepare context string for LLM.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            project_context: Optional README content for project context

        Returns:
            Formatted context string with file contents and line numbers
        """
        lines = []

        # Add project context (README) if provided
        if project_context:
            lines.extend(
                [
                    "== PROJECT CONTEXT ==",
                    "The following is the project README for background context:",
                    "",
                    "--- README.md ---",
                    project_context,
                    "--- END README ---",
                    "",
                    "== CODE REVIEW ==",
                ]
            )

        lines.extend(
            [
                f"Analyzing Batch {batch_number}/{total_batches}",
                f"Files in this batch: {len(files_content)}",
                "",
                "=" * 80,
                "",
            ]
        )

        for file_path, content in files_content.items():
            lines.append(f"File: {file_path}")
            lines.append("-" * 80)

            # Add line numbers (use extend with generator for efficiency)
            lines.extend(
                f"{i:4d} | {line}"
                for i, line in enumerate(content.splitlines(), start=1)
            )

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

    # --- Retry framework ---
    # Subclasses override these hooks for provider-specific behavior.

    def _track_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Track token usage from a provider response.

        Override in subclasses (via TokenTrackingMixin) for actual tracking.
        Default implementation is a no-op.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        pass

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable (e.g., rate limit, gateway timeout).

        Override in subclasses for provider-specific retryable error detection.

        Args:
            error: The exception to check

        Returns:
            True if the error should trigger a retry
        """
        return False

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Calculate backoff wait time for a retry attempt.

        Override in subclasses for custom backoff strategies.

        Args:
            error: The exception that triggered the retry
            attempt: Current attempt number (0-based)
            config: Retry configuration

        Returns:
            Wait time in seconds
        """
        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract (input_tokens, output_tokens) from provider response.

        Override in subclasses for provider-specific metadata extraction.

        Args:
            result: The result from chain invocation

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        return (0, 0)

    def _invoke_chain(self, chain_input: dict[str, str]) -> Any:
        """Invoke the LLM chain. Override for provider-specific wrappers.

        Args:
            chain_input: Input dictionary for the chain

        Returns:
            Chain invocation result
        """
        return self.chain.invoke(chain_input)  # type: ignore[attr-defined]

    def _execute_with_retry(
        self,
        chain_input: dict[str, str],
        retry_config: RetryConfig,
        batch_context: str,
    ) -> CodeReviewReport:
        """Execute chain invocation with retry logic, token tracking, and validation.

        This is the shared retry loop used by all providers. Provider-specific
        behavior is injected via hook methods:
        - _invoke_chain(): wraps chain invocation (e.g., warning suppression)
        - _is_retryable_error(): identifies retryable exceptions
        - _calculate_backoff(): computes wait time per attempt
        - _extract_token_usage(): extracts tokens from response metadata

        Args:
            chain_input: Input dictionary for the chain
            retry_config: Retry configuration
            batch_context: Raw batch context string for token estimation fallback

        Returns:
            CodeReviewReport with findings

        Raises:
            ValidationError: If structured output parsing fails after all retries
            Exception: Provider-specific errors after all retries exhausted
        """
        last_error: Exception | None = None
        for attempt in range(retry_config.max_retries + 1):
            try:
                result = self._invoke_chain(chain_input)

                # Handle None result (structured output parsing failed)
                if result is None:
                    raise ValidationError.from_exception_data(  # type: ignore[attr-defined]
                        "Model returned None - structured output parsing failed",
                        [],
                    )

                # Extract token usage from response metadata
                input_tokens, output_tokens = self._extract_token_usage(result)

                # Fallback to estimation if actual counts unavailable
                if input_tokens == 0:
                    input_tokens = self._estimate_tokens(batch_context)
                if output_tokens == 0:
                    output_tokens = self._estimate_tokens(str(result.model_dump_json()))

                self._track_tokens(input_tokens, output_tokens)

                return result

            except ValidationError as e:
                # Output parsing/validation failed
                last_error = e

                enable_fixing = getattr(self, "enable_output_fixing", False)
                if enable_fixing and attempt < retry_config.max_retries:
                    time.sleep(retry_config.validation_retry_sleep)
                    continue

                raise

            except Exception as e:
                last_error = e

                if self._is_retryable_error(e) and attempt < retry_config.max_retries:
                    wait = self._calculate_backoff(e, attempt, retry_config)
                    time.sleep(wait)
                    continue

                raise

        # If we exhausted all retries, last_error must be set
        # (loop only exits early on success via return)
        if last_error is None:
            raise RuntimeError("Retry loop exited without error or success")
        raise last_error
