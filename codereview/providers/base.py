"""Abstract base class for LLM providers."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.exceptions import ContextOverflowError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import ValidationError

from codereview.models import CodeReviewReport

# Shared prompt template used by all providers
BATCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("human", "{batch_context}"),
    ]
)


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

    def get_model_display_name(self) -> str:
        """Get human-readable model name.

        Default reads ``self.model_config.name``. Providers without a
        ``model_config`` attribute must override.

        Returns:
            Display name like "Claude Opus 4.6" or "GPT-5.3 Codex"
        """
        model_config = getattr(self, "model_config", None)
        if model_config is None:
            raise NotImplementedError(
                "Provider has no model_config; override get_model_display_name()"
            )
        return str(model_config.name)

    def get_pricing(self) -> dict[str, float]:
        """Get pricing information for the model.

        Default reads ``self.model_config.pricing``. Providers without a
        ``model_config`` attribute must override.

        Returns:
            Dictionary with keys: input_price_per_million, output_price_per_million
        """
        model_config = getattr(self, "model_config", None)
        if model_config is None:
            raise NotImplementedError(
                "Provider has no model_config; override get_pricing()"
            )
        return {
            "input_price_per_million": model_config.pricing.input_per_million,
            "output_price_per_million": model_config.pricing.output_per_million,
        }

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

    @staticmethod
    def _resolve_temperature(
        override: float | None,
        model_config: Any,
        provider_default: float,
        allow_none: bool = False,
    ) -> float | None:
        """Resolve the effective temperature for a provider.

        Precedence: explicit ``override`` > ``model_config.inference_params.temperature``
        > ``provider_default``. When ``allow_none`` is True, a model that
        explicitly sets ``temperature=None`` (reasoning models like Claude
        Opus 4.7) stays None.

        Args:
            override: Caller-supplied temperature (usually from CLI), or None
            model_config: ModelConfig with optional ``inference_params``
            provider_default: Fallback when no other value is set
            allow_none: If True, preserves an explicit None from inference_params

        Returns:
            Effective temperature, or None for reasoning models when allow_none

        Raises:
            ValueError: If override is outside [0.0, 2.0]
        """
        if override is not None:
            if not 0.0 <= override <= 2.0:
                raise ValueError(
                    f"Temperature must be between 0.0 and 2.0, got {override}"
                )
            return override

        params = getattr(model_config, "inference_params", None)
        if params is not None:
            if params.temperature is not None:
                return float(params.temperature)
            if allow_none:
                # Reasoning models explicitly opt out of temperature
                return None
        return provider_default

    @staticmethod
    def _build_rate_limiter(requests_per_second: float) -> InMemoryRateLimiter:
        """Construct the shared InMemoryRateLimiter used by every provider."""
        return InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

    def _system_prompt_with_format_instructions(self, system_prompt: str) -> str:
        """Append Pydantic format instructions when using prompt-based parsing.

        Models without tool-use support (e.g. DeepSeek-R1) need the JSON schema
        embedded in the system prompt. Providers that never set
        ``_use_prompt_parsing`` get the prompt back unchanged.
        """
        if getattr(self, "_use_prompt_parsing", False):
            parser = getattr(self, "_output_parser", None)
            if parser is not None:
                return f"{system_prompt}\n\n{parser.get_format_instructions()}"
        return system_prompt

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
        """Extract (input_tokens, output_tokens) from a provider response.

        Override in subclasses to extract actual token counts from
        provider-specific response metadata.

        Args:
            result: The provider response object

        Returns:
            Tuple of (input_tokens, output_tokens), or (0, 0) if unavailable
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

        Handles two result formats:
        - dict from include_raw=True: {"raw": AIMessage, "parsed": CodeReviewReport}
        - CodeReviewReport directly from prompt-parsing path (e.g., DeepSeek-R1)

        Args:
            chain_input: Input dictionary for the chain
            retry_config: Retry configuration
            batch_context: Raw batch context string for token estimation fallback

        Returns:
            CodeReviewReport with findings

        Raises:
            ValueError: If input exceeds model context window (ContextOverflowError)
            ValidationError: If structured output parsing fails after all retries
            Exception: Provider-specific errors after all retries exhausted
        """
        last_error: Exception | None = None
        for attempt in range(retry_config.max_retries + 1):
            try:
                result = self._invoke_chain(chain_input)

                # Handle include_raw=True dict format:
                # {"raw": AIMessage, "parsed": CodeReviewReport, "parsing_error": ...}
                if isinstance(result, dict):
                    raw = result.get("raw")
                    parsed = result.get("parsed")
                    parsing_error = result.get("parsing_error")

                    if parsed is None:
                        msg = "Structured output parsing failed"
                        if parsing_error:
                            msg = f"{msg}: {parsing_error}"
                        raise ValueError(msg)

                    # Extract token usage from the raw AIMessage
                    input_tokens, output_tokens = self._extract_token_usage(raw)

                    # Fallback to estimation if actual counts unavailable
                    if input_tokens == 0:
                        input_tokens = self._estimate_tokens(batch_context)
                    if output_tokens == 0:
                        output_tokens = self._estimate_tokens(
                            str(parsed.model_dump_json())
                        )

                    self._track_tokens(input_tokens, output_tokens)
                    return parsed

                # Handle direct CodeReviewReport (prompt-parsing path)
                if result is None:
                    raise ValueError(
                        "Model returned None - structured output parsing failed"
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

            except ContextOverflowError as e:
                # Input exceeds model context window — retrying won't help
                raise ValueError(
                    f"Input exceeds model context window ({e}). "
                    "Try reducing --batch-size."
                ) from e

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
