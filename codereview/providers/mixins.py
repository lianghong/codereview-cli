"""Provider mixins for shared functionality."""

import threading
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from codereview.config.models import ModelConfig


def require_https(url: str, label: str) -> str:
    """Return ``url`` if it uses HTTPS, else raise ValueError (fail closed).

    Called at client construction (``_create_model``) so a provider used
    directly — without first calling ``validate_credentials`` — still cannot
    send an API key / bearer token to a cleartext ``http://`` endpoint (CWE-319).
    ``label`` names the config field for the error message (e.g. "base_url").
    """
    if not str(url).lower().startswith("https://"):
        raise ValueError(f"{label} must use HTTPS, got: {url!r}")
    return str(url)


# Generic placeholder strings common to provider docs and READMEs. Each
# provider passes its README-documented export string(s) as ``extra`` so the
# exact copy-paste fails fast at --validate instead of 401'ing later.
_GENERIC_PLACEHOLDER_KEYS = frozenset(
    {
        "placeholder",
        "your-api-key",
        "your-api-key-here",
    }
)


def is_placeholder_api_key(api_key: str, extra: tuple[str, ...] = ()) -> bool:
    """Return True when ``api_key`` is a documentation placeholder.

    CLAUDE.md contract: the placeholder set must include the exact strings the
    README tells users to export, matched case-insensitively after ``strip()``.
    ``extra`` carries the provider-specific README strings (e.g.
    ``"your-deepseek-key"``); the generic set lives here so every provider
    rejects the common ones without re-declaring them.
    """
    normalized = api_key.strip().lower()
    return normalized in _GENERIC_PLACEHOLDER_KEYS or normalized in {
        e.lower() for e in extra
    }


def extract_openai_token_usage(result: Any) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an OpenAI-shaped result.

    Shared by every provider on the OpenAI client (Azure, DeepSeek, Moonshot,
    Z.AI, OpenAI-on-Bedrock), which all surface usage under
    ``response_metadata.token_usage``. Returns ``(0, 0)`` when metadata is
    absent so callers fall back to estimation.
    """
    if hasattr(result, "response_metadata"):
        token_usage = result.response_metadata.get("token_usage", {})
        return (
            token_usage.get("prompt_tokens", 0),
            token_usage.get("completion_tokens", 0),
        )
    return (0, 0)


def parse_retry_after(error: Exception, max_wait: float) -> float | None:
    """Return the Retry-After wait (seconds, capped at ``max_wait``) or None.

    Reads the ``retry-after`` header from a rate-limit error's response. Returns
    ``None`` when the error has no usable header, so each provider keeps its own
    exponential-backoff fallback (and its own base-wait policy) — Azure, for
    example, uses a longer fixed base than the OpenAI-compat default.
    """
    response = getattr(error, "response", None)
    if isinstance(error, RateLimitError) and response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                wait = float(retry_after)
            # PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
            except ValueError, TypeError:
                return None
            # A malformed/proxy Retry-After (e.g. "-1") must not become
            # time.sleep(-1) → ValueError. Treat negatives as "no usable
            # header" so the caller falls back to exponential backoff.
            if wait < 0:
                return None
            return min(wait, max_wait)
    return None


def is_openai_retryable_error(error: Exception) -> bool:
    """Return True for transient errors worth retrying on OpenAI-compatible APIs.

    Shared by every provider built on the OpenAI client (Azure, DeepSeek,
    Moonshot, Z.AI, and OpenAI-on-Bedrock), which all surface the same
    exception types. Retries:

    - ``RateLimitError`` (HTTP 429) — also an ``APIStatusError``, handled first.
    - ``APITimeoutError`` / ``APIConnectionError`` — network timeouts, resets,
      DNS/TLS failures. (``APITimeoutError`` subclasses ``APIConnectionError``.)
    - ``APIStatusError`` with a 5xx status — transient server-side failures.

    A 4xx ``APIStatusError`` other than 429 (e.g. 400/401/404) is NOT retried —
    those indicate a request/credential problem that a retry won't fix.
    """
    if isinstance(error, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIStatusError):
        return 500 <= error.status_code < 600
    return False


class TokenTrackingMixin:
    """Mixin providing token tracking and cost estimation.

    Provides standardized token counting, state management, and cost
    calculation for LLM providers. Token counter mutations are guarded
    by a lock so concurrent batch workers can safely increment totals.

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
    _token_lock: threading.Lock
    model_config: ModelConfig

    def _init_token_tracking(self) -> None:
        """Initialize token counters. Call in __init__."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._token_lock = threading.Lock()

    def _track_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add tokens to running totals.

        Args:
            input_tokens: Number of input tokens to add
            output_tokens: Number of output tokens to add
        """
        with self._token_lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

    def reset_state(self) -> None:
        """Reset token counters for fresh run."""
        with self._token_lock:
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
