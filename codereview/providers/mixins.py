"""Provider mixins for shared functionality."""

import threading
from typing import Any
from urllib.parse import urlsplit

import httpx
import requests
from langchain_core.callbacks import BaseCallbackHandler
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from codereview.config.models import ModelConfig

# Transport-level failures that clear on their own: the connection never
# completed, so no request was processed and a retry is safe. Providers whose
# SDK does *not* wrap these in its own exception type (NVIDIA NIM on
# ``requests``, google-genai on ``httpx``/``requests``) must name them, or a
# single DNS blip or read timeout throws away a whole batch on attempt 1.
#
# Both libraries are already installed transitively — ``requests`` by the NVIDIA
# and google-genai clients, ``httpx`` by the openai client — so this adds no
# dependency. Timeout subclasses ConnectionError in neither library, hence both.
TRANSPORT_TRANSIENT_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)


def is_https_url(url: str) -> bool:
    """Return True when ``url`` is a well-formed HTTPS URL with a host.

    The single spelling of this test, shared by ``require_https`` (which fails
    closed at client construction) and every provider's ``validate_credentials``
    (which reports it as a check). They must agree: a URL the constructor
    accepts but ``--validate`` rejects makes the pre-flight check lie about a
    config that runs fine, and the reverse would let a cleartext endpoint pass
    validation.

    Scheme comparison is case-insensitive per RFC 3986 §3.1 — ``HTTPS://host``
    is a valid HTTPS URL, and the providers' plain ``startswith("https://")``
    used to reject it while the constructor let it through.

    A hostname is required, not just the scheme prefix. A bare ``"https://"``
    (or ``"https:///v1"``) satisfies a ``startswith`` test but names no server,
    so it passed ``--validate`` as a green check and then failed at client
    construction or on the first request — the pre-flight check reporting OK on
    a config that cannot work is the failure this prevents. Parsed with
    ``urlsplit`` rather than string surgery so bracketed IPv6 authorities
    (``https://[::1]/v1``) and ``user@host`` forms are read correctly.
    """
    try:
        parsed = urlsplit(str(url).strip())
    except ValueError:
        # urlsplit raises on a malformed IPv6 authority, e.g. "https://[::1".
        return False
    return parsed.scheme.lower() == "https" and bool(parsed.hostname)


def require_https(url: str, label: str) -> str:
    """Return the normalized HTTPS ``url``, else raise ValueError (fail closed).

    Called at client construction (``_create_model``) so a provider used
    directly — without first calling ``validate_credentials`` — still cannot
    send an API key / bearer token to a cleartext ``http://`` endpoint (CWE-319).
    ``label`` names the config field for the error message (e.g. "base_url").

    The return value is stripped, because :func:`is_https_url` strips before
    testing: returning the raw string handed a padded ``"  https://host  "``
    straight to the HTTP client, so the value validation accepted was not the
    value the client got. Whatever this returns is what was actually checked.
    """
    normalized = str(url).strip()
    if not is_https_url(normalized):
        raise ValueError(f"{label} must use HTTPS and name a host, got: {url!r}")
    return normalized


def is_blank(value: str | None) -> bool:
    """Return True when ``value`` is absent, empty, or only whitespace.

    The presence test for credentials and endpoints in ``validate_credentials``.
    A plain ``not value`` is not enough: Pydantic's ``min_length=1`` on every
    provider's ``api_key`` accepts ``"   "``, and a whitespace-only string is
    truthy — so the loader registered the provider and ``validate_credentials``
    reported *every* check as passing, deferring the failure to a 401 on the
    first real call. Stripping here applies the same normalization
    :func:`is_placeholder_api_key` does, so the presence check and the
    placeholder check can't disagree about what counts as a value.

    Accepts ``None`` for configs that type a field optional (NVIDIA's
    ``base_url``); the credential fields are all ``str``.
    """
    return not value or not value.strip()


# Shortest length any provider's real key reaches. Below this the value is
# probably a truncated paste rather than a credential — a warning, never a
# hard failure, since it's a heuristic and no provider documents a minimum.
MIN_PLAUSIBLE_API_KEY_LENGTH = 20


def is_short_api_key(api_key: str) -> bool:
    """Return True when ``api_key`` is too short to plausibly be a real key.

    One spelling of the threshold, previously an inline ``len(api_key) < 20`` in
    five providers. Strips first so surrounding whitespace doesn't pad a short
    key past the bar — the same normalization :func:`is_blank` and
    :func:`is_placeholder_api_key` apply.
    """
    return len(api_key.strip()) < MIN_PLAUSIBLE_API_KEY_LENGTH


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


def wants_token_streaming(callbacks: list[Any] | None) -> bool:
    """Return True only when a callback actually consumes streamed tokens.

    Providers used to pass ``streaming=bool(self.callbacks)``, but the two
    handlers this project ships are not equivalent: ``StreamingCallbackHandler``
    overrides ``on_llm_new_token`` and renders each chunk, while
    ``ProgressCallbackHandler`` (the ``--verbose`` handler) does not override it
    at all — it only reacts to start/end. So ``--verbose`` switched every
    OpenAI-compatible provider onto the streaming wire path to feed a handler
    that cannot observe a single token, and the switch is not free: see
    ``openai_stream_params`` for the token-accounting cost. Deciding on the
    handler's actual capability instead of on list-emptiness keeps ``--verbose``
    on the non-streaming path.

    Checked by attribute rather than by importing the concrete handler classes:
    ``codereview.callbacks`` imports Rich, this module is imported by every
    provider, and the real contract is LangChain's — any handler overriding
    ``on_llm_new_token`` wants tokens, including one a caller supplies.
    """
    if not callbacks:
        return False
    base = BaseCallbackHandler.on_llm_new_token
    return any(
        getattr(type(handler), "on_llm_new_token", base) is not base
        for handler in callbacks
    )


def openai_stream_params(callbacks: list[Any] | None) -> dict[str, Any]:
    """Build the ``streaming``/``stream_usage`` kwargs for an OpenAI-compat client.

    ``stream_usage`` is what makes the client send ``stream_options:
    {"include_usage": true}``, and **without it a streaming response carries no
    usage at all**: OpenAI-compatible servers omit the final usage chunk unless
    asked, so ``usage_metadata`` comes back ``None``, ``extract_openai_token_usage``
    returns ``(0, 0)``, and ``base.py`` silently falls back to its byte-heuristic
    estimate — which cannot see reasoning tokens. That is the same failure this
    project already measured on Azure's Responses API path (~13x under-report on
    a think-heavy batch), reached by a different route.

    langchain-openai auto-enables ``stream_usage`` only when no ``base_url`` is
    configured and ``OPENAI_BASE_URL`` is unset (it assumes a non-OpenAI endpoint
    may not support the option). Every provider here passes an explicit
    ``base_url``/``api_base``, so the auto-enable never fires and the flag has to
    be set by hand. ``AzureChatOpenAI`` is the exception — it keys off
    ``base_url`` being None, which is true for a deployment-routed client, so it
    already defaults to True; passing it again is harmless and keeps the five
    providers identical.

    Only meaningful on the streaming path: ``_stream`` is the sole place that
    turns ``stream_usage`` into ``stream_options``, and ``_get_request_payload``
    for a non-streaming call omits it, so the flag is inert when
    ``streaming`` is False.
    """
    streaming = wants_token_streaming(callbacks)
    params: dict[str, Any] = {"streaming": streaming}
    if streaming:
        params["stream_usage"] = True
    return params


def extract_openai_token_usage(result: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from an OpenAI-shaped result.

    Shared by every provider on the OpenAI client (Azure, DeepSeek, Moonshot,
    Z.AI, OpenAI-on-Bedrock). Returns ``(0, 0)`` when no usage is present at
    all, so callers fall back to estimation.

    ``AIMessage`` carries usage in **two independent places** and only one of
    them is populated on every path:

    - ``usage_metadata`` — LangChain's normalized
      ``input_tokens``/``output_tokens``, set by *both* the Chat Completions and
      the Responses API converters.
    - ``response_metadata["token_usage"]`` — the vendor's raw
      ``prompt_tokens``/``completion_tokens``, which **only** the Chat
      Completions converter copies through.

    Reading the raw dict first therefore returned ``(0, 0)`` for every
    ``use_responses_api: true`` model, and ``.get(..., 0)`` made that silent:
    ``base.py`` fell back to *estimating* the counts, and estimation cannot see
    reasoning tokens at all. Azure ``gpt-5.4``/``gpt-5.4-pro`` (Responses API,
    tool-use path, where real vendor counts *were* available) under-reported by
    ~13x on a think-heavy batch — 40,000 in / 9,000 out billed, 6,211 / 145
    recorded, $0.2350 of spend reported as $0.0177.

    Prefer the normalized field; keep the raw dict as the fallback so a
    hand-built or future response that carries only ``token_usage`` still
    reports real numbers.
    """
    usage = getattr(result, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        if input_tokens or output_tokens:
            return (input_tokens, output_tokens)

    metadata = getattr(result, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage") or {}
        if isinstance(token_usage, dict):
            return (
                token_usage.get("prompt_tokens", 0) or 0,
                token_usage.get("completion_tokens", 0) or 0,
            )
    return (0, 0)


def parse_retry_after(error: Exception, max_wait: float) -> float | None:
    """Return the Retry-After wait (seconds, capped at ``max_wait``) or None.

    Reads the ``retry-after`` header off the error's response. Returns ``None``
    when the error has no usable header, so each provider keeps its own
    exponential-backoff fallback (and its own base-wait policy) — Azure, for
    example, uses a longer fixed base than the OpenAI-compat default.

    Accepts any ``APIStatusError``, not just ``RateLimitError``. ``Retry-After``
    is defined for 503 (RFC 9110 §10.2.3) and every provider here already
    *retries* 5xx (:func:`is_openai_retryable_error`), so narrowing the header
    read to 429 meant a server that said "come back in 30s" during a capacity
    window got blind exponential backoff instead — the openai SDK's own
    ``_calculate_retry_timeout`` honours the header on every retryable status.
    The value is still bounded by ``max_wait``, so a hostile or broken header
    cannot stall a run.
    """
    response = getattr(error, "response", None)
    if isinstance(error, APIStatusError) and response is not None:
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
