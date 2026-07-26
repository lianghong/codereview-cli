"""Google Generative AI (Gemini) provider implementation."""

import re
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from codereview.config.models import GoogleGenAIConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import (
    TRANSPORT_TRANSIENT_ERRORS,
    TokenTrackingMixin,
    is_blank,
    is_placeholder_api_key,
    is_short_api_key,
)

# Status codes worth another attempt: quota exhaustion and transient server-side
# failures. Deliberately excludes 400/401/403/404 — a retry can't fix a bad
# request or a bad key, it just makes the failure slower.
#
# 429 and 503 are the status equivalents of the ResourceExhausted /
# ServiceUnavailable classes this provider used to test for, so those two
# restore the intended policy exactly. 500 and 504 are added deliberately: the
# google-genai SDK raises them as plain ServerError with no api_core analogue,
# they are the server admitting the fault is its own, and Gemini returns them on
# overload. Locked by the status-code table in tests/test_retry_contract.py.
_RETRYABLE_GOOGLE_STATUS_CODES = frozenset({429, 500, 503, 504})


class GoogleGenAIProvider(TokenTrackingMixin, ModelProvider):
    """Google Generative AI (Gemini) implementation of ModelProvider."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: GoogleGenAIConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        """Initialize Google GenAI provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: Google GenAI-specific configuration (API key)
            temperature: Override temperature (uses model default if None)
            requests_per_second: Rate limit for API calls (default: 1.0)
            callbacks: Optional list of callback handlers for streaming/progress
            enable_output_fixing: Enable automatic retry on malformed output (default: True)
            project_context: Optional project README/documentation content
        """
        self.callbacks = callbacks or []
        self.enable_output_fixing = enable_output_fixing
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # allow_none lets a Gemini reasoning model opt out of temperature
        # (inference_params.temperature = None), matching the other providers.
        self.temperature = self._resolve_temperature(
            override=temperature,
            model_config=model_config,
            provider_default=0.15,
            allow_none=True,
        )

        # Get model-specific inference parameters
        self.top_p: float | None = None
        self.top_k: int | None = None
        self.max_tokens = 65536  # Default for Gemini models

        if model_config.inference_params:
            self.top_p = model_config.inference_params.top_p
            self.top_k = model_config.inference_params.top_k
            if model_config.inference_params.max_output_tokens:
                self.max_tokens = model_config.inference_params.max_output_tokens

        # Token tracking (from mixin)
        self._init_token_tracking()

        # Rate limiter for API calls
        self.rate_limiter = self._build_rate_limiter(requests_per_second)

        # Create LangChain model and chain
        self.model = self._create_model()
        self.chain = self._create_chain()

    def _create_model(self) -> Any:
        """Create LangChain Google GenAI model with structured output."""
        if not self.model_config.full_id:
            raise ValueError(
                f"Google GenAI model {self.model_config.id} missing required full_id"
            )

        # Build model parameters
        model_params: dict[str, Any] = {
            "model": self.model_config.full_id,
            "google_api_key": SecretStr(str(self.provider_config.api_key)),
            "max_output_tokens": self.max_tokens,
            "timeout": self.provider_config.request_timeout,
            "callbacks": self.callbacks if self.callbacks else None,
            "rate_limiter": self.rate_limiter,
        }

        # Omit temperature for reasoning models that opt out (temperature=None)
        if self.temperature is not None:
            model_params["temperature"] = self.temperature

        # Add optional parameters
        if self.top_p is not None:
            model_params["top_p"] = self.top_p
        if self.top_k is not None:
            model_params["top_k"] = self.top_k

        base_model = ChatGoogleGenerativeAI(**model_params)
        # Tool-use vs prompt-parsing routing (and _create_chain) live in the
        # base class; supports_tool_use in models.yaml decides the path.
        # Gemini's structured output wants method="json_schema".
        return self._apply_structured_output(base_model, method="json_schema")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is a retryable Google API error.

        Classified on the **status code**, not on an exception class hierarchy.
        ``langchain-google-genai`` 4.x runs on the ``google-genai`` SDK, which
        raises ``google.genai.errors.ClientError`` / ``ServerError`` carrying a
        ``.code``. It does *not* raise the ``google.api_core.exceptions`` types
        (``ResourceExhausted``/``ServiceUnavailable``) this used to test for —
        those belong to the older ``google-generativeai`` stack, so the
        isinstance check matched nothing and **every** 429 and 503 aborted the
        batch on attempt 1. ``api_core`` is still installed as a transitive
        dependency, which is why the dead branch stayed invisible: it imported
        fine and the tests constructed the exceptions by hand.

        Retryable: 429 (quota), 500/503/504 (transient server-side), plus the
        transport failures the SDK surfaces from httpx/requests. A 400/401/403/404
        is a request or credential problem that a retry cannot fix.
        """
        return self._google_status_code(error) in _RETRYABLE_GOOGLE_STATUS_CODES or (
            isinstance(error, TRANSPORT_TRANSIENT_ERRORS)
        )

    @staticmethod
    def _google_status_code(error: Exception) -> int | None:
        """Return the HTTP status carried by a google-genai error, if any.

        ``APIError.code`` is the documented attribute. The langchain wrapper also
        re-raises some failures as ``ChatGoogleGenerativeAIError`` with the status
        only in the message text (see the wrapper's own 429 handling example), so
        fall back to a leading-status scan of the string form.
        """
        code = getattr(error, "code", None)
        if isinstance(code, int):
            return code
        match = re.match(r"\s*(\d{3})\b", str(error))
        return int(match.group(1)) if match else None

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Calculate backoff: longer for rate limits (429) on preview models."""
        # Google preview models have strict rate limits — use longer base (10s)
        # giving 10, 20, 40, 60, 60 ... progression
        if self._google_status_code(error) == 429:
            return min(10.0 * (2**attempt), config.max_wait)
        return min(config.base_wait * (2**attempt), config.max_wait)

    @classmethod
    def supports_token_streaming(cls) -> bool:
        """False — ``streaming`` is deliberately not passed to the client.

        ``ChatGoogleGenerativeAI`` accepts ``streaming=True`` and its streaming
        path does accumulate usage, so this *could* be enabled — but the Gemini
        entry here runs structured output through ``method="json_schema"``, and
        whether that survives the streaming wire path is unproven against the
        live endpoint. This project's rule for exactly that situation is
        assume-not-until-a-live-run-proves-it (the same rule that governs
        ``supports_tool_use``), so ``--stream`` keeps its concurrency here
        instead of paying for a path that renders nothing.
        """
        return False

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from Google GenAI response metadata.

        langchain-google-genai >=4.x populates AIMessage.usage_metadata as
        a UsageMetadata dict with input_tokens/output_tokens keys.
        """
        usage = getattr(result, "usage_metadata", None)
        if isinstance(usage, dict):
            return (usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        return (0, 0)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int | None = None,
    ) -> CodeReviewReport:
        """Analyze a batch of files using Google Generative AI.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            max_retries: Maximum number of retries for API errors (None uses
                this provider's default of 5 — preview models throttle hard)

        Returns:
            CodeReviewReport with findings
        """
        retries = self._resolve_max_retries(max_retries, self.provider_config, 5)

        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content, self.project_context
        )

        chain_input = {
            "system_prompt": self._build_batch_system_prompt(files_content),
            "batch_context": batch_context,
        }

        retry_config = RetryConfig(max_retries=retries, base_wait=5.0)
        return self._execute_with_retry(chain_input, retry_config, batch_context)

    def validate_credentials(self) -> ValidationResult:
        """Validate Google API credentials and configuration.

        Checks:
        1. API key is configured (not empty/placeholder)
        2. Model ID is set

        Returns:
            ValidationResult with check details
        """
        result = ValidationResult(valid=True, provider="Google GenAI")

        # Check 1: API key configured
        api_key = self.provider_config.api_key
        # "your-api-key-here" (the exact README string) is in the generic set
        if is_blank(api_key) or is_placeholder_api_key(api_key):
            result.valid = False
            result.add_check(
                "API Key",
                False,
                "Google API key not configured",
            )
            result.add_suggestion("Set GOOGLE_API_KEY environment variable")
            result.add_suggestion(
                "Get your API key from https://aistudio.google.com/apikey"
            )
            return result

        # Every other key-taking provider emits this warning; Google was the
        # one omission, so a truncated GOOGLE_API_KEY reported all-green and
        # then 401'd on the first batch. Deliberately a warning, not a failure —
        # Google documents no minimum length.
        if is_short_api_key(api_key):
            result.add_warning("API key seems unusually short. Verify it's correct.")
        result.add_check("API Key", True, "API key configured")

        # Check 2: Model ID
        model_id = self.model_config.full_id
        if not model_id:
            result.valid = False
            result.add_check(
                "Model ID",
                False,
                "Model full_id not configured",
            )
            result.add_suggestion(
                f"Configure full_id for model '{self.model_config.id}' "
                "in config/models.yaml"
            )
            return result

        result.add_check("Model ID", True, f"Model: {model_id}")

        return result
