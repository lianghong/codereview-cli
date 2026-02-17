"""NVIDIA NIM API provider implementation."""

import contextlib
import os
import warnings
from collections.abc import Generator
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import SecretStr

# Import system prompt from config
from codereview.config import SYSTEM_PROMPT
from codereview.config.models import ModelConfig, NVIDIAConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    BATCH_PROMPT_TEMPLATE,
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import TokenTrackingMixin


@contextlib.contextmanager
def suppress_nvidia_warnings() -> Generator[None, None, None]:
    """Context manager to suppress known NVIDIA langchain warnings.

    Suppresses warnings about:
    - Non-standard parameters (timeout, chat_template_kwargs)
    - Unknown model types
    - Structured output support
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*timeout is not default parameter.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*chat_template_kwargs is not default parameter.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*available_models.*type is unknown.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*not known to support structured output.*",
            category=UserWarning,
        )
        yield


class NVIDIAProvider(TokenTrackingMixin, ModelProvider):
    """NVIDIA NIM API implementation of ModelProvider."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: NVIDIAConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        """Initialize NVIDIA provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: NVIDIA-specific configuration (API key, base URL)
            temperature: Override temperature (uses model default if None)
            requests_per_second: Rate limit for API calls (default: 1.0)
            callbacks: Optional list of callback handlers for streaming/progress
            enable_output_fixing: Enable automatic retry on malformed output (default: True)
            project_context: Optional project README/documentation content for context
        """
        self.callbacks = callbacks or []
        self.enable_output_fixing = enable_output_fixing
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # Determine temperature (override > model default > 0.15)
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(
                    f"Temperature must be between 0.0 and 2.0, got {temperature}"
                )
            self.temperature = temperature
        elif (
            model_config.inference_params
            and model_config.inference_params.temperature is not None
        ):
            self.temperature = model_config.inference_params.temperature
        else:
            self.temperature = 0.15

        # Get model-specific inference parameters
        self.top_p = None
        self.max_tokens = 8192  # Default for NVIDIA models

        if model_config.inference_params:
            self.top_p = model_config.inference_params.top_p
            if model_config.inference_params.max_output_tokens:
                self.max_tokens = model_config.inference_params.max_output_tokens

        # Token tracking (from mixin)
        self._init_token_tracking()

        # Rate limiter for API calls
        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

        # Create LangChain model and chain
        with suppress_nvidia_warnings():
            self.model = self._create_model()
            self.chain = self._create_chain()

    def _create_model(self) -> Any:
        """Create LangChain NVIDIA model with structured output."""
        # Ensure full_id is present for NVIDIA models
        if not self.model_config.full_id:
            raise ValueError(
                f"NVIDIA model {self.model_config.id} missing required full_id"
            )

        # Build model parameters
        # Note: ChatNVIDIA passes kwargs to _NVIDIAClient, including 'timeout' for 202 polling.
        # The 'timeout' parameter controls how long to wait for async responses (HTTP 202).
        model_params: dict[str, Any] = {
            "model": self.model_config.full_id,
            "api_key": SecretStr(str(self.provider_config.api_key)),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "callbacks": self.callbacks if self.callbacks else None,
            "timeout": self.provider_config.polling_timeout,  # 202 polling timeout
        }

        # Add optional parameters
        if self.top_p is not None:
            model_params["top_p"] = self.top_p

        # Add base URL for self-hosted NIMs
        if self.provider_config.base_url:
            model_params["base_url"] = self.provider_config.base_url

        # Add thinking mode parameters (for models like GLM-4.7 that support it)
        if self.model_config.inference_params:
            chat_template_kwargs: dict[str, Any] = {}
            if self.model_config.inference_params.enable_thinking is not None:
                chat_template_kwargs["enable_thinking"] = (
                    self.model_config.inference_params.enable_thinking
                )
            if self.model_config.inference_params.clear_thinking is not None:
                chat_template_kwargs["clear_thinking"] = (
                    self.model_config.inference_params.clear_thinking
                )
            if chat_template_kwargs:
                model_params["chat_template_kwargs"] = chat_template_kwargs

        # Suppress warnings about unknown model types and parameters from langchain-nvidia
        with suppress_nvidia_warnings():
            base_model = ChatNVIDIA(**model_params)
            # Try include_raw=True first (returns {"raw": AIMessage, "parsed": CodeReviewReport}
            # so we can extract actual token counts from the raw AIMessage).
            # Some models (e.g., GLM-5) don't support include_raw — fall back gracefully.
            try:
                return base_model.with_structured_output(
                    CodeReviewReport, include_raw=True
                )
            except NotImplementedError:
                return base_model.with_structured_output(CodeReviewReport)

    def _create_chain(self) -> Any:
        """Create LangChain chain with prompt template."""
        return BATCH_PROMPT_TEMPLATE | self.model

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is a retryable NVIDIA gateway/rate limit error."""
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in {429, 502, 503, 504}
        return False

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Calculate adaptive backoff: longer for 504 gateway timeouts."""
        if isinstance(error, httpx.HTTPStatusError):
            # For gateway timeouts (504), use longer initial wait (4, 8, 16, 32...)
            # For rate limits (429) and other gateway errors, use (2, 4, 8, 16...)
            base = 4.0 if error.response.status_code == 504 else 2.0
            return min(base * (2**attempt), config.max_wait)
        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from NVIDIA response metadata (dual format fallback)."""
        if hasattr(result, "response_metadata"):
            # NVIDIA may use different metadata formats
            usage = result.response_metadata.get(
                "usage", {}
            ) or result.response_metadata.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0) or usage.get(
                "output_tokens", 0
            )
            return (input_tokens, output_tokens)
        return (0, 0)

    def _invoke_chain(self, chain_input: dict[str, str]) -> Any:
        """Invoke chain with NVIDIA warning suppression."""
        with suppress_nvidia_warnings():
            return self.chain.invoke(chain_input)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int | None = None,
    ) -> CodeReviewReport:
        """Analyze a batch of files using NVIDIA NIM API.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            max_retries: Maximum number of retries for gateway errors (uses config default if None)

        Returns:
            CodeReviewReport with findings

        Raises:
            httpx.HTTPStatusError: If NVIDIA API gateway errors persist after all retries
        """
        # Use configured max_retries if not overridden
        if max_retries is None:
            max_retries = self.provider_config.max_retries

        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content, self.project_context
        )

        chain_input = {
            "system_prompt": SYSTEM_PROMPT,
            "batch_context": batch_context,
        }

        retry_config = RetryConfig(max_retries=max_retries, base_wait=2.0)
        return self._execute_with_retry(chain_input, retry_config, batch_context)

    def get_model_display_name(self) -> str:
        """Get human-readable model name."""
        return self.model_config.name

    def get_pricing(self) -> dict[str, float]:
        """Get pricing information for the model."""
        return {
            "input_price_per_million": self.model_config.pricing.input_per_million,
            "output_price_per_million": self.model_config.pricing.output_per_million,
        }

    def validate_credentials(self) -> ValidationResult:
        """Validate NVIDIA API credentials and configuration.

        Checks:
        1. API key is configured (not empty/placeholder)
        2. API key format looks valid (starts with nvapi-)
        3. Model ID is set

        Returns:
            ValidationResult with check details
        """
        result = ValidationResult(valid=True, provider="NVIDIA NIM")

        # Check 1: API key configured
        api_key = self.provider_config.api_key
        if not api_key or api_key in ("", "your-api-key-here", "placeholder"):
            result.valid = False
            result.add_check(
                "API Key",
                False,
                "NVIDIA API key not configured",
            )
            result.add_suggestion("Set NVIDIA_API_KEY environment variable")
            result.add_suggestion(
                "Get your API key from https://build.nvidia.com/explore/discover"
            )
            return result

        # Check 2: API key format (NVIDIA keys typically start with nvapi-)
        if not api_key.startswith("nvapi-"):
            result.add_warning(
                "API key doesn't start with 'nvapi-'. "
                "Verify it's a valid NVIDIA API key from build.nvidia.com"
            )

        result.add_check("API Key", True, "API key configured")

        # Check 3: Model ID
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

        # Check 4: Base URL (informational)
        if self.provider_config.base_url:
            result.add_check(
                "Base URL",
                True,
                f"Using custom endpoint: {self.provider_config.base_url}",
            )
        else:
            result.add_check(
                "Base URL",
                True,
                "Using NVIDIA cloud API (build.nvidia.com)",
            )

        # Check 5: Optional connection test
        env_skip_test = os.environ.get("CODEREVIEW_SKIP_CONNECTION_TEST", "").lower()
        if env_skip_test not in ("1", "true", "yes"):
            try:
                # Quick request to check API is reachable
                base_url = (
                    self.provider_config.base_url
                    or "https://integrate.api.nvidia.com/v1"
                )
                test_url = f"{base_url}/models"

                with httpx.Client(timeout=5.0) as client:
                    response = client.get(
                        test_url,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if response.status_code == 200:
                        result.add_check(
                            "Connection",
                            True,
                            "API endpoint is reachable and authenticated",
                        )
                    elif response.status_code in (401, 403):
                        result.add_warning(
                            "API key may be invalid or expired. "
                            "Check your key at https://build.nvidia.com"
                        )
                    else:
                        result.add_check(
                            "Connection",
                            True,
                            f"API responded (status: {response.status_code})",
                        )

            except httpx.TimeoutException:
                result.add_warning("Connection test timed out. API may be slow.")
            except Exception as e:
                result.add_warning(f"Connection test failed: {e}")
                result.add_suggestion("Verify network connectivity to NVIDIA API")

        return result
