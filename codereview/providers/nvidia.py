"""NVIDIA NIM API provider implementation."""

import contextlib
import os
import time
import warnings
from collections.abc import Generator
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import SecretStr, ValidationError

# Import system prompt from config
from codereview.config import SYSTEM_PROMPT
from codereview.config.models import ModelConfig, NVIDIAConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider, ValidationResult

# Shared prompt template for consistent formatting
BATCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("human", "{batch_context}"),
    ]
)


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


class NVIDIAProvider(ModelProvider):
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
        """
        self.callbacks = callbacks or []
        self.enable_output_fixing = enable_output_fixing
        self._output_parser = PydanticOutputParser(pydantic_object=CodeReviewReport)
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

        # Token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

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
            # Configure for structured output
            return base_model.with_structured_output(CodeReviewReport)

    def _create_chain(self) -> Any:
        """Create LangChain chain with prompt template."""
        return BATCH_PROMPT_TEMPLATE | self.model

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

        # Use chain with prompt template for cleaner invocation
        chain_input = {
            "system_prompt": SYSTEM_PROMPT,
            "batch_context": batch_context,
        }

        last_error: httpx.HTTPStatusError | ValidationError | None = None
        for attempt in range(max_retries + 1):
            try:
                # Suppress warnings during invocation
                with suppress_nvidia_warnings():
                    result = self.chain.invoke(chain_input)

                # Handle None result (structured output parsing failed)
                if result is None:
                    raise ValidationError.from_exception_data(
                        "Model returned None - structured output parsing failed",
                        [],
                    )

                # Track token usage from NVIDIA response metadata
                input_tokens = 0
                output_tokens = 0

                if hasattr(result, "response_metadata"):
                    # NVIDIA may use different metadata formats
                    usage = result.response_metadata.get(
                        "usage", {}
                    ) or result.response_metadata.get("token_usage", {})
                    input_tokens = usage.get("prompt_tokens", 0) or usage.get(
                        "input_tokens", 0
                    )
                    output_tokens = usage.get("completion_tokens", 0) or usage.get(
                        "output_tokens", 0
                    )

                # Fallback to estimation if actual counts unavailable
                if input_tokens == 0:
                    input_tokens = self._estimate_tokens(batch_context)
                if output_tokens == 0:
                    output_tokens = self._estimate_tokens(str(result.model_dump_json()))

                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                return result

            except ValidationError as e:
                # Output parsing/validation failed
                last_error = e

                if self.enable_output_fixing and attempt < max_retries:
                    # Try again - LangChain structured output will retry
                    time.sleep(1)
                    continue

                raise

            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code

                # Retryable errors: 429 (rate limit), 502/503/504 (gateway errors)
                # These are often transient and may succeed on retry
                retryable_codes = {429, 502, 503, 504}
                if status_code in retryable_codes:
                    if attempt < max_retries:
                        # Exponential backoff: 2^attempt seconds
                        # For gateway timeouts (504), use longer initial wait
                        base_wait = 4 if status_code == 504 else 2
                        wait_time = base_wait**attempt
                        time.sleep(wait_time)
                        continue

                # For other errors or all retries exhausted, raise
                raise

        # If we exhausted all retries, last_error must be set
        # (loop only exits early on success via return)
        assert last_error is not None, "Retry loop exited without error or success"
        raise last_error

    def get_model_display_name(self) -> str:
        """Get human-readable model name."""
        return self.model_config.name

    def get_pricing(self) -> dict[str, float]:
        """Get pricing information for the model."""
        return {
            "input_price_per_million": self.model_config.pricing.input_per_million,
            "output_price_per_million": self.model_config.pricing.output_per_million,
        }

    def reset_state(self) -> None:
        """Reset token counters."""
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
