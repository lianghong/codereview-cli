"""Azure OpenAI provider implementation."""

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import AzureChatOpenAI
from openai import RateLimitError
from pydantic import SecretStr, ValidationError

# Import system prompt from config
from codereview.config import SYSTEM_PROMPT
from codereview.config.models import AzureOpenAIConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider, ValidationResult

# Shared prompt template for consistent formatting
BATCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("human", "{batch_context}"),
    ]
)


class AzureOpenAIProvider(ModelProvider):
    """Azure OpenAI implementation of ModelProvider."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: AzureOpenAIConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
    ):
        """Initialize Azure OpenAI provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: Azure-specific configuration (endpoint, API key, etc.)
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

        # Determine temperature (override > model default > 0.0 for Azure)
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
            self.temperature = 0.0

        # Get model-specific inference parameters
        self.top_p = None
        self.max_tokens = 16000  # Default

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
        self.model = self._create_model()
        self.chain = self._create_chain()

    def _create_model(self) -> Any:
        """Create LangChain Azure OpenAI model with structured output."""
        # Build model kwargs
        model_kwargs: dict = {}
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p

        base_model = AzureChatOpenAI(
            azure_deployment=self.model_config.deployment_name,
            azure_endpoint=str(self.provider_config.endpoint),
            api_key=SecretStr(self.provider_config.api_key),  # type: ignore[arg-type]
            api_version=self.provider_config.api_version,
            temperature=self.temperature,
            rate_limiter=self.rate_limiter,
            callbacks=self.callbacks if self.callbacks else None,
            streaming=bool(self.callbacks),  # Enable streaming if callbacks provided
            model_kwargs={**model_kwargs, "max_tokens": self.max_tokens},
        )

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
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """Analyze a batch of files using Azure OpenAI.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            max_retries: Maximum number of retries for rate limiting

        Returns:
            CodeReviewReport with findings

        Raises:
            RateLimitError: If Azure API rate limit exceeded after all retries
        """
        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content
        )

        # Use chain with prompt template for cleaner invocation
        chain_input = {
            "system_prompt": SYSTEM_PROMPT,
            "batch_context": batch_context,
        }

        last_error: RateLimitError | ValidationError | None = None
        for attempt in range(max_retries + 1):
            try:
                result = self.chain.invoke(chain_input)

                # Track token usage from Azure OpenAI response metadata
                input_tokens = 0
                output_tokens = 0

                if hasattr(result, "response_metadata"):
                    token_usage = result.response_metadata.get("token_usage", {})
                    input_tokens = token_usage.get("prompt_tokens", 0)
                    output_tokens = token_usage.get("completion_tokens", 0)

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

            except RateLimitError as e:
                last_error = e

                # Rate limiter handles most cases, but we still need manual retry
                # for edge cases
                if attempt < max_retries:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue

                # For all retries exhausted, raise
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
        """Validate Azure OpenAI credentials and configuration.

        Checks:
        1. API key is configured (not empty/placeholder)
        2. Endpoint URL is valid
        3. Deployment name is set
        4. Optionally tests connection

        Returns:
            ValidationResult with check details
        """
        import os
        from urllib.parse import urlparse

        result = ValidationResult(valid=True, provider="Azure OpenAI")

        # Check 1: API key configured
        api_key = self.provider_config.api_key
        if not api_key or api_key in ("", "your-api-key-here", "placeholder"):
            result.valid = False
            result.add_check(
                "API Key",
                False,
                "Azure OpenAI API key not configured",
            )
            result.add_suggestion("Set AZURE_OPENAI_API_KEY environment variable")
            result.add_suggestion(
                "Or configure api_key in config/models.yaml under azure_openai section"
            )
            return result

        # Check if it looks like a placeholder
        if len(api_key) < 20:
            result.add_warning("API key seems unusually short. Verify it's correct.")

        result.add_check("API Key", True, "API key configured")

        # Check 2: Endpoint URL valid
        endpoint = str(self.provider_config.endpoint).rstrip("/")
        if not endpoint or endpoint in ("", "https://your-resource.openai.azure.com"):
            result.valid = False
            result.add_check(
                "Endpoint",
                False,
                "Azure OpenAI endpoint not configured",
            )
            result.add_suggestion("Set AZURE_OPENAI_ENDPOINT environment variable")
            result.add_suggestion(
                "Or configure endpoint in config/models.yaml under azure_openai section"
            )
            return result

        # Also check for placeholder patterns in URL
        if "your-resource" in endpoint or "your-endpoint" in endpoint:
            result.valid = False
            result.add_check(
                "Endpoint",
                False,
                "Azure OpenAI endpoint appears to be a placeholder",
            )
            result.add_suggestion(
                "Replace placeholder endpoint with your actual Azure OpenAI resource URL"
            )
            return result

        # Validate URL format
        try:
            parsed = urlparse(endpoint)
            if not parsed.scheme or not parsed.netloc:
                result.valid = False
                result.add_check(
                    "Endpoint",
                    False,
                    f"Invalid endpoint URL format: {endpoint}",
                )
                result.add_suggestion(
                    "Endpoint should be like: https://your-resource.openai.azure.com"
                )
                return result

            if not endpoint.startswith("https://"):
                result.add_warning("Endpoint should use HTTPS for security")

            result.add_check("Endpoint", True, f"Endpoint: {endpoint}")

        except Exception as e:
            result.valid = False
            result.add_check("Endpoint", False, f"Error parsing endpoint: {e}")
            return result

        # Check 3: Deployment name
        deployment = self.model_config.deployment_name
        if not deployment:
            result.valid = False
            result.add_check(
                "Deployment",
                False,
                "Deployment name not configured for model",
            )
            result.add_suggestion(
                f"Configure deployment_name for model '{self.model_config.id}' "
                "in config/models.yaml"
            )
            return result

        result.add_check("Deployment", True, f"Deployment: {deployment}")

        # Check 4: API version
        api_version = self.provider_config.api_version
        if api_version:
            result.add_check("API Version", True, f"Version: {api_version}")
        else:
            result.add_warning("No API version specified, using default")

        # Check 5: Optional connection test (lightweight)
        env_skip_test = os.environ.get("CODEREVIEW_SKIP_CONNECTION_TEST", "").lower()
        if env_skip_test not in ("1", "true", "yes"):
            try:
                import httpx

                # Quick HEAD request to check endpoint is reachable
                test_url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"
                with httpx.Client(timeout=5.0) as client:
                    response = client.head(
                        test_url,
                        headers={"api-key": api_key},
                    )
                    # 401/403 means endpoint exists but we'd need proper auth
                    # 404 might mean deployment doesn't exist
                    # Any response means endpoint is reachable
                    if response.status_code == 404:
                        result.add_warning(
                            f"Deployment '{deployment}' may not exist. "
                            "Verify deployment name in Azure portal."
                        )
                    else:
                        result.add_check(
                            "Connection",
                            True,
                            "Endpoint is reachable",
                        )

            except ImportError:
                # httpx not available, skip connection test
                result.add_warning("Could not test connection (httpx not installed)")
            except Exception as e:
                result.add_warning(f"Connection test failed: {e}")
                result.add_suggestion("Verify endpoint URL and network connectivity")

        return result
