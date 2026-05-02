"""Azure OpenAI provider implementation."""

import logging
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI
from openai import RateLimitError
from pydantic import SecretStr

# Import system prompt from config
from codereview.config import SYSTEM_PROMPT
from codereview.config.models import AzureOpenAIConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    BATCH_PROMPT_TEMPLATE,
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import TokenTrackingMixin


class AzureOpenAIProvider(TokenTrackingMixin, ModelProvider):
    """Azure OpenAI implementation of ModelProvider."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: AzureOpenAIConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        """Initialize Azure OpenAI provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: Azure-specific configuration (endpoint, API key, etc.)
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

        self.temperature = self._resolve_temperature(
            override=temperature,
            model_config=model_config,
            provider_default=0.0,
        )

        # Get model-specific inference parameters
        self.top_p = None
        self.max_tokens = 16000  # Default

        if model_config.inference_params:
            self.top_p = model_config.inference_params.top_p
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
        """Create LangChain Azure OpenAI model with structured output."""
        # Build base parameters
        model_params: dict[str, Any] = {
            "azure_deployment": self.model_config.deployment_name,
            "azure_endpoint": str(self.provider_config.endpoint),
            "api_key": SecretStr(str(self.provider_config.api_key)),
            "api_version": self.provider_config.api_version,
            "max_tokens": self.max_tokens,
            "rate_limiter": self.rate_limiter,
            "callbacks": self.callbacks if self.callbacks else None,
            "streaming": bool(self.callbacks),  # Enable streaming if callbacks provided
            "timeout": self.provider_config.request_timeout,  # Request timeout in seconds
        }

        # Enable Responses API if model requires it (e.g., GPT-5.3 Codex)
        # Models using Responses API don't support temperature/top_p parameters
        if self.model_config.use_responses_api:
            model_params["use_responses_api"] = True
        else:
            # Only add temperature/top_p for models that support them
            model_params["temperature"] = self.temperature
            if self.top_p is not None:
                model_params["top_p"] = self.top_p

        base_model = AzureChatOpenAI(**model_params)

        # Configure for structured output
        # include_raw=True returns {"raw": AIMessage, "parsed": CodeReviewReport}
        # so we can extract actual token counts from the raw AIMessage
        return base_model.with_structured_output(CodeReviewReport, include_raw=True)

    def _create_chain(self) -> Any:
        """Create LangChain chain with prompt template."""
        return BATCH_PROMPT_TEMPLATE | self.model

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is a retryable Azure rate limit error."""
        return isinstance(error, RateLimitError)

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Calculate backoff respecting Azure's Retry-After header.

        Azure 429 responses include a Retry-After header indicating the
        seconds to wait before the rate limit window resets. Using this
        value instead of short exponential backoff prevents wasting all
        retries within the same rate limit window.
        """
        if isinstance(error, RateLimitError) and hasattr(error, "response"):
            retry_after = error.response.headers.get("retry-after")
            if retry_after:
                try:
                    wait = min(float(retry_after), config.max_wait)
                    logging.info(
                        "Azure rate limit: waiting %ds (Retry-After header)", wait
                    )
                    return wait
                # PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
                except ValueError, TypeError:
                    pass

        # Fallback: longer base wait for Azure (rate limit windows are ~60s)
        return min(10.0 * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from Azure OpenAI response metadata."""
        if hasattr(result, "response_metadata"):
            token_usage = result.response_metadata.get("token_usage", {})
            return (
                token_usage.get("prompt_tokens", 0),
                token_usage.get("completion_tokens", 0),
            )
        return (0, 0)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 5,
    ) -> CodeReviewReport:
        """Analyze a batch of files using Azure OpenAI.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            max_retries: Maximum number of retries for rate limiting (default: 5)

        Returns:
            CodeReviewReport with findings

        Raises:
            RateLimitError: If Azure API rate limit exceeded after all retries
        """
        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content, self.project_context
        )

        chain_input = {
            "system_prompt": SYSTEM_PROMPT,
            "batch_context": batch_context,
        }

        retry_config = RetryConfig(max_retries=max_retries, base_wait=5.0)
        return self._execute_with_retry(chain_input, retry_config, batch_context)

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

                # Normalize endpoint: remove common suffixes that users might add
                base_endpoint = endpoint.rstrip("/")
                for suffix in ["/openai/v1", "/openai", "/v1"]:
                    if base_endpoint.endswith(suffix):
                        base_endpoint = base_endpoint[: -len(suffix)]
                        break

                # Quick request to check endpoint is reachable
                # Use GET on models endpoint as HEAD requests are unreliable on Azure
                test_url = f"{base_endpoint}/openai/models"
                params = {"api-version": api_version} if api_version else {}
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(
                        test_url,
                        headers={"api-key": api_key},
                        params=params,
                    )
                    # 200 means endpoint is reachable and auth works
                    # 401/403 means endpoint exists but auth issue
                    # Other codes mean endpoint is at least reachable
                    if response.status_code == 200:
                        result.add_check(
                            "Connection",
                            True,
                            "Endpoint is reachable and authenticated",
                        )
                    elif response.status_code in (401, 403):
                        result.add_check(
                            "Connection",
                            True,
                            "Endpoint is reachable (auth will be verified on first call)",
                        )
                    else:
                        result.add_check(
                            "Connection",
                            True,
                            f"Endpoint responded (status: {response.status_code})",
                        )

            except ImportError:
                # httpx not available, skip connection test
                result.add_warning("Could not test connection (httpx not installed)")
            except Exception as e:
                error_msg = str(e)
                # Redact the API key aggressively: full-string match is
                # insufficient because intermediate layers may log only a
                # prefix or URL-encode special chars. We scrub both the full
                # key and any 16-char prefix if the key is long enough.
                if api_key:
                    error_msg = error_msg.replace(api_key, "***")
                    if len(api_key) >= 16:
                        error_msg = error_msg.replace(api_key[:16], "***")
                result.add_warning(f"Connection test failed: {error_msg}")
                result.add_suggestion("Verify endpoint URL and network connectivity")

        return result
