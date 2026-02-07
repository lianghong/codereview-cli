"""Google Generative AI (Gemini) provider implementation."""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

# Import system prompt from config
from codereview.config import SYSTEM_PROMPT
from codereview.config.models import GoogleGenAIConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider, RetryConfig, ValidationResult
from codereview.providers.mixins import TokenTrackingMixin

# Shared prompt template for consistent formatting
BATCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("human", "{batch_context}"),
    ]
)


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
        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

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
            "google_api_key": self.provider_config.api_key,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "timeout": self.provider_config.request_timeout,
            "callbacks": self.callbacks if self.callbacks else None,
        }

        # Add optional parameters
        if self.top_p is not None:
            model_params["top_p"] = self.top_p
        if self.top_k is not None:
            model_params["top_k"] = self.top_k

        base_model = ChatGoogleGenerativeAI(**model_params)
        return base_model.with_structured_output(CodeReviewReport, method="json_schema")

    def _create_chain(self) -> Any:
        """Create LangChain chain with prompt template."""
        return BATCH_PROMPT_TEMPLATE | self.model

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is a retryable Google API error."""
        error_type = type(error).__name__
        return error_type in ("ResourceExhausted", "ServiceUnavailable")

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Calculate exponential backoff for retry attempts."""
        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from Google GenAI response metadata."""
        # Google GenAI uses usage_metadata attribute
        if hasattr(result, "usage_metadata"):
            usage = result.usage_metadata
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                return (input_tokens, output_tokens)
        # Fallback to response_metadata (older versions)
        if hasattr(result, "response_metadata"):
            usage = result.response_metadata.get("usage_metadata", {})
            input_tokens = usage.get("prompt_token_count", 0) or usage.get(
                "input_tokens", 0
            )
            output_tokens = usage.get("candidates_token_count", 0) or usage.get(
                "output_tokens", 0
            )
            return (input_tokens, output_tokens)
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
            max_retries: Maximum number of retries for API errors (default: 3)

        Returns:
            CodeReviewReport with findings
        """
        if max_retries is None:
            max_retries = 3

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
        if not api_key or api_key in ("", "your-api-key-here", "placeholder"):
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
