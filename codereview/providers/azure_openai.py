"""Azure OpenAI provider implementation."""

import time
from typing import Any

from langchain_openai import AzureChatOpenAI
from openai import RateLimitError

from codereview.config.models import AzureOpenAIConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider

# Import system prompt from config
from codereview.config import SYSTEM_PROMPT


class AzureOpenAIProvider(ModelProvider):
    """Azure OpenAI implementation of ModelProvider."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: AzureOpenAIConfig,
        temperature: float | None = None,
    ):
        """Initialize Azure OpenAI provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: Azure-specific configuration (endpoint, API key, etc.)
            temperature: Override temperature (uses model default if None)
        """
        self.model_config = model_config
        self.provider_config = provider_config

        # Determine temperature (override > model default > 0.0 for Azure)
        if temperature is not None:
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

        # Create LangChain model
        self.model = self._create_model()

    def _create_model(self) -> Any:
        """Create LangChain Azure OpenAI model with structured output."""
        # Build model kwargs
        model_kwargs: dict = {}
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p

        base_model = AzureChatOpenAI(
            deployment_name=self.model_config.deployment_name,
            azure_endpoint=str(self.provider_config.endpoint),
            api_key=self.provider_config.api_key,
            api_version=self.provider_config.api_version,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_kwargs=model_kwargs,
        )

        # Configure for structured output
        return base_model.with_structured_output(CodeReviewReport)

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
        context = self._prepare_batch_context(
            batch_number, total_batches, files_content
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        last_error: RateLimitError | None = None
        for attempt in range(max_retries + 1):
            try:
                result = self.model.invoke(messages)

                # Track token usage from Azure OpenAI response metadata
                input_tokens = 0
                output_tokens = 0

                if hasattr(result, "response_metadata"):
                    token_usage = result.response_metadata.get("token_usage", {})
                    input_tokens = token_usage.get("prompt_tokens", 0)
                    output_tokens = token_usage.get("completion_tokens", 0)

                # Fallback to estimation if actual counts unavailable
                if input_tokens == 0 and output_tokens == 0:
                    input_tokens = self._estimate_tokens(context)
                    output_tokens = self._estimate_tokens(str(result.model_dump_json()))

                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                return result

            except RateLimitError as e:
                last_error = e

                if attempt < max_retries:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue

                # For all retries exhausted, raise
                raise

        # If we exhausted all retries
        if last_error is None:
            raise RuntimeError("Unexpected: last_error is None after retry loop")
        raise last_error

    def _prepare_batch_context(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
    ) -> str:
        """Prepare context string for LLM.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents

        Returns:
            Formatted context string
        """
        lines = [
            f"Analyzing Batch {batch_number}/{total_batches}",
            f"Files in this batch: {len(files_content)}",
            "",
            "=" * 80,
            "",
        ]

        for file_path, content in files_content.items():
            lines.append(f"File: {file_path}")
            lines.append("-" * 80)

            # Add line numbers
            for i, line in enumerate(content.splitlines(), start=1):
                lines.append(f"{i:4d} | {line}")

            lines.append("")
            lines.append("=" * 80)
            lines.append("")

        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token heuristic)."""
        return len(text) // 4

    def get_model_display_name(self) -> str:
        """Get human-readable model name."""
        return self.model_config.name

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
        output_cost = (self._total_output_tokens / 1_000_000) * pricing.output_per_million

        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }
