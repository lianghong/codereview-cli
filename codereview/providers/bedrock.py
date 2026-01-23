"""AWS Bedrock provider implementation."""

import time
from typing import Any

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse

# Import system prompt from config
from codereview.config import SYSTEM_PROMPT  # type: ignore[attr-defined]
from codereview.config.models import BedrockConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider


class BedrockProvider(ModelProvider):
    """AWS Bedrock implementation of ModelProvider."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: BedrockConfig,
        temperature: float | None = None,
    ):
        """Initialize Bedrock provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: Bedrock-specific configuration (region, etc.)
            temperature: Override temperature (uses model default if None)
        """
        self.model_config = model_config
        self.provider_config = provider_config

        # Determine temperature (override > model default > 0.1)
        if temperature is not None:
            self.temperature = temperature
        elif (
            model_config.inference_params
            and model_config.inference_params.temperature is not None
        ):
            self.temperature = model_config.inference_params.temperature
        else:
            self.temperature = 0.1

        # Get model-specific inference parameters
        self.top_p = None
        self.top_k = None
        self.max_tokens = 16000  # Default

        if model_config.inference_params:
            self.top_p = model_config.inference_params.top_p
            self.top_k = model_config.inference_params.top_k
            if model_config.inference_params.max_output_tokens:
                self.max_tokens = model_config.inference_params.max_output_tokens

        # Token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Create LangChain model
        self.model = self._create_model()

    def _create_model(self) -> Any:
        """Create LangChain Bedrock model with structured output."""
        # Ensure full_id is present for Bedrock models
        if not self.model_config.full_id:
            raise ValueError(
                f"Bedrock model {self.model_config.id} missing required full_id"
            )

        # Build additional model request fields
        additional_fields: dict = {}
        if self.top_p is not None:
            additional_fields["top_p"] = self.top_p
        if self.top_k is not None:
            additional_fields["top_k"] = self.top_k

        base_model = ChatBedrockConverse(
            model=self.model_config.full_id,
            region_name=self.provider_config.region,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            additional_model_request_fields=(
                additional_fields if additional_fields else None
            ),
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
        """Analyze a batch of files using AWS Bedrock.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            max_retries: Maximum number of retries for rate limiting

        Returns:
            CodeReviewReport with findings

        Raises:
            ClientError: If AWS API call fails after all retries
        """
        context = self._prepare_batch_context(
            batch_number, total_batches, files_content
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        last_error: ClientError | None = None
        for attempt in range(max_retries + 1):
            try:
                result = self.model.invoke(messages)

                # Track token usage from AWS Bedrock response metadata
                input_tokens = 0
                output_tokens = 0

                if hasattr(result, "response_metadata"):
                    usage = result.response_metadata.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

                # Fallback to estimation if actual counts unavailable
                if input_tokens == 0:
                    input_tokens = self._estimate_tokens(context)
                    output_tokens = self._estimate_tokens(str(result.model_dump_json()))

                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                return result

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                last_error = e

                # Check if it's a throttling error
                if error_code in ["ThrottlingException", "TooManyRequestsException"]:
                    if attempt < max_retries:
                        # Exponential backoff: 2^attempt seconds
                        wait_time = 2**attempt
                        time.sleep(wait_time)
                        continue

                # For other errors, raise immediately
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
        """Calculate cost from token usage."""
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
