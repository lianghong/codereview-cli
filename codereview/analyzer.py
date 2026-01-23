"""LLM-based code analyzer using AWS Bedrock."""

import time
from typing import Any, List, Tuple

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse

from codereview.batcher import FileBatch
from codereview.config import MODEL_CONFIG, SUPPORTED_MODELS, SYSTEM_PROMPT, ModelInfo
from codereview.models import CodeReviewReport


class CodeAnalyzer:
    """Analyzes code using LLM models via AWS Bedrock."""

    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
        temperature: float | None = None,
    ):
        """
        Initialize analyzer.

        Args:
            region: AWS region (uses MODEL_CONFIG default if not provided)
            model_id: Model ID to use (uses MODEL_CONFIG default if not provided)
            temperature: Temperature for inference (uses model-specific default if not provided)
        """
        self.region = region or MODEL_CONFIG["region"]
        self.model_id = model_id or MODEL_CONFIG["model_id"]

        # Get model-specific defaults
        model_info: ModelInfo | None = SUPPORTED_MODELS.get(self.model_id)
        if model_info:
            default_temp = model_info.get(
                "default_temperature", MODEL_CONFIG["temperature"]
            )
        else:
            default_temp = MODEL_CONFIG["temperature"]
        self.temperature = temperature if temperature is not None else default_temp

        # Get optional inference parameters (model-specific)
        self.top_p = model_info.get("default_top_p") if model_info else None
        self.top_k = model_info.get("default_top_k") if model_info else None
        self.max_tokens = (
            model_info.get("max_output_tokens", MODEL_CONFIG["max_tokens"])
            if model_info
            else MODEL_CONFIG["max_tokens"]
        )

        self.model = self._create_model()
        self.reset_state()

    def reset_state(self) -> None:
        """Reset analysis state for a fresh run."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.skipped_files: List[Tuple[str, str]] = []

    def _create_model(self) -> Any:
        """Create LangChain model with structured output."""
        # Build additional model request fields for non-standard parameters
        # These are passed to the Bedrock Converse API's inferenceConfig
        additional_fields: dict = {}
        if self.top_p is not None:
            additional_fields["top_p"] = self.top_p
        if self.top_k is not None:
            additional_fields["top_k"] = self.top_k

        base_model = ChatBedrockConverse(
            model=self.model_id,
            region_name=self.region,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            additional_model_request_fields=(
                additional_fields if additional_fields else None
            ),
        )

        # Configure for structured output
        return base_model.with_structured_output(CodeReviewReport)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token heuristic)."""
        return len(text) // 4

    def analyze_batch(self, batch: FileBatch, max_retries: int = 3) -> CodeReviewReport:
        """
        Analyze a batch of files with retry logic.

        Args:
            batch: FileBatch to analyze
            max_retries: Maximum number of retries for rate limiting (default: 3)

        Returns:
            CodeReviewReport with findings

        Raises:
            ClientError: If AWS API call fails after all retries
        """
        context = self._prepare_batch_context(batch)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        last_error: ClientError | None = None
        for attempt in range(max_retries + 1):
            try:
                result = self.model.invoke(messages)

                # Track token usage from AWS Bedrock response metadata
                # Try to access actual usage from response, fall back to estimation
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

                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens

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

        # If we exhausted all retries (should only happen with throttling errors)
        if last_error is None:
            raise RuntimeError("Unexpected: last_error is None after retry loop")
        raise last_error

    def _prepare_batch_context(self, batch: FileBatch) -> str:
        """
        Prepare context string for LLM.

        Args:
            batch: FileBatch to prepare

        Returns:
            Formatted context string
        """
        lines = [
            f"Analyzing Batch {batch.batch_number}/{batch.total_batches}",
            f"Files in this batch: {len(batch.files)}",
            "",
            "=" * 80,
            "",
        ]

        for file_path in batch.files:
            try:
                content = file_path.read_text(encoding="utf-8")
                lines.append(f"File: {file_path.name}")
                lines.append(f"Path: {file_path}")
                lines.append("-" * 80)

                # Add line numbers
                for i, line in enumerate(content.splitlines(), start=1):
                    lines.append(f"{i:4d} | {line}")

                lines.append("")
                lines.append("=" * 80)
                lines.append("")

            except (OSError, IOError, UnicodeDecodeError) as e:
                error_msg = f"ERROR reading {file_path}: {e}"
                lines.append(error_msg)
                lines.append("")
                # Track skipped file for reporting
                self.skipped_files.append((str(file_path), str(e)))

        return "\n".join(lines)
