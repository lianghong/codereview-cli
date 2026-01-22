"""LLM-based code analyzer using AWS Bedrock."""
from pathlib import Path
import time
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from codereview.models import CodeReviewReport
from codereview.batcher import FileBatch
from codereview.config import MODEL_CONFIG, SYSTEM_PROMPT


class CodeAnalyzer:
    """Analyzes code using Claude Opus 4.5 via AWS Bedrock."""

    def __init__(self, region: str | None = None):
        """
        Initialize analyzer.

        Args:
            region: AWS region (uses MODEL_CONFIG default if not provided)
        """
        self.region = region or MODEL_CONFIG["region"]
        self.model = self._create_model()

    def _create_model(self):
        """Create LangChain model with structured output."""
        base_model = ChatBedrockConverse(
            model=MODEL_CONFIG["model_id"],
            region_name=self.region,
            temperature=MODEL_CONFIG["temperature"],
            max_tokens=MODEL_CONFIG["max_tokens"],
        )

        # Configure for structured output
        return base_model.with_structured_output(CodeReviewReport)

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
            {"role": "user", "content": context}
        ]

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = self.model.invoke(messages)
                return result

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                last_error = e

                # Check if it's a throttling error
                if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                    if attempt < max_retries:
                        # Exponential backoff: 2^attempt seconds
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue

                # For other errors, raise immediately
                raise

            except Exception as e:
                # For non-AWS errors, raise immediately
                raise

        # If we exhausted all retries
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
            ""
        ]

        for file_path in batch.files:
            try:
                content = file_path.read_text()
                lines.append(f"File: {file_path.name}")
                lines.append(f"Path: {file_path}")
                lines.append("-" * 80)

                # Add line numbers
                for i, line in enumerate(content.splitlines(), start=1):
                    lines.append(f"{i:4d} | {line}")

                lines.append("")
                lines.append("=" * 80)
                lines.append("")

            except Exception as e:
                lines.append(f"ERROR reading {file_path}: {e}")
                lines.append("")

        return "\n".join(lines)
