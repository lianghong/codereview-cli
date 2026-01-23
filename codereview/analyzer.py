"""LLM-based code analyzer using provider abstraction."""

import warnings
from typing import List, Tuple

from codereview.batcher import FileBatch
from codereview.models import CodeReviewReport
from codereview.providers.factory import ProviderFactory


class CodeAnalyzer:
    """Analyzes code using LLM models via provider abstraction."""

    def __init__(
        self,
        model_name: str = "opus",
        temperature: float | None = None,
        provider_factory: ProviderFactory | None = None,
        # Legacy parameters for backward compatibility
        region: str | None = None,
        model_id: str | None = None,
    ):
        """Initialize analyzer.

        Args:
            model_name: Model name (ID or alias) - e.g., "opus", "gpt-5.2-codex"
            temperature: Temperature for inference (uses model-specific default if not provided)
            provider_factory: ProviderFactory instance (creates default if not provided)
            region: DEPRECATED - Use model_name instead
            model_id: DEPRECATED - Use model_name instead
        """
        # Handle legacy parameters
        if model_id is not None or region is not None:
            warnings.warn(
                "The 'model_id' and 'region' parameters are deprecated. "
                "Use 'model_name' instead. "
                "Example: CodeAnalyzer(model_name='opus')",
                DeprecationWarning,
                stacklevel=2,
            )
            # Map old model_id to new model_name if possible
            if model_id:
                model_name = self._map_legacy_model_id(model_id)

        self.model_name = model_name
        self.temperature = temperature
        self.factory = provider_factory or ProviderFactory()

        # Create provider
        self.provider = self.factory.create_provider(model_name, temperature)

        # Tracking state (analyzer-level, not delegated to provider)
        self.skipped_files: List[Tuple[str, str]] = []

    @staticmethod
    def _map_legacy_model_id(model_id: str) -> str:
        """Map legacy full model IDs to new short names.

        Args:
            model_id: Old full model ID

        Returns:
            New short model name
        """
        # Map common legacy IDs to new names
        legacy_mappings = {
            "global.anthropic.claude-opus-4-5-20251101-v1:0": "opus",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0": "sonnet",
            "global.anthropic.claude-haiku-4-5-20251001-v1:0": "haiku",
            "minimax.minimax-m2": "minimax",
            "mistral.mistral-large-3-675b-instruct": "mistral",
            "moonshot.kimi-k2-thinking": "kimi",
            "qwen.qwen3-coder-480b-a35b-v1:0": "qwen",
        }

        return legacy_mappings.get(model_id, model_id)

    def reset_state(self) -> None:
        """Reset analysis state for a fresh run."""
        self.provider.reset_state()
        self.skipped_files = []

    def analyze_batch(
        self,
        batch: FileBatch,
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """Analyze a batch of files with retry logic.

        Args:
            batch: FileBatch to analyze
            max_retries: Maximum number of retries for rate limiting (default: 3)

        Returns:
            CodeReviewReport with findings
        """
        # Prepare files_content dict
        files_content: dict[str, str] = {}

        for file_path in batch.files:
            try:
                content = file_path.read_text(encoding="utf-8")
                files_content[str(file_path)] = content
            except (OSError, IOError, UnicodeDecodeError) as e:
                # Track skipped file for reporting
                self.skipped_files.append((str(file_path), str(e)))

        # Delegate to provider
        return self.provider.analyze_batch(
            batch_number=batch.batch_number,
            total_batches=batch.total_batches,
            files_content=files_content,
            max_retries=max_retries,
        )

    # Properties that delegate to provider

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens used."""
        return self.provider.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens used."""
        return self.provider.total_output_tokens

    def estimate_cost(self) -> dict[str, float]:
        """Calculate cost from token usage."""
        return self.provider.estimate_cost()

    def get_model_display_name(self) -> str:
        """Get human-readable model name."""
        return self.provider.get_model_display_name()
