"""DeepSeek provider via the dedicated langchain-deepseek package.

Uses ``langchain_deepseek.ChatDeepSeek`` rather than the OpenAI-compatible
adapter because the dedicated package handles DeepSeek-specific quirks
(thinking-mode toggling, the ``api_base`` parameter name, etc.) and it's
the literal recommendation in the langchain integrations doc.

Both ``deepseek-v4-pro`` and ``deepseek-v4-flash`` support tool calling
and structured output natively, so no prompt-based JSON fallback is
required (unlike DeepSeek-V4-Pro on Azure Foundry where the SGLang
backend lacks tool calling).
"""

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_deepseek import ChatDeepSeek
from openai import RateLimitError
from pydantic import SecretStr

from codereview.config.models import DeepSeekConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    BATCH_PROMPT_TEMPLATE,
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import TokenTrackingMixin


class DeepSeekProvider(TokenTrackingMixin, ModelProvider):
    """DeepSeek implementation of ModelProvider via langchain-deepseek."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: DeepSeekConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        """Initialize DeepSeek provider.

        Args:
            model_config: Model configuration with pricing and inference params.
            provider_config: DeepSeek-specific configuration (api_key, api_base).
            temperature: Override temperature (uses model default if None).
            requests_per_second: Rate limit for API calls (default: 1.0).
            callbacks: Optional list of callback handlers for streaming/progress.
            enable_output_fixing: Enable automatic retry on malformed output.
            project_context: Optional project README/documentation content.
        """
        self.callbacks = callbacks or []
        self.enable_output_fixing = enable_output_fixing
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # DeepSeek V4 family accepts temperature; allow_none preserves
        # opt-out for any future reasoning-only variants.
        self.temperature = self._resolve_temperature(
            override=temperature,
            model_config=model_config,
            provider_default=0.3,
            allow_none=True,
        )

        self.top_p = None
        self.max_tokens = 16000

        if model_config.inference_params:
            self.top_p = model_config.inference_params.top_p
            if model_config.inference_params.max_output_tokens:
                self.max_tokens = model_config.inference_params.max_output_tokens

        self._init_token_tracking()
        self.rate_limiter = self._build_rate_limiter(requests_per_second)

        self.model = self._create_model()
        self.chain = self._create_chain()

    def _create_model(self) -> Any:
        """Create LangChain ChatDeepSeek model with structured output."""
        # full_id holds the wire-level model name (deepseek-v4-pro etc.);
        # fall back to id if full_id is not set.
        wire_model = self.model_config.full_id or self.model_config.id

        # ChatDeepSeek uses `api_base` (not `base_url`) and `model_name`
        # (aliased `model`); keep the aliased form for readability.
        model_params: dict[str, Any] = {
            "model": wire_model,
            "api_base": self.provider_config.api_base,
            "api_key": SecretStr(str(self.provider_config.api_key)),
            "max_tokens": self.max_tokens,
            "rate_limiter": self.rate_limiter,
            "callbacks": self.callbacks if self.callbacks else None,
            "streaming": bool(self.callbacks),
            "timeout": self.provider_config.request_timeout,
            # DeepSeek-V4-Pro defaults to thinking/reasoner mode server-side,
            # which rejects tool_choice="auto" (the choice langchain pins
            # for with_structured_output). Disable thinking explicitly so
            # tool-calling-based structured output works. V4-Flash is
            # non-thinking by default; this field is harmless for it.
            # Operators who want chain-of-thought can override via
            # `inference_params.thinking: enabled` in models.yaml.
            "extra_body": self._build_extra_body(),
        }

        if self.temperature is not None:
            model_params["temperature"] = self.temperature
        if self.top_p is not None:
            model_params["top_p"] = self.top_p

        base_model = ChatDeepSeek(**model_params)

        # Both V4-Pro and V4-Flash support tool-calling-based structured
        # output (per https://api-docs.deepseek.com/guides/tool_calls)
        # *when thinking mode is disabled*. include_raw=True so we can
        # pull real token counts from the underlying AIMessage.
        return base_model.with_structured_output(CodeReviewReport, include_raw=True)

    def _build_extra_body(self) -> dict[str, Any]:
        """Construct the OpenAI-client extra_body payload for DeepSeek.

        Disables thinking mode by default so ``tool_choice`` (forced by
        ``with_structured_output``) is accepted on V4-Pro (which routes to
        ``deepseek-reasoner`` server-side and otherwise rejects forced
        tool calls). Honors an override in ``inference_params.thinking``:

        - ``True`` / ``"enabled"`` / ``"high"`` / ``"max"`` → enabled
        - ``False`` / ``"disabled"`` / ``None`` (default) → disabled
        """
        thinking_state = "disabled"
        params = getattr(self.model_config, "inference_params", None)
        override = getattr(params, "thinking", None) if params is not None else None
        if override is True or (
            isinstance(override, str) and override.lower() in ("enabled", "high", "max")
        ):
            thinking_state = "enabled"
        return {"thinking": {"type": thinking_state}}

    def _create_chain(self) -> Any:
        return BATCH_PROMPT_TEMPLATE | self.model

    def _is_retryable_error(self, error: Exception) -> bool:
        """DeepSeek surfaces 429s as openai.RateLimitError via the OpenAI client."""
        return isinstance(error, RateLimitError)

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Exponential backoff honoring DeepSeek's Retry-After header."""
        if isinstance(error, RateLimitError) and hasattr(error, "response"):
            retry_after = error.response.headers.get("retry-after")
            if retry_after:
                try:
                    wait = min(float(retry_after), config.max_wait)
                    logging.info(
                        "DeepSeek rate limit: waiting %ds (Retry-After header)",
                        wait,
                    )
                    return wait
                # PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
                except ValueError, TypeError:
                    pass

        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from DeepSeek response metadata.

        DeepSeek follows OpenAI's response shape, so prompt_tokens /
        completion_tokens land in ``response_metadata.token_usage``.
        """
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
        """Analyze a batch of files using DeepSeek."""
        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content, self.project_context
        )

        chain_input = {
            "system_prompt": self._build_batch_system_prompt(files_content),
            "batch_context": batch_context,
        }

        retry_config = RetryConfig(max_retries=max_retries, base_wait=2.0)
        return self._execute_with_retry(chain_input, retry_config, batch_context)

    def validate_credentials(self) -> ValidationResult:
        """Validate DeepSeek configuration before any analysis call."""
        result = ValidationResult(valid=True, provider="DeepSeek")

        api_key = self.provider_config.api_key
        if not api_key:
            result.valid = False
            result.add_check("API Key", False, "DEEPSEEK_API_KEY is not set")
            result.add_suggestion(
                "Export DEEPSEEK_API_KEY=<your-deepseek-key>; "
                "get one at https://platform.deepseek.com/api_keys"
            )
            return result

        if api_key in ("your-deepseek-api-key-here", "placeholder"):
            result.valid = False
            result.add_check(
                "API Key", False, "DEEPSEEK_API_KEY appears to be a placeholder"
            )
            return result

        result.add_check("API Key", True, f"Key set ({len(api_key)} chars)")

        if not self.provider_config.api_base.startswith("https://"):
            result.valid = False
            result.add_check("Base URL", False, "api_base must use HTTPS")
            return result

        result.add_check("Base URL", True, f"Endpoint: {self.provider_config.api_base}")

        wire_model = self.model_config.full_id or self.model_config.id
        result.add_check("Model", True, f"Model: {wire_model}")

        return result
