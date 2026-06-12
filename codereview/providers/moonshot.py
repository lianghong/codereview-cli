"""Moonshot AI (Kimi) provider via the langchain-moonshot package.

Uses ``langchain_moonshot.ChatMoonshot`` (extends ``BaseChatOpenAI``), which
handles Moonshot-specific quirks like the kimi-k2.5 thinking-mode parameter
constraints. The user-facing env var is ``KIMI_API_KEY`` per project
convention; we plumb it through explicitly rather than relying on
ChatMoonshot's default ``MOONSHOT_API_KEY`` lookup so the naming matches
the rest of the CLI.

Models with ``supports_tool_use: false`` (e.g. kimi-k2.6, whose thinking
mode is incompatible with the tool_choice that ``.with_structured_output``
would set) are routed through prompt-based JSON parsing via
``PydanticOutputParser`` — same pattern as MiniMax M2.5 on Bedrock and
DeepSeek-V4-Pro on Azure.
"""

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_moonshot import ChatMoonshot
from pydantic import SecretStr

from codereview.config.models import ModelConfig, MoonshotConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import (
    TokenTrackingMixin,
    extract_openai_token_usage,
    is_openai_retryable_error,
    is_placeholder_api_key,
    parse_retry_after,
    require_https,
)


class MoonshotProvider(TokenTrackingMixin, ModelProvider):
    """Moonshot (Kimi) implementation of ModelProvider via langchain-moonshot."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: MoonshotConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        """Initialize Moonshot provider.

        Args:
            model_config: Model configuration with pricing and inference params.
            provider_config: Moonshot-specific configuration (api_key, base_url).
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

        # Kimi family accepts temperature; allow_none preserves opt-out for
        # any future reasoning-only variants.
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
        """Create LangChain ChatMoonshot model with structured output."""
        # full_id holds the wire-level model name (kimi-k2.6 etc.); fall
        # back to id if full_id is not set.
        wire_model = self.model_config.full_id or self.model_config.id

        # ChatMoonshot uses `api_base` (alias `base_url`) and `model_name`
        # (aliased `model`); `api_key` accepts SecretStr | None directly.
        model_params: dict[str, Any] = {
            "model": wire_model,
            # Fail closed on cleartext so KIMI_API_KEY can't be sent over HTTP
            # even if validate_credentials was skipped.
            "base_url": require_https(self.provider_config.base_url, "base_url"),
            "api_key": SecretStr(str(self.provider_config.api_key)),
            "max_tokens": self.max_tokens,
            "rate_limiter": self.rate_limiter,
            "callbacks": self.callbacks if self.callbacks else None,
            "streaming": bool(self.callbacks),
            "timeout": self.provider_config.request_timeout,
        }

        if self.temperature is not None:
            model_params["temperature"] = self.temperature
        if self.top_p is not None:
            model_params["top_p"] = self.top_p

        base_model = ChatMoonshot(**model_params)

        if self.model_config.supports_tool_use:
            # Tool-calling-based structured output: ChatMoonshot extends
            # BaseChatOpenAI so .with_structured_output works the same way.
            # include_raw=True so we can pull real token counts from the raw
            # AIMessage's response_metadata.
            return base_model.with_structured_output(CodeReviewReport, include_raw=True)

        # Prompt-based JSON path for models whose tool_choice conflicts with
        # thinking mode (kimi-k2.6) — routing and _create_chain live in the
        # base class.
        return self._apply_structured_output(base_model)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Retry rate limits plus transient timeouts/connection/5xx errors.

        Moonshot surfaces these via the OpenAI client, so the shared helper applies.
        """
        return is_openai_retryable_error(error)

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Exponential backoff honoring Moonshot's Retry-After header."""
        wait = parse_retry_after(error, config.max_wait)
        if wait is not None:
            logging.info(
                "Moonshot rate limit: waiting %.1fs (Retry-After header)", wait
            )
            return wait
        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from Moonshot's OpenAI-shaped response metadata.

        ChatMoonshot extends BaseChatOpenAI, so usage lands in
        ``response_metadata.token_usage`` like the other OpenAI-compat providers.
        """
        return extract_openai_token_usage(result)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 5,
    ) -> CodeReviewReport:
        """Analyze a batch of files using Moonshot."""
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
        """Validate Moonshot configuration before any analysis call."""
        result = ValidationResult(valid=True, provider="Moonshot")

        api_key = self.provider_config.api_key
        if not api_key:
            result.valid = False
            result.add_check("API Key", False, "KIMI_API_KEY is not set")
            result.add_suggestion(
                "Export KIMI_API_KEY=<your-moonshot-key>; "
                "get one at https://platform.moonshot.cn "
                "(or platform.moonshot.ai for the international platform)"
            )
            return result

        # Reject the documented placeholders too: the README tells users to
        # export KIMI_API_KEY="your-moonshot-key", so --validate must fail fast
        # on that exact string instead of deferring a 401 to the first call.
        if is_placeholder_api_key(
            api_key, ("your-kimi-api-key-here", "your-moonshot-key")
        ):
            result.valid = False
            result.add_check(
                "API Key", False, "KIMI_API_KEY appears to be a placeholder"
            )
            return result

        if len(api_key) < 20:
            result.add_warning("API key seems unusually short. Verify it's correct.")
        result.add_check("API Key", True, "API key configured")

        if not self.provider_config.base_url.startswith("https://"):
            result.valid = False
            result.add_check("Base URL", False, "base_url must use HTTPS")
            return result

        result.add_check("Base URL", True, f"Endpoint: {self.provider_config.base_url}")

        wire_model = self.model_config.full_id or self.model_config.id
        result.add_check("Model", True, f"Model: {wire_model}")

        return result
