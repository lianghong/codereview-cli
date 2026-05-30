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
from langchain_core.output_parsers import PydanticOutputParser
from langchain_moonshot import ChatMoonshot
from openai import RateLimitError
from pydantic import SecretStr

from codereview.config.models import ModelConfig, MoonshotConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    BATCH_PROMPT_TEMPLATE,
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import TokenTrackingMixin


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
        self._output_parser = PydanticOutputParser(pydantic_object=CodeReviewReport)
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # Set in _create_model when model_config.supports_tool_use is False
        # (kimi-k2.6 — server rejects tool_choice while thinking is enabled).
        self._use_prompt_parsing = False

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
            "base_url": self.provider_config.base_url,
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

        # Prompt-based JSON path for models whose tool_choice conflicts
        # with thinking mode (kimi-k2.6). Format instructions are injected
        # into the system prompt by `_build_batch_system_prompt` since
        # `_use_prompt_parsing` is set; the parser turns the raw response
        # into a CodeReviewReport.
        self._use_prompt_parsing = True
        return base_model

    def _create_chain(self) -> Any:
        if self._use_prompt_parsing:
            return BATCH_PROMPT_TEMPLATE | self.model | self._output_parser
        return BATCH_PROMPT_TEMPLATE | self.model

    def _is_retryable_error(self, error: Exception) -> bool:
        """Moonshot surfaces 429s as openai.RateLimitError via the OpenAI client."""
        return isinstance(error, RateLimitError)

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Exponential backoff honoring Moonshot's Retry-After header."""
        response = getattr(error, "response", None)
        if isinstance(error, RateLimitError) and response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    wait = min(float(retry_after), config.max_wait)
                    logging.info(
                        "Moonshot rate limit: waiting %.1fs (Retry-After header)",
                        wait,
                    )
                    return wait
                # PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
                except ValueError, TypeError:
                    pass

        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from Moonshot response metadata.

        ChatMoonshot extends BaseChatOpenAI so the response shape mirrors
        OpenAI's: prompt_tokens / completion_tokens land in
        ``response_metadata.token_usage``.
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

        if api_key in ("your-kimi-api-key-here", "placeholder"):
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
