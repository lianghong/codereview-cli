"""Z.AI (Zhipu international) provider via OpenAI-compatible endpoint.

Z.AI exposes its GLM models through an OpenAI-compatible API at
https://api.z.ai/api/paas/v4/. We reuse langchain-openai's ChatOpenAI with
a custom ``base_url`` rather than depending on langchain-community for the
ChatZhipuAI class — that path uses the Chinese endpoint (open.bigmodel.cn)
and a different env var (ZHIPUAI_API_KEY), and pulls in a heavy dependency
tree we don't otherwise need.

The OpenAI-compatible adapter natively supports ``with_structured_output``
and tool calling, so unlike DeepSeek-V4-Pro on Azure Foundry we don't fall
back to prompt-based JSON parsing.
"""

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from codereview.config.models import ModelConfig, ZAIConfig
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


class ZAIProvider(TokenTrackingMixin, ModelProvider):
    """Z.AI implementation of ModelProvider via OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: ZAIConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        """Initialize Z.AI provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: Z.AI-specific configuration (api_key, base_url)
            temperature: Override temperature (uses model default if None)
            requests_per_second: Rate limit for API calls (default: 1.0)
            callbacks: Optional list of callback handlers for streaming/progress
            enable_output_fixing: Enable automatic retry on malformed output
            project_context: Optional project README/documentation content
        """
        self.callbacks = callbacks or []
        self.enable_output_fixing = enable_output_fixing
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # GLM models accept temperature; allow_none preserves opt-out for
        # any future reasoning variants that don't.
        self.temperature = self._resolve_temperature(
            override=temperature,
            model_config=model_config,
            provider_default=0.3,
            allow_none=True,
        )

        # Inference params
        self.top_p = None
        self.max_tokens = 16000

        if model_config.inference_params:
            self.top_p = model_config.inference_params.top_p
            if model_config.inference_params.max_output_tokens:
                self.max_tokens = model_config.inference_params.max_output_tokens

        self._init_token_tracking()

        # Rate limiter for API calls
        self.rate_limiter = self._build_rate_limiter(requests_per_second)

        self.model = self._create_model()
        self.chain = self._create_chain()

    def _create_model(self) -> Any:
        """Create LangChain ChatOpenAI model pointing at Z.AI's endpoint."""
        # Z.AI's OpenAI-compatible endpoint requires the model name in the
        # request body (real OpenAI ignores it; routing happens via URL).
        # full_id holds the wire-level model name (e.g. "glm-5.1"); fall
        # back to id if full_id is not set.
        wire_model = self.model_config.full_id or self.model_config.id

        model_params: dict[str, Any] = {
            "model": wire_model,
            # Fail closed on cleartext so the API key can't be sent over HTTP
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

        base_model = ChatOpenAI(**model_params)

        # Tool-use vs prompt-parsing routing (and _create_chain) live in the
        # base class; supports_tool_use in models.yaml decides the path.
        # GLM-5.1 sets it false: Z.AI's endpoint ignores OpenAI's json_schema
        # response_format and emits markdown-fenced JSON, which the
        # PydanticOutputParser path strips.
        return self._apply_structured_output(base_model)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Retry rate limits plus transient timeouts/connection/5xx errors.

        Z.AI's OpenAI-compatible endpoint surfaces these via the OpenAI client,
        so the shared helper applies.
        """
        return is_openai_retryable_error(error)

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Exponential backoff for Z.AI, honoring a Retry-After header."""
        wait = parse_retry_after(error, config.max_wait)
        if wait is not None:
            logging.info("Z.AI rate limit: waiting %.1fs (Retry-After header)", wait)
            return wait
        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from Z.AI's OpenAI-shaped response metadata."""
        return extract_openai_token_usage(result)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 5,
    ) -> CodeReviewReport:
        """Analyze a batch of files using Z.AI."""
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
        """Validate Z.AI configuration before any analysis call."""
        result = ValidationResult(valid=True, provider="Z.AI")

        api_key = self.provider_config.api_key
        if not api_key:
            result.valid = False
            result.add_check("API Key", False, "ZAI_API_KEY is not set")
            result.add_suggestion(
                "Export ZAI_API_KEY=<your-z.ai-key>; get one at https://z.ai"
            )
            return result

        # "your-zai-key" is the exact string the README documents
        if is_placeholder_api_key(api_key, ("your-zai-api-key-here", "your-zai-key")):
            result.valid = False
            result.add_check(
                "API Key", False, "ZAI_API_KEY appears to be a placeholder"
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
