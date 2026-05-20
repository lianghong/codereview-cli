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
from openai import RateLimitError
from pydantic import SecretStr

from codereview.config.models import ModelConfig, ZAIConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    BATCH_PROMPT_TEMPLATE,
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import TokenTrackingMixin


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

        base_model = ChatOpenAI(**model_params)

        # Z.AI's GLM models support tool calling (per docs.z.ai); use the
        # tool-calling-based structured output path. include_raw=True so we
        # can extract real token counts from the AIMessage.
        return base_model.with_structured_output(CodeReviewReport, include_raw=True)

    def _create_chain(self) -> Any:
        """Create LangChain chain with prompt template."""
        return BATCH_PROMPT_TEMPLATE | self.model

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is a retryable Z.AI rate limit error.

        Z.AI's OpenAI-compatible endpoint surfaces 429s as openai.RateLimitError
        through the same client, so the Azure pattern applies.
        """
        return isinstance(error, RateLimitError)

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Exponential backoff for Z.AI rate-limit responses."""
        if isinstance(error, RateLimitError) and hasattr(error, "response"):
            retry_after = error.response.headers.get("retry-after")
            if retry_after:
                try:
                    wait = min(float(retry_after), config.max_wait)
                    logging.info(
                        "Z.AI rate limit: waiting %ds (Retry-After header)", wait
                    )
                    return wait
                # PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
                except ValueError, TypeError:
                    pass

        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from Z.AI response metadata.

        Z.AI mirrors OpenAI's response shape, so prompt_tokens /
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

        if api_key in ("your-zai-api-key-here", "placeholder"):
            result.valid = False
            result.add_check(
                "API Key", False, "ZAI_API_KEY appears to be a placeholder"
            )
            return result

        result.add_check("API Key", True, f"Key set ({len(api_key)} chars)")

        if not self.provider_config.base_url.startswith("https://"):
            result.valid = False
            result.add_check("Base URL", False, "base_url must use HTTPS")
            return result

        result.add_check("Base URL", True, f"Endpoint: {self.provider_config.base_url}")

        wire_model = self.model_config.full_id or self.model_config.id
        result.add_check("Model", True, f"Model: {wire_model}")

        return result
