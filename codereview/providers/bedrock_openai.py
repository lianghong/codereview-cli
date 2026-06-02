"""OpenAI-on-Bedrock provider via the OpenAI-compatible endpoint.

AWS exposes OpenAI's frontier models (GPT-5.5, GPT-5.4, Codex) on Amazon
Bedrock through an OpenAI-compatible surface. This is a *different* path from
every other Bedrock model in this project: rather than ``ChatBedrockConverse``
and the AWS SigV4 credential chain, the OpenAI-compatible endpoint
authenticates with an Amazon Bedrock **API key** (a bearer token) and is driven
with langchain-openai's ``ChatOpenAI`` pointed at a custom ``base_url`` — the
same mechanism as the Z.AI provider (``providers/zai.py``), which is why this
mirrors it closely.

Underlying transport: ``ChatOpenAI`` → the ``openai`` SDK → Bedrock's endpoint.
Both packages are already dependencies (``langchain-openai`` pulls ``openai``),
so no new dependency is required. The Bedrock API key / endpoint are read from
the canonical ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` env vars.

GPT-5.5 / GPT-5.4 are reasoning models: they reject ``temperature`` / ``top_p``
(handled via ``allow_none=True`` + omitting ``default_temperature`` in the YAML)
and surface chain-of-thought through the OpenAI **Responses API**, enabled with
``use_responses_api`` on the model entry — the same flag the Azure provider sets
for the GPT-5 series.
"""

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from codereview.config.models import BedrockOpenAIConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    BATCH_PROMPT_TEMPLATE,
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import (
    TokenTrackingMixin,
    extract_openai_token_usage,
    is_openai_retryable_error,
    parse_retry_after,
    require_https,
)


class BedrockOpenAIProvider(TokenTrackingMixin, ModelProvider):
    """OpenAI-on-Bedrock implementation via the OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: BedrockOpenAIConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        """Initialize the OpenAI-on-Bedrock provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: Bedrock OpenAI config (api_key, base_url)
            temperature: Override temperature (uses model default if None)
            requests_per_second: Rate limit for API calls (default: 1.0)
            callbacks: Optional list of callback handlers for streaming/progress
            enable_output_fixing: Enable automatic retry on malformed output
            project_context: Optional project README/documentation content
        """
        self.callbacks = callbacks or []
        self.enable_output_fixing = enable_output_fixing
        self._output_parser = PydanticOutputParser(pydantic_object=CodeReviewReport)
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # Set in _create_model when model_config.supports_tool_use is False.
        self._use_prompt_parsing = False

        # GPT-5.5 / GPT-5.4 are reasoning models and reject temperature/top_p;
        # allow_none preserves that opt-out (no default_temperature in YAML).
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
        """Create a ChatOpenAI model pointing at Bedrock's OpenAI endpoint."""
        # The OpenAI-compatible endpoint requires the model name in the request
        # body. full_id holds the wire-level model id (e.g. the value of
        # BEDROCK_OPENAI_MODEL_ID such as "openai.gpt-5.5-..."); fall back to id.
        wire_model = self.model_config.full_id or self.model_config.id

        model_params: dict[str, Any] = {
            "model": wire_model,
            # Fail closed on cleartext so the Bedrock bearer key can't be sent
            # over HTTP even if validate_credentials was skipped.
            "base_url": require_https(self.provider_config.base_url, "base_url"),
            "api_key": SecretStr(str(self.provider_config.api_key)),
            "max_tokens": self.max_tokens,
            "rate_limiter": self.rate_limiter,
            "callbacks": self.callbacks if self.callbacks else None,
            "streaming": bool(self.callbacks),
            "timeout": self.provider_config.request_timeout,
        }

        # GPT-5 reasoning models surface reasoning summaries only through the
        # Responses API, and reject temperature/top_p. Mirror the Azure
        # provider: enable the Responses API and skip the sampling params.
        if self.model_config.use_responses_api:
            model_params["use_responses_api"] = True
        else:
            if self.temperature is not None:
                model_params["temperature"] = self.temperature
            if self.top_p is not None:
                model_params["top_p"] = self.top_p

        base_model = ChatOpenAI(**model_params)

        if self.model_config.supports_tool_use:
            # Tool-calling-based structured output. include_raw=True so we can
            # extract real token counts from the AIMessage.
            return base_model.with_structured_output(CodeReviewReport, include_raw=True)

        # Tool-use-less models: parse JSON via prompt format instructions
        # (injected in _build_batch_system_prompt because _use_prompt_parsing is
        # set) and PydanticOutputParser — same pattern as Bedrock's MiniMax M2.5
        # / Z.AI's GLM-5.1 path.
        self._use_prompt_parsing = True
        return base_model

    def _create_chain(self) -> Any:
        """Create LangChain chain with prompt template."""
        if self._use_prompt_parsing:
            return BATCH_PROMPT_TEMPLATE | self.model | self._output_parser
        return BATCH_PROMPT_TEMPLATE | self.model

    def _is_retryable_error(self, error: Exception) -> bool:
        """Retry rate limits plus transient timeouts/connection/5xx errors.

        The OpenAI-compatible endpoint surfaces these as the standard openai
        client exceptions, so the shared helper applies.
        """
        return is_openai_retryable_error(error)

    def _calculate_backoff(
        self, error: Exception, attempt: int, config: RetryConfig
    ) -> float:
        """Exponential backoff honoring a Retry-After header when present."""
        wait = parse_retry_after(error, config.max_wait)
        if wait is not None:
            logging.info(
                "Bedrock OpenAI rate limit: waiting %.1fs (Retry-After header)", wait
            )
            return wait
        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from the OpenAI-shaped response metadata."""
        return extract_openai_token_usage(result)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 5,
    ) -> CodeReviewReport:
        """Analyze a batch of files using an OpenAI model on Bedrock."""
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
        """Validate Bedrock OpenAI configuration before any analysis call."""
        result = ValidationResult(valid=True, provider="Bedrock OpenAI")

        api_key = self.provider_config.api_key
        if not api_key:
            result.valid = False
            result.add_check("API Key", False, "OPENAI_API_KEY is not set")
            result.add_suggestion(
                "Export OPENAI_API_KEY=<your-amazon-bedrock-api-key>; generate "
                "one in the Amazon Bedrock console (API keys)."
            )
            return result

        if api_key in ("your-bedrock-api-key-here", "placeholder"):
            result.valid = False
            result.add_check(
                "API Key", False, "OPENAI_API_KEY appears to be a placeholder"
            )
            return result

        if len(api_key) < 20:
            result.add_warning("API key seems unusually short. Verify it's correct.")
        result.add_check("API Key", True, "API key configured")

        if not self.provider_config.base_url:
            result.valid = False
            result.add_check("Base URL", False, "OPENAI_BASE_URL is not set")
            result.add_suggestion(
                "Export OPENAI_BASE_URL to your region's Bedrock OpenAI endpoint."
            )
            return result

        if not self.provider_config.base_url.startswith("https://"):
            result.valid = False
            result.add_check("Base URL", False, "base_url must use HTTPS")
            return result

        result.add_check("Base URL", True, f"Endpoint: {self.provider_config.base_url}")

        wire_model = self.model_config.full_id or self.model_config.id
        result.add_check("Model", True, f"Model: {wire_model}")

        return result
