"""Provider for Bedrock's OpenAI-compatible ``bedrock-mantle`` endpoint.

AWS exposes several vendors' frontier models on Amazon Bedrock through an
OpenAI-compatible surface. This is a *different* path from every other Bedrock
model in this project: rather than ``ChatBedrockConverse`` and the AWS SigV4
credential chain, the OpenAI-compatible endpoint authenticates with an Amazon
Bedrock **API key** (a bearer token) and is driven with langchain-openai's
``ChatOpenAI`` pointed at a custom ``base_url`` — the same mechanism as the Z.AI
provider (``providers/zai.py``), which is why this mirrors it closely.

Despite the module name, this is **not OpenAI-only**: xAI's Grok rides the same
endpoint and lives here too. Consult ``models.yaml`` for the current entries
rather than trusting a list in this docstring — the registry is authoritative
and this text has gone stale before.

Underlying transport: ``ChatOpenAI`` → the ``openai`` SDK → Bedrock's endpoint.
Both packages are already dependencies (``langchain-openai`` pulls ``openai``),
so no new dependency is required. The Bedrock API key / endpoint are read from
the canonical ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` env vars.

Two capability axes vary *per entry*, and the code reads both off the model
config rather than assuming either:

* **Sampling params.** The GPT-5.x entries are reasoning models that reject
  ``temperature`` / ``top_p`` (handled via ``allow_none=True`` plus omitting
  ``default_temperature`` in the YAML). Grok accepts both.
* **Which API.** ``use_responses_api`` on the model entry selects the OpenAI
  **Responses API**, which the GPT-5.x entries require (they do not support Chat
  Completions here, and it is how chain-of-thought surfaces). Grok omits the
  flag and uses Chat Completions.

Every entry on this endpoint currently sets ``supports_tool_use: false``: they
engage server-side reasoning per request, and a think-heavy batch returns a
reasoning-only response (``tool_calls=[]``, no ``parsed``) that breaks the
forced ``tool_choice`` ``.with_structured_output()`` sets. The base class routes
them to prompt-based JSON parsing.
"""

import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from codereview.config.models import BedrockOpenAIConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import (
    CLIENT_RETRIES_DISABLED,
    TokenTrackingMixin,
    extract_openai_token_usage,
    is_blank,
    is_https_url,
    is_openai_retryable_error,
    is_placeholder_api_key,
    is_short_api_key,
    openai_stream_params,
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
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # The GPT-5.x tiers here are reasoning models and reject
        # temperature/top_p; allow_none preserves that opt-out (no
        # default_temperature in YAML). It stays a per-entry opt-out rather
        # than an unconditional drop because this endpoint also serves models
        # that accept both — Grok 4.3 did, at card defaults 0.7/0.95.
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

    def _resolve_base_url(self) -> str:
        """Point the client at the Region that actually serves this model.

        ``base_url`` is a *provider*-level setting, but ``bedrock-mantle``'s
        Regions are **per model and non-overlapping**: GPT-6 Astra is served
        from ``us-west-2`` only, GPT-5.6 Sol is In-Region ``us-east-1`` /
        ``us-east-2`` only. One ``OPENAI_BASE_URL`` therefore cannot reach both,
        and pointing at the wrong Region does not fall back — it returns a 404
        naming the model id ("The model 'openai.gpt-6-astra' does not exist")
        on every batch of the run. So a model entry may carry ``region:`` to
        override the Region component of the configured URL.

        Only the Region label in the host is rewritten, never the scheme, host
        pattern or path: the Region is what varies per the model card, and
        deriving from the configured value keeps ``OPENAI_BASE_URL``
        authoritative for everything else (a custom gateway, a future URL
        shape). A host with no Region label to swap is returned **unchanged** —
        fail soft, since refusing to run would break a working self-hosted
        endpoint over a cosmetic mismatch.
        """
        configured = self.provider_config.base_url
        region = self.model_config.region
        if not region:
            return configured

        parts = urlsplit(configured)
        host = parts.hostname or ""
        # An AWS Region label: two letters, one or more words, a digit —
        # us-west-2, eu-central-1, ap-southeast-4, us-gov-west-1. Matched
        # against a whole host component so a hostname that merely contains
        # such a substring is left alone.
        labels = host.split(".")
        for index, label in enumerate(labels):
            if re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-\d", label):
                if label == region:
                    return configured
                labels[index] = region
                netloc = ".".join(labels)
                if parts.port:
                    netloc = f"{netloc}:{parts.port}"
                rewritten = urlunsplit(
                    (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
                )
                logging.debug(
                    "Model %s declares region %s; using endpoint %s",
                    self.model_config.id,
                    region,
                    rewritten,
                )
                return rewritten

        logging.debug(
            "Model %s declares region %s but base_url %s has no Region label to "
            "rewrite; using it as configured.",
            self.model_config.id,
            region,
            configured,
        )
        return configured

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
            # Gate the URL actually used, so a per-model Region override can
            # never become a way to send the bearer key over cleartext.
            "base_url": require_https(self._resolve_base_url(), "base_url"),
            "api_key": SecretStr(str(self.provider_config.api_key)),
            "max_tokens": self.max_tokens,
            "rate_limiter": self.rate_limiter,
            "callbacks": self.callbacks if self.callbacks else None,
            "timeout": self.provider_config.request_timeout,
            # This provider's retry loop owns every attempt; see the constant.
            "max_retries": CLIENT_RETRIES_DISABLED,
            # streaming only for a handler that actually consumes tokens, and
            # stream_usage alongside it so the billed counts survive the
            # streaming path. Both halves live in openai_stream_params.
            **openai_stream_params(self.callbacks),
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

        # Tool-use vs prompt-based JSON parsing is decided once in the base
        # class from supports_tool_use; tool-use-less models here (currently
        # GPT-5.6 Sol's adaptive thinking) get the PydanticOutputParser path.
        return self._apply_structured_output(base_model)

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
                "Bedrock OpenAI backoff: waiting %.1fs (Retry-After header)", wait
            )
            return wait
        return min(config.base_wait * (2**attempt), config.max_wait)

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from the OpenAI-shaped response.

        The GPT-5.x entries here set ``use_responses_api``, whose converter
        populates only ``usage_metadata`` — the shared helper reads that first.
        """
        return extract_openai_token_usage(result)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int | None = None,
    ) -> CodeReviewReport:
        """Analyze a batch of files using an OpenAI model on Bedrock.

        ``max_retries=None`` uses this provider's default of 5.
        """
        retries = self._resolve_max_retries(max_retries, self.provider_config, 5)

        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content, self.project_context
        )

        chain_input = {
            "system_prompt": self._build_batch_system_prompt(files_content),
            "batch_context": batch_context,
        }

        retry_config = RetryConfig(max_retries=retries, base_wait=2.0)
        return self._execute_with_retry(chain_input, retry_config, batch_context)

    def validate_credentials(self) -> ValidationResult:
        """Validate Bedrock OpenAI configuration before any analysis call."""
        result = ValidationResult(valid=True, provider="Bedrock OpenAI")

        api_key = self.provider_config.api_key
        if is_blank(api_key):
            result.valid = False
            result.add_check("API Key", False, "OPENAI_API_KEY is not set")
            result.add_suggestion(
                "Export OPENAI_API_KEY=<your-amazon-bedrock-api-key>; generate "
                "one in the Amazon Bedrock console (API keys)."
            )
            return result

        # "<your-amazon-bedrock-api-key>" is the exact string README.md's export
        # line documents (angle brackets included) — per the CLAUDE.md contract
        # it must hard-fail --validate, not 401 on the first real call.
        if is_placeholder_api_key(
            api_key,
            (
                "your-bedrock-api-key-here",
                "<your-amazon-bedrock-api-key>",
                "your-amazon-bedrock-api-key",
            ),
        ):
            result.valid = False
            result.add_check(
                "API Key", False, "OPENAI_API_KEY appears to be a placeholder"
            )
            return result

        if is_short_api_key(api_key):
            result.add_warning("API key seems unusually short. Verify it's correct.")
        result.add_check("API Key", True, "API key configured")

        if is_blank(self.provider_config.base_url):
            result.valid = False
            result.add_check("Base URL", False, "OPENAI_BASE_URL is not set")
            result.add_suggestion(
                "Export OPENAI_BASE_URL to your region's Bedrock OpenAI endpoint."
            )
            return result

        if not is_https_url(self.provider_config.base_url):
            result.valid = False
            result.add_check(
                "Base URL", False, "base_url must be an HTTPS URL with a host"
            )
            return result

        result.add_check("Base URL", True, f"Endpoint: {self.provider_config.base_url}")

        wire_model = self.model_config.full_id or self.model_config.id
        result.add_check("Model", True, f"Model: {wire_model}")

        return result
