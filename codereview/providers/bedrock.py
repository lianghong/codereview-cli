"""AWS Bedrock provider implementation."""

from typing import Any

from botocore.config import Config as BotocoreConfig  # type: ignore[import-untyped]
from botocore.exceptions import (  # type: ignore[import-untyped]
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler

# Import system prompt from config
from codereview.config.models import BedrockConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import (
    ModelProvider,
    RetryConfig,
    ValidationResult,
)
from codereview.providers.mixins import TokenTrackingMixin

# Botocore transport failures that clear on their own. botocore's internal
# retries are switched off (``retries={"max_attempts": 0}``) so this provider's
# retry loop owns every attempt — which means these have to be named here or a
# DNS blip aborts a whole batch on the first try.
BOTOCORE_TRANSIENT_ERRORS = (
    ConnectTimeoutError,
    ReadTimeoutError,
    EndpointConnectionError,
    ConnectionClosedError,
)

# Bedrock/AWS error codes worth another attempt. Throttling clears in a couple
# of seconds; the rest are the service reporting a fault on its own side.
# Config errors (AccessDenied, Validation, ResourceNotFound) are absent on
# purpose — retrying those only delays the message the user needs to read.
RETRYABLE_BEDROCK_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "InternalServerException",
        "InternalFailure",
        "ServiceInternalError",
        "ModelTimeoutException",
        "ModelNotReadyException",
        "RequestTimeout",
        "RequestTimeoutException",
    }
)

# Cross-region inference-profile prefixes. An inference profile id is the base
# foundation-model id with a routing prefix; ``ListFoundationModels`` returns the
# *base* ids only, so the prefix has to come off before comparing. Every prefix
# AWS defines is listed — stripping only ``global.`` left ``us.``-prefixed ids
# (which is most of our registry) matching by luck, via a substring test.
CROSS_REGION_PREFIXES = ("global.", "us-gov.", "us.", "eu.", "apac.", "jp.", "au.")


def strip_cross_region_prefix(model_id: str) -> str:
    """Return *model_id* without its cross-region inference-profile prefix."""
    for prefix in CROSS_REGION_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


class BedrockProvider(TokenTrackingMixin, ModelProvider):
    """AWS Bedrock implementation of ModelProvider."""

    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: BedrockConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        """Initialize Bedrock provider.

        Args:
            model_config: Model configuration with pricing and inference params
            provider_config: Bedrock-specific configuration (region, etc.)
            temperature: Override temperature (uses model default if None)
            requests_per_second: Rate limit for API calls (default: 1.0)
            callbacks: Optional list of callback handlers for streaming/progress
            enable_output_fixing: Enable automatic retry on malformed output (default: True)
            project_context: Optional project README/documentation content for context
        """
        self.callbacks = callbacks or []
        self.enable_output_fixing = enable_output_fixing
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # Region-restricted models (e.g. Fable 5's geo-US profile, which
        # also needs the per-region provider_data_share opt-in) carry their
        # own region; everything else uses the provider-level default.
        self.region = model_config.region or provider_config.region

        # Determine temperature; allow_none preserves opt-out for reasoning
        # models (e.g. Opus 5) that set inference_params.temperature = None.
        self.temperature = self._resolve_temperature(
            override=temperature,
            model_config=model_config,
            provider_default=0.1,
            allow_none=True,
        )

        # Get model-specific inference parameters
        self.top_p = None
        self.top_k = None
        self.max_tokens = 16000  # Default

        if model_config.inference_params:
            self.top_p = model_config.inference_params.top_p
            self.top_k = model_config.inference_params.top_k
            if model_config.inference_params.max_output_tokens:
                self.max_tokens = model_config.inference_params.max_output_tokens

        # Token tracking (from mixin)
        self._init_token_tracking()

        # Rate limiter for API calls
        self.rate_limiter = self._build_rate_limiter(requests_per_second)

        # Create LangChain model and chain
        self.model = self._create_model()
        self.chain = self._create_chain()

    def _create_model(self) -> Any:
        """Create LangChain Bedrock model with structured output."""
        # Ensure full_id is present for Bedrock models
        if not self.model_config.full_id:
            raise ValueError(
                f"Bedrock model {self.model_config.id} missing required full_id"
            )

        # Build additional model request fields
        additional_fields: dict = {}
        if self.top_p is not None:
            additional_fields["top_p"] = self.top_p
        if self.top_k is not None:
            additional_fields["top_k"] = self.top_k

        # Configure botocore with timeout settings. Models with always-on
        # thinking (e.g. Fable 5) stream nothing until the full response is
        # generated, so think-heavy batches outlast the provider default;
        # they carry their own read_timeout.
        botocore_config = BotocoreConfig(
            read_timeout=self.model_config.read_timeout
            or self.provider_config.read_timeout,
            connect_timeout=self.provider_config.connect_timeout,
            retries={"max_attempts": 0},  # We handle retries ourselves
        )

        # Build model kwargs - omit temperature for reasoning models
        model_kwargs: dict = {
            "model": self.model_config.full_id,
            "region_name": self.region,
            "max_tokens": self.max_tokens,
            "config": botocore_config,
            "rate_limiter": self.rate_limiter,
            "callbacks": self.callbacks if self.callbacks else None,
            "additional_model_request_fields": (
                additional_fields if additional_fields else None
            ),
            # State the non-streaming choice instead of inheriting it. This
            # provider never passes `streaming=True`, so `_should_stream` is
            # already False for every entry and the value changes no behavior —
            # but langchain-aws 1.6.3 added a `logger.warning` whenever it has
            # to *infer* `disable_streaming`, and it fires for any model absent
            # from its hardcoded streaming allowlist. `claude-opus-5` is absent
            # (the list has claude-opus-4 / fable-5 / sonnet-5), so the CLI's
            # own default model printed a paragraph of upstream advice above
            # the Rich UI on every run, as did kimi-k2.5 / minimax-m2.5 / glm-5.
            # Passing any explicit value suppresses it (upstream gates the
            # warning on the key being absent), and the explicit `True` is also
            # what the `read_timeout: 1800` overrides on fable5/opus5 assume:
            # non-streaming Converse emits no bytes until generation completes.
            # Do NOT change this to False to "enable streaming" without a live
            # run — ConverseStream is a different wire path.
            "disable_streaming": True,
        }

        # Only add temperature if model supports it (reasoning models don't)
        if self.temperature is not None:
            model_kwargs["temperature"] = self.temperature

        base_model = ChatBedrockConverse(**model_kwargs)

        # Tool-use vs prompt-parsing routing (and _create_chain) live in the
        # base class; supports_tool_use in models.yaml decides the path.
        return self._apply_structured_output(base_model)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is a retryable AWS throttling or transport failure.

        botocore's own retries are disabled (``max_attempts: 0`` in
        ``_create_model``) so that this loop owns every attempt — which means
        the transport failures botocore would normally absorb have to be named
        here or they abort the batch on attempt 1. A read timeout on a
        think-heavy batch, a DNS blip, or a Bedrock 503 is exactly the
        transient class the retry framework exists for; treating only
        throttling as retryable threw away a whole batch's work (and the tokens
        already spent on it) on a failure that clears by itself.

        Retryable:
        - throttling: ``ThrottlingException`` / ``TooManyRequestsException``
        - service-side transients: ``ServiceUnavailableException``,
          ``InternalServerException``, ``ModelTimeoutException``,
          ``ModelNotReadyException``, plus any 5xx ``ClientError``
        - transport: connect/read timeouts, endpoint connection errors

        Deliberately NOT retryable: ``AccessDeniedException``,
        ``ValidationException``, ``ResourceNotFoundException`` — a config
        problem that retrying only makes slower.
        """
        if isinstance(error, BOTOCORE_TRANSIENT_ERRORS):
            return True
        if isinstance(error, ClientError):
            error_code = error.response.get("Error", {}).get("Code", "")
            if error_code in RETRYABLE_BEDROCK_ERROR_CODES:
                return True
            status_code = error.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode", 0
            )
            # A 5xx is the server's own admission that the failure is on its
            # side; the code list above can't enumerate every service's naming.
            return isinstance(status_code, int) and 500 <= status_code < 600
        return False

    @classmethod
    def supports_token_streaming(cls) -> bool:
        """False — ``_create_model`` passes ``disable_streaming=True``.

        Every Converse call here is non-streaming, so ``--stream`` would render
        nothing token-by-token while still dropping the run to one worker. The
        flag is also load-bearing for the ``read_timeout: 1800`` overrides on
        the always-thinking entries, so this can't flip without a live run on
        the ConverseStream wire path.
        """
        return False

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from a Bedrock Converse response.

        Read ``AIMessage.usage_metadata``, not ``response_metadata["usage"]``.
        Two independent reasons the latter never worked:

        1. ``langchain_aws``'s ``_extract_usage_metadata`` **pops** ``usage``
           off the raw response before ``_extract_response_metadata`` runs, so
           the key is gone by the time it reaches ``response_metadata``.
        2. Converse itself spells the fields ``inputTokens``/``outputTokens``
           (camelCase); ``input_tokens``/``output_tokens`` is LangChain's
           normalized ``usage_metadata`` spelling.

        Both misses are silent — ``.get(..., 0)`` returns zeros — so every
        Bedrock run reported 0 tokens and $0.0000 while being billed in full.
        The camelCase fallback covers a raw Converse dict reaching us
        unnormalized (e.g. a hand-built response or a future client change).
        """
        usage = getattr(result, "usage_metadata", None)
        if isinstance(usage, dict):
            return (usage.get("input_tokens", 0), usage.get("output_tokens", 0))

        metadata = getattr(result, "response_metadata", None)
        if isinstance(metadata, dict):
            raw = metadata.get("usage") or {}
            if isinstance(raw, dict):
                return (
                    raw.get("inputTokens", raw.get("input_tokens", 0)) or 0,
                    raw.get("outputTokens", raw.get("output_tokens", 0)) or 0,
                )
        return (0, 0)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int | None = None,
    ) -> CodeReviewReport:
        """Analyze a batch of files using AWS Bedrock.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            max_retries: Maximum number of retries for rate limiting (None uses
                this provider's default of 3 — Bedrock throttling clears fast)

        Returns:
            CodeReviewReport with findings

        Raises:
            ClientError: If AWS API call fails after all retries
        """
        retries = self._resolve_max_retries(max_retries, self.provider_config, 3)

        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content, self.project_context
        )

        chain_input = {
            "system_prompt": self._build_batch_system_prompt(files_content),
            "batch_context": batch_context,
        }

        retry_config = RetryConfig(max_retries=retries, base_wait=1.0)
        return self._execute_with_retry(chain_input, retry_config, batch_context)

    def validate_credentials(self) -> ValidationResult:
        """Validate AWS credentials and Bedrock access.

        Checks:
        1. AWS credentials are configured
        2. Can access AWS STS (identity check)
        3. Bedrock model is accessible in region

        Returns:
            ValidationResult with check details
        """
        import boto3  # type: ignore[import-untyped]

        result = ValidationResult(valid=True, provider="AWS Bedrock")

        # Check 1: AWS credentials configured
        try:
            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials is None:
                result.valid = False
                result.add_check(
                    "AWS Credentials",
                    False,
                    "No AWS credentials found",
                )
                result.add_suggestion("Run 'aws configure' to set up credentials")
                result.add_suggestion(
                    "Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
                )
                result.add_suggestion("Or use --aws-profile flag to specify a profile")
                return result

            result.add_check("AWS Credentials", True, "Credentials found")

        except Exception as e:
            result.valid = False
            result.add_check(
                "AWS Credentials", False, f"Error checking credentials: {e}"
            )
            return result

        # Check 2: STS identity (validates credentials work)
        try:
            sts = session.client("sts", region_name=self.region)
            identity = sts.get_caller_identity()
            account_id = identity.get("Account", "unknown")
            result.add_check(
                "AWS Identity",
                True,
                f"Authenticated as account {account_id}",
            )

        except ClientError as e:
            # Surface only the AWS error code, never the raw `Message`.
            # Bedrock error messages can include SCP fragments, ARNs, and
            # explicit-deny details that reveal IAM policy structure to
            # whoever runs `--validate`. The error code alone is enough
            # for troubleshooting; the suggestions below cover the common
            # codes.
            error_code = e.response.get("Error", {}).get("Code", "")
            result.valid = False
            result.add_check(
                "AWS Identity",
                False,
                f"STS error ({error_code})",
            )
            if error_code == "ExpiredToken":
                result.add_suggestion("Your AWS session token has expired. Refresh it.")
            elif error_code == "InvalidClientTokenId":
                result.add_suggestion("Your AWS access key ID is invalid.")
            return result

        except Exception as e:
            # Same redaction reasoning as the ClientError branch above.
            result.valid = False
            result.add_check("AWS Identity", False, f"Error: {type(e).__name__}")
            return result

        # Check 3: Bedrock model access
        try:
            bedrock = session.client(
                "bedrock",
                region_name=self.region,
            )

            # List foundation models to check access
            response = bedrock.list_foundation_models(
                byOutputModality="TEXT",
            )

            # Drop summaries with no modelId — an id-less entry carries no
            # information, and under the old substring test "" matched
            # everything and reported every model as available.
            model_id = self.model_config.full_id or ""
            available_models = {
                model_summary_id
                for m in response.get("modelSummaries", [])
                if (model_summary_id := m.get("modelId", ""))
            }

            # ListFoundationModels returns base foundation-model ids, so compare
            # against the id with its cross-region routing prefix removed.
            base_model_id = strip_cross_region_prefix(model_id)

            # Exact match, not substring. The predicate used to be
            # ``any(base in m or m in base for m in available)``, which confirms
            # access whenever either id is a prefix/infix of the other — so a
            # *version* difference read as a match: with only
            # ``minimax.minimax-m2`` and ``minimax.minimax-m2.1`` enabled in the
            # account, ``minimax.minimax-m2.5`` reported a green "Model Access"
            # check (verified against the live us-west-2 catalog), and the run
            # then failed with AccessDeniedException on the first real call.
            # `zai.glm-5` against a catalog holding only `zai.glm-5.2` did the
            # same. That is the one failure mode --validate exists to catch, and
            # a false green is worse than the inconclusive warning below: the
            # warning at least says "could not confirm".
            #
            # This check is deliberately allowed to be inconclusive rather than
            # wrong. It only reads the catalog, which lists what the *region*
            # offers, not what this account has been granted, so a real match is
            # necessary-but-not-sufficient for access — hence the miss path is a
            # warning, never a hard failure.
            model_found = bool(base_model_id) and base_model_id in available_models

            if model_found:
                result.add_check(
                    "Model Access",
                    True,
                    f"Model {self.model_config.name} is available",
                )
            else:
                result.add_warning(
                    f"Could not confirm model '{self.model_config.name}' access. "
                    "It may still work if enabled in Bedrock console."
                )
                result.add_suggestion(
                    f"Ensure '{self.model_config.name}' is enabled in AWS Bedrock console "
                    f"for region {self.region}"
                )

        except ClientError as e:
            # AWS error messages can leak SCP details / ARNs; surface only
            # the error code (same reasoning as STS branch above).
            error_code = e.response.get("Error", {}).get("Code", "")

            if error_code == "AccessDeniedException":
                result.add_warning(
                    "Cannot list Bedrock models (AccessDeniedException). "
                    "Model may still work if you have InvokeModel permission."
                )
                result.add_suggestion(
                    "Ensure IAM policy includes 'bedrock:ListFoundationModels' "
                    "for pre-flight validation"
                )
            else:
                result.add_warning(f"Bedrock check warning ({error_code})")

        except Exception as e:
            result.add_warning(f"Could not verify Bedrock access: {type(e).__name__}")

        return result
