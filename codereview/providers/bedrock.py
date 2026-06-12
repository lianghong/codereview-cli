"""AWS Bedrock provider implementation."""

from typing import Any

from botocore.config import Config as BotocoreConfig  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
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
        # models (e.g. Opus 4.7) that set inference_params.temperature = None.
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
        }

        # Only add temperature if model supports it (reasoning models don't)
        if self.temperature is not None:
            model_kwargs["temperature"] = self.temperature

        base_model = ChatBedrockConverse(**model_kwargs)

        # Tool-use vs prompt-parsing routing (and _create_chain) live in the
        # base class; supports_tool_use in models.yaml decides the path.
        return self._apply_structured_output(base_model)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is a retryable AWS throttling error."""
        if isinstance(error, ClientError):
            error_code = error.response.get("Error", {}).get("Code", "")
            return error_code in ["ThrottlingException", "TooManyRequestsException"]
        return False

    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """Extract token usage from AWS Bedrock response metadata."""
        if hasattr(result, "response_metadata"):
            usage = result.response_metadata.get("usage", {})
            return (
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            )
        return (0, 0)

    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """Analyze a batch of files using AWS Bedrock.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            max_retries: Maximum number of retries for rate limiting

        Returns:
            CodeReviewReport with findings

        Raises:
            ClientError: If AWS API call fails after all retries
        """
        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content, self.project_context
        )

        chain_input = {
            "system_prompt": self._build_batch_system_prompt(files_content),
            "batch_context": batch_context,
        }

        retry_config = RetryConfig(max_retries=max_retries, base_wait=1.0)
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

            # Check if our model is in the list
            model_id = self.model_config.full_id or ""
            available_models = [
                m.get("modelId", "") for m in response.get("modelSummaries", [])
            ]

            # For cross-region inference, check base model ID
            base_model_id: str = model_id
            if model_id.startswith("global."):
                # Extract base model from global inference ID
                # e.g., "global.anthropic.claude-opus-4-6-v1"
                # -> "anthropic.claude-opus-4-6-v1"
                parts = model_id.split(".", 1)
                if len(parts) > 1:
                    base_model_id = parts[1]

            # Check if model or a variant is available
            model_found = any(
                base_model_id in m or m in base_model_id for m in available_models
            )

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
