"""AWS Bedrock provider implementation."""

import time
from typing import Any

from botocore.config import Config as BotocoreConfig  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import ValidationError

# Import system prompt from config
from codereview.config import SYSTEM_PROMPT
from codereview.config.models import BedrockConfig, ModelConfig
from codereview.models import CodeReviewReport
from codereview.providers.base import ModelProvider, ValidationResult

# Shared prompt template for consistent formatting
BATCH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("human", "{batch_context}"),
    ]
)


class BedrockProvider(ModelProvider):
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
        """
        self.callbacks = callbacks or []
        self.enable_output_fixing = enable_output_fixing
        self._output_parser = PydanticOutputParser(pydantic_object=CodeReviewReport)
        self.model_config = model_config
        self.provider_config = provider_config
        self.project_context = project_context

        # Determine temperature (override > model default > 0.1)
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(
                    f"Temperature must be between 0.0 and 2.0, got {temperature}"
                )
            self.temperature = temperature
        elif (
            model_config.inference_params
            and model_config.inference_params.temperature is not None
        ):
            self.temperature = model_config.inference_params.temperature
        else:
            self.temperature = 0.1

        # Get model-specific inference parameters
        self.top_p = None
        self.top_k = None
        self.max_tokens = 16000  # Default

        if model_config.inference_params:
            self.top_p = model_config.inference_params.top_p
            self.top_k = model_config.inference_params.top_k
            if model_config.inference_params.max_output_tokens:
                self.max_tokens = model_config.inference_params.max_output_tokens

        # Token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Flag for prompt-based parsing (set in _create_model if model doesn't support tool use)
        self._use_prompt_parsing = False

        # Rate limiter for API calls
        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

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

        # Configure botocore with timeout settings
        botocore_config = BotocoreConfig(
            read_timeout=self.provider_config.read_timeout,
            connect_timeout=self.provider_config.connect_timeout,
            retries={"max_attempts": 0},  # We handle retries ourselves
        )

        base_model = ChatBedrockConverse(
            model=self.model_config.full_id,
            region_name=self.provider_config.region,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            config=botocore_config,
            rate_limiter=self.rate_limiter,
            callbacks=self.callbacks if self.callbacks else None,
            additional_model_request_fields=(
                additional_fields if additional_fields else None
            ),
        )

        # Check if model supports tool use for structured output
        if self.model_config.supports_tool_use:
            # Configure for structured output using tool calling
            return base_model.with_structured_output(CodeReviewReport)
        else:
            # Return base model for prompt-based JSON parsing
            self._use_prompt_parsing = True
            return base_model

    def _create_chain(self) -> Any:
        """Create LangChain chain with prompt template."""
        if self._use_prompt_parsing:
            # For models without tool use, add output parser to chain
            return BATCH_PROMPT_TEMPLATE | self.model | self._output_parser
        return BATCH_PROMPT_TEMPLATE | self.model

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

        # Build system prompt (add format instructions for prompt-based parsing)
        system_prompt = SYSTEM_PROMPT
        if self._use_prompt_parsing:
            # Add JSON format instructions for models without tool use
            format_instructions = self._output_parser.get_format_instructions()
            system_prompt = f"{SYSTEM_PROMPT}\n\n{format_instructions}"

        # Use chain with prompt template for cleaner invocation
        chain_input = {
            "system_prompt": system_prompt,
            "batch_context": batch_context,
        }

        last_error: ClientError | ValidationError | None = None
        for attempt in range(max_retries + 1):
            try:
                result = self.chain.invoke(chain_input)

                # Handle None result (structured output parsing failed)
                if result is None:
                    raise ValidationError.from_exception_data(
                        "Model returned None - structured output parsing failed",
                        [],
                    )

                # Track token usage from AWS Bedrock response metadata
                input_tokens = 0
                output_tokens = 0

                if hasattr(result, "response_metadata"):
                    usage = result.response_metadata.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

                # Fallback to estimation if actual counts unavailable
                if input_tokens == 0:
                    input_tokens = self._estimate_tokens(batch_context)
                if output_tokens == 0:
                    output_tokens = self._estimate_tokens(str(result.model_dump_json()))

                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens

                return result

            except ValidationError as e:
                # Output parsing/validation failed
                last_error = e

                if self.enable_output_fixing and attempt < max_retries:
                    # Try again - LangChain structured output will retry
                    time.sleep(1)
                    continue

                raise

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                last_error = e

                # Check if it's a throttling error (rate limiter handles most cases,
                # but we still need manual retry for edge cases)
                if error_code in ["ThrottlingException", "TooManyRequestsException"]:
                    if attempt < max_retries:
                        # Exponential backoff: 2^attempt seconds
                        wait_time = 2**attempt
                        time.sleep(wait_time)
                        continue

                # For other errors, raise immediately
                raise

        # If we exhausted all retries, last_error must be set
        # (loop only exits early on success via return)
        assert last_error is not None, "Retry loop exited without error or success"
        raise last_error

    def get_model_display_name(self) -> str:
        """Get human-readable model name."""
        return self.model_config.name

    def get_pricing(self) -> dict[str, float]:
        """Get pricing information for the model."""
        return {
            "input_price_per_million": self.model_config.pricing.input_per_million,
            "output_price_per_million": self.model_config.pricing.output_per_million,
        }

    def reset_state(self) -> None:
        """Reset token counters."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens used."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens used."""
        return self._total_output_tokens

    def estimate_cost(self) -> dict[str, float]:
        """Calculate cost from token usage."""
        pricing = self.model_config.pricing

        input_cost = (self._total_input_tokens / 1_000_000) * pricing.input_per_million
        output_cost = (
            self._total_output_tokens / 1_000_000
        ) * pricing.output_per_million

        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }

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
            sts = session.client("sts", region_name=self.provider_config.region)
            identity = sts.get_caller_identity()
            account_id = identity.get("Account", "unknown")
            result.add_check(
                "AWS Identity",
                True,
                f"Authenticated as account {account_id}",
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_msg = e.response.get("Error", {}).get("Message", "")
            result.valid = False
            result.add_check(
                "AWS Identity",
                False,
                f"STS error ({error_code}): {error_msg}",
            )
            if error_code == "ExpiredToken":
                result.add_suggestion("Your AWS session token has expired. Refresh it.")
            elif error_code == "InvalidClientTokenId":
                result.add_suggestion("Your AWS access key ID is invalid.")
            return result

        except Exception as e:
            result.valid = False
            result.add_check("AWS Identity", False, f"Error: {e}")
            return result

        # Check 3: Bedrock model access
        try:
            bedrock = session.client(
                "bedrock",
                region_name=self.provider_config.region,
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
                # e.g., "global.anthropic.claude-opus-4-5-20251101-v1:0"
                # -> "anthropic.claude-opus-4-5-20251101-v1:0"
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
                    f"for region {self.provider_config.region}"
                )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_msg = e.response.get("Error", {}).get("Message", "")

            if error_code == "AccessDeniedException":
                result.add_warning(
                    f"Cannot list Bedrock models: {error_msg}. "
                    "Model may still work if you have InvokeModel permission."
                )
                result.add_suggestion(
                    "Ensure IAM policy includes 'bedrock:ListFoundationModels' "
                    "for pre-flight validation"
                )
            else:
                result.add_warning(f"Bedrock check warning ({error_code}): {error_msg}")

        except Exception as e:
            result.add_warning(f"Could not verify Bedrock access: {e}")

        return result
