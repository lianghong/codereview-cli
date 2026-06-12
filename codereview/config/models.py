"""Pydantic data models for configuration validation."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class PricingConfig(BaseModel):
    """Pricing configuration for model API calls.

    Attributes:
        input_per_million: Cost per million input tokens in USD.
        output_per_million: Cost per million output tokens in USD.
    """

    model_config = {"frozen": True}

    input_per_million: float = Field(
        ..., ge=0, description="Cost per million input tokens in USD"
    )
    output_per_million: float = Field(
        ..., ge=0, description="Cost per million output tokens in USD"
    )


class InferenceParams(BaseModel):
    """Optional inference parameters for model configuration.

    Attributes:
        temperature: Sampling temperature (0.0-2.0). Higher values make output more random.
        top_p: Nucleus sampling parameter (0.0-1.0). Considers tokens with cumulative probability.
        top_k: Top-k sampling parameter (>=0). Considers only the k most likely tokens.
        max_output_tokens: Maximum number of tokens to generate (>0).
        enable_thinking: Enable thinking/reasoning mode (model-specific).
        clear_thinking: Clear thinking content between turns (False preserves reasoning).
        thinking: Thinking mode selector (e.g. DeepSeek-V4-Pro: False/'high'/'max').
        reasoning_effort: Per-request reasoning budget (Mistral Medium 3.5 and similar).
    """

    model_config = {"frozen": True}

    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )
    top_p: float | None = Field(
        None, ge=0.0, le=1.0, description="Nucleus sampling parameter (0.0-1.0)"
    )
    top_k: int | None = Field(None, ge=0, description="Top-k sampling parameter (>=0)")
    max_output_tokens: int | None = Field(
        None, gt=0, description="Maximum number of tokens to generate"
    )
    enable_thinking: bool | None = Field(
        None, description="Enable thinking/reasoning mode (model-specific)"
    )
    clear_thinking: bool | None = Field(
        None,
        description="Clear thinking content between turns (False preserves reasoning)",
    )
    thinking: bool | str | None = Field(
        None,
        description=(
            "Thinking mode selector passed as chat_template_kwargs['thinking']. "
            "Used by DeepSeek-V4-Pro (False=Non-think, 'high'=Think High, "
            "'max'=Think Max)."
        ),
    )
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = Field(
        None,
        description=(
            "Reasoning effort budget sent with each request. Used by models "
            "like Mistral Medium 3.5 that accept 'none'/'high' to toggle "
            "between instant reply and deep reasoning."
        ),
    )


class ModelConfig(BaseModel):
    """Configuration for a single model.

    Attributes:
        id: Short identifier for CLI usage (e.g., 'opus', 'sonnet').
        name: Human-readable display name.
        aliases: Alternative names for CLI (optional).
        pricing: Pricing configuration for input/output tokens.
        inference_params: Default inference parameters (optional).
        full_id: Full model ID for API calls (provider-specific, optional).
        deployment_name: Deployment name for Azure (optional).
        use_responses_api: Use OpenAI Responses API instead of ChatCompletion (Azure, optional).
        supports_tool_use: Whether the model supports tool/function calling (default: True).
        region: Per-model AWS region override for region-restricted Bedrock models (optional).
    """

    model_config = {"frozen": True}

    id: str = Field(..., min_length=1, description="Short identifier for CLI usage")
    name: str = Field(..., min_length=1, description="Human-readable display name")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names for CLI"
    )
    pricing: PricingConfig = Field(..., description="Token pricing configuration")
    inference_params: InferenceParams | None = Field(
        None, description="Default inference parameters"
    )
    full_id: str | None = Field(
        None,
        min_length=1,
        description="Full model ID for API calls (provider-specific)",
    )
    deployment_name: str | None = Field(
        None, min_length=1, description="Deployment name for Azure"
    )
    use_responses_api: bool | None = Field(
        None,
        description="Use OpenAI Responses API instead of ChatCompletion (required for some models like GPT-5.3 Codex)",
    )
    supports_tool_use: bool = Field(
        True,
        description="Whether the model supports tool/function calling for structured output",
    )
    context_window: int | None = Field(
        None, gt=0, description="Maximum context window size in tokens"
    )
    region: str | None = Field(
        None,
        min_length=1,
        description=(
            "Per-model AWS region override for region-restricted Bedrock "
            "models (e.g. Fable 5's geo-US profile is us-east-1 only); "
            "falls back to the provider-level region when unset"
        ),
    )


class ProviderConfig(BaseModel):
    """Base configuration for a model provider.

    Attributes:
        models: List of model configurations for this provider.
    """

    model_config = {"frozen": True}

    models: list[ModelConfig] = Field(
        default_factory=list, description="List of model configurations"
    )


class BedrockConfig(ProviderConfig):
    """Configuration for AWS Bedrock provider.

    Attributes:
        region: AWS region for Bedrock API calls.
        read_timeout: Read timeout in seconds for API calls.
        connect_timeout: Connection timeout in seconds for API calls.
        models: List of model configurations for Bedrock.
    """

    model_config = {"frozen": True}

    region: str = Field(
        default="us-west-2",
        min_length=1,
        description="AWS region for Bedrock API calls",
    )
    read_timeout: int = Field(
        default=300,
        gt=0,
        description="Read timeout in seconds for API calls (default: 5 minutes)",
    )
    connect_timeout: int = Field(
        default=60,
        gt=0,
        description="Connection timeout in seconds for API calls",
    )


class AzureOpenAIConfig(ProviderConfig):
    """Configuration for Azure OpenAI provider.

    Attributes:
        endpoint: Azure OpenAI endpoint URL.
        api_key: Azure OpenAI API key or reference to environment variable.
        api_version: Azure OpenAI API version.
        request_timeout: Request timeout in seconds for API calls.
        models: List of model configurations for Azure.
    """

    model_config = {"frozen": True}

    endpoint: HttpUrl = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(
        ...,
        min_length=1,
        description="Azure OpenAI API key or environment variable reference",
    )
    api_version: str = Field(..., min_length=1, description="Azure OpenAI API version")
    request_timeout: int = Field(
        default=300,
        gt=0,
        description="Request timeout in seconds for API calls (default: 5 minutes)",
    )


class NVIDIAConfig(ProviderConfig):
    """Configuration for NVIDIA NIM API provider.

    Attributes:
        api_key: NVIDIA API key (from build.nvidia.com).
        base_url: Optional base URL for self-hosted NIMs.
        polling_timeout: Timeout in seconds for 202 polling (waiting for async results).
        max_retries: Maximum retries for gateway errors (504/502/503).
        models: List of model configurations for NVIDIA.
    """

    model_config = {"frozen": True}

    api_key: str = Field(
        ...,
        min_length=1,
        description="NVIDIA API key from build.nvidia.com",
    )
    base_url: str | None = Field(
        None,
        description="Optional base URL for self-hosted NIMs (leave empty for cloud)",
    )
    polling_timeout: int = Field(
        default=900,
        gt=0,
        description="Timeout in seconds for 202 polling (default: 15 minutes)",
    )
    max_retries: int = Field(
        default=5,
        ge=0,
        description="Maximum retries for gateway errors (504/502/503)",
    )


class GoogleGenAIConfig(ProviderConfig):
    """Configuration for Google Generative AI provider.

    Attributes:
        api_key: Google AI API key (from Google AI Studio).
        request_timeout: Request timeout in seconds for API calls.
        models: List of model configurations for Google GenAI.
    """

    model_config = {"frozen": True}

    api_key: str = Field(
        ...,
        min_length=1,
        description="Google AI API key from Google AI Studio",
    )
    request_timeout: int = Field(
        default=300,
        gt=0,
        description="Request timeout in seconds for API calls (default: 5 minutes)",
    )


class MoonshotConfig(ProviderConfig):
    """Configuration for Moonshot AI (Kimi) direct API.

    Uses the dedicated ``langchain-moonshot`` package's ``ChatMoonshot``
    client. The package itself reads ``MOONSHOT_API_KEY`` by default; this
    config plumbs the ``KIMI_API_KEY`` env var through explicitly so the
    naming matches the existing CLI conventions (KIMI_API_KEY).

    Attributes:
        api_key: Moonshot API key, read from KIMI_API_KEY env var.
        base_url: Base URL for Moonshot's API.
        request_timeout: Request timeout in seconds for API calls.
        models: List of model configurations.
    """

    model_config = {"frozen": True}

    api_key: str = Field(
        ...,
        min_length=1,
        description="Moonshot API key (read from KIMI_API_KEY env var)",
    )
    base_url: str = Field(
        default="https://api.moonshot.cn/v1",
        description=(
            "Base URL for Moonshot's API. Defaults to the Chinese platform "
            "(platform.moonshot.cn) which is what KIMI_API_KEY usually targets. "
            "Set to https://api.moonshot.ai/v1 for the international platform."
        ),
    )
    request_timeout: int = Field(
        default=300,
        gt=0,
        description="Request timeout in seconds for API calls (default: 5 minutes)",
    )


class DeepSeekConfig(ProviderConfig):
    """Configuration for DeepSeek's direct API.

    Uses the dedicated ``langchain-deepseek`` package's ``ChatDeepSeek``
    client (small single-purpose dep, not the heavy langchain-community).
    DeepSeek's API is OpenAI-format compatible at https://api.deepseek.com.

    Attributes:
        api_key: DeepSeek API key, read from DEEPSEEK_API_KEY env var.
        api_base: Base URL for DeepSeek's API.
        request_timeout: Request timeout in seconds for API calls.
        models: List of model configurations.
    """

    model_config = {"frozen": True}

    api_key: str = Field(
        ...,
        min_length=1,
        description="DeepSeek API key (read from DEEPSEEK_API_KEY env var)",
    )
    api_base: str = Field(
        default="https://api.deepseek.com",
        description="Base URL for DeepSeek's OpenAI-compatible API",
    )
    request_timeout: int = Field(
        default=300,
        gt=0,
        description="Request timeout in seconds for API calls (default: 5 minutes)",
    )


class ZAIConfig(ProviderConfig):
    """Configuration for Z.AI (Zhipu international) provider.

    Z.AI exposes an OpenAI-compatible API; this config is reused by the
    ChatOpenAI client wired with a custom base_url. The Chinese Zhipu
    endpoint (open.bigmodel.cn) is intentionally NOT configured here
    because it has different pricing, region constraints, and uses a
    different env var (ZHIPUAI_API_KEY).

    Attributes:
        api_key: Z.AI API key (from z.ai), read from ZAI_API_KEY env var.
        base_url: OpenAI-compatible base URL. Defaults to Z.AI international.
        request_timeout: Request timeout in seconds for API calls.
        models: List of model configurations for Z.AI.
    """

    model_config = {"frozen": True}

    api_key: str = Field(
        ...,
        min_length=1,
        description="Z.AI API key (read from ZAI_API_KEY env var)",
    )
    base_url: str = Field(
        default="https://api.z.ai/api/paas/v4/",
        description="OpenAI-compatible base URL (defaults to Z.AI international)",
    )
    request_timeout: int = Field(
        default=300,
        gt=0,
        description="Request timeout in seconds for API calls (default: 5 minutes)",
    )


class BedrockOpenAIConfig(ProviderConfig):
    """Configuration for OpenAI models hosted on Amazon Bedrock.

    AWS exposes OpenAI's frontier models (GPT-5.5, GPT-5.4, Codex) on Amazon
    Bedrock through an OpenAI-compatible surface. Unlike every other Bedrock
    model in this project — which goes through ``ChatBedrockConverse`` and the
    AWS SigV4 credential chain — the OpenAI-compatible endpoint authenticates
    with an **Amazon Bedrock API key** (a bearer token) and is driven with
    langchain-openai's ``ChatOpenAI`` pointed at a custom ``base_url`` (the same
    mechanism as the Z.AI provider). It is therefore configured as its own
    provider, NOT under ``bedrock``.

    The Bedrock API key is read from ``OPENAI_API_KEY`` and the endpoint from
    ``OPENAI_BASE_URL`` — the canonical env vars the underlying ``openai`` SDK
    reads by default — so an existing OpenAI-SDK setup works by changing only
    the base URL and key.

    Attributes:
        api_key: Amazon Bedrock API key (bearer token), read from OPENAI_API_KEY.
        base_url: OpenAI-compatible Bedrock endpoint, read from OPENAI_BASE_URL.
        request_timeout: Request timeout in seconds for API calls.
        models: List of model configurations for Bedrock-hosted OpenAI models.
    """

    model_config = {"frozen": True}

    api_key: str = Field(
        ...,
        min_length=1,
        description="Amazon Bedrock API key (bearer token, read from OPENAI_API_KEY)",
    )
    base_url: str = Field(
        ...,
        min_length=1,
        description="OpenAI-compatible Bedrock endpoint (read from OPENAI_BASE_URL)",
    )
    request_timeout: int = Field(
        default=300,
        gt=0,
        description="Request timeout in seconds for API calls (default: 5 minutes)",
    )


class ScanningConfig(BaseModel):
    """Configuration for file scanning.

    Attributes:
        max_file_size_kb: Maximum file size in KB to include in analysis.
        warn_file_size_kb: File size threshold for warnings.
        exclude_patterns: Glob patterns to exclude from scanning.
        exclude_extensions: File extensions to exclude from scanning.
    """

    model_config = {"frozen": True}

    max_file_size_kb: int = Field(
        default=500, gt=0, description="Maximum file size in KB to include"
    )
    warn_file_size_kb: int = Field(
        default=100, gt=0, description="File size threshold for warnings"
    )
    exclude_patterns: list[str] = Field(
        default_factory=list, description="Glob patterns to exclude"
    )
    exclude_extensions: list[str] = Field(
        default_factory=list, description="File extensions to exclude"
    )


class ModelsConfigFile(BaseModel):
    """Root configuration model for models.yaml file.

    Attributes:
        providers: Dictionary mapping provider names to their configurations.
    """

    model_config = {"frozen": True}

    providers: dict[
        str,
        ProviderConfig
        | BedrockConfig
        | AzureOpenAIConfig
        | NVIDIAConfig
        | GoogleGenAIConfig
        | DeepSeekConfig
        | MoonshotConfig
        | ZAIConfig
        | BedrockOpenAIConfig,
    ] = Field(default_factory=dict, description="Provider configurations")
