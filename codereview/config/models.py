"""Pydantic data models for configuration validation."""

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
        description="Use OpenAI Responses API instead of ChatCompletion (required for some models like GPT-5.2 Codex)",
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
        str, ProviderConfig | BedrockConfig | AzureOpenAIConfig | NVIDIAConfig
    ] = Field(default_factory=dict, description="Provider configurations")
