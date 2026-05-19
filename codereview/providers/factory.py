"""Factory for creating provider instances with auto-detection."""

from langchain_core.callbacks import BaseCallbackHandler

from codereview.config import ConfigLoader, get_config_loader
from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    DeepSeekConfig,
    GoogleGenAIConfig,
    MoonshotConfig,
    NVIDIAConfig,
    ZAIConfig,
)
from codereview.providers.base import ModelProvider


class ProviderFactory:
    """Factory for creating model provider instances.

    Automatically detects which provider to use based on model name
    (ID or alias) and instantiates the appropriate provider.
    """

    def __init__(self, config_loader: ConfigLoader | None = None):
        """Initialize factory.

        Args:
            config_loader: ConfigLoader instance (creates default if None)
        """
        self.config_loader = config_loader or get_config_loader()

    def create_provider(
        self,
        model_name: str,
        temperature: float | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        project_context: str | None = None,
    ) -> ModelProvider:
        """Create provider based on model name.

        Args:
            model_name: Model ID or alias (e.g., "opus", "gpt-5.4")
            temperature: Optional temperature override
            callbacks: Optional list of callback handlers for streaming/progress
            project_context: Optional project README/documentation content

        Returns:
            Instantiated provider (BedrockProvider, AzureOpenAIProvider, NVIDIAProvider, or GoogleGenAIProvider)

        Raises:
            ValueError: If model name not found or provider unknown
        """
        # Resolve model name to provider and config
        provider_name, model_config = self.config_loader.resolve_model(model_name)

        # Get provider-specific configuration
        provider_config = self.config_loader.get_provider_config(provider_name)

        # Create appropriate provider
        if provider_name == "bedrock":
            if not isinstance(provider_config, BedrockConfig):
                raise ValueError(
                    f"Expected BedrockConfig for bedrock provider, "
                    f"got {type(provider_config).__name__}"
                )
            from codereview.providers.bedrock import BedrockProvider

            return BedrockProvider(
                model_config,
                provider_config,
                temperature,
                callbacks=callbacks,
                project_context=project_context,
            )

        elif provider_name == "azure_openai":
            if not isinstance(provider_config, AzureOpenAIConfig):
                raise ValueError(
                    f"Expected AzureOpenAIConfig for azure_openai provider, "
                    f"got {type(provider_config).__name__}"
                )
            from codereview.providers.azure_openai import AzureOpenAIProvider

            return AzureOpenAIProvider(
                model_config,
                provider_config,
                temperature,
                callbacks=callbacks,
                project_context=project_context,
            )

        elif provider_name == "nvidia":
            if not isinstance(provider_config, NVIDIAConfig):
                raise ValueError(
                    f"Expected NVIDIAConfig for nvidia provider, "
                    f"got {type(provider_config).__name__}"
                )
            from codereview.providers.nvidia import NVIDIAProvider

            return NVIDIAProvider(
                model_config,
                provider_config,
                temperature,
                callbacks=callbacks,
                project_context=project_context,
            )

        elif provider_name == "google_genai":
            if not isinstance(provider_config, GoogleGenAIConfig):
                raise ValueError(
                    f"Expected GoogleGenAIConfig for google_genai provider, "
                    f"got {type(provider_config).__name__}"
                )
            from codereview.providers.google_genai import GoogleGenAIProvider

            return GoogleGenAIProvider(
                model_config,
                provider_config,
                temperature,
                callbacks=callbacks,
                project_context=project_context,
            )

        elif provider_name == "zai":
            if not isinstance(provider_config, ZAIConfig):
                raise ValueError(
                    f"Expected ZAIConfig for zai provider, "
                    f"got {type(provider_config).__name__}"
                )
            from codereview.providers.zai import ZAIProvider

            return ZAIProvider(
                model_config,
                provider_config,
                temperature,
                callbacks=callbacks,
                project_context=project_context,
            )

        elif provider_name == "deepseek":
            if not isinstance(provider_config, DeepSeekConfig):
                raise ValueError(
                    f"Expected DeepSeekConfig for deepseek provider, "
                    f"got {type(provider_config).__name__}"
                )
            from codereview.providers.deepseek import DeepSeekProvider

            return DeepSeekProvider(
                model_config,
                provider_config,
                temperature,
                callbacks=callbacks,
                project_context=project_context,
            )

        elif provider_name == "moonshot":
            if not isinstance(provider_config, MoonshotConfig):
                raise ValueError(
                    f"Expected MoonshotConfig for moonshot provider, "
                    f"got {type(provider_config).__name__}"
                )
            from codereview.providers.moonshot import MoonshotProvider

            return MoonshotProvider(
                model_config,
                provider_config,
                temperature,
                callbacks=callbacks,
                project_context=project_context,
            )

        else:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Supported providers: bedrock, azure_openai, nvidia, "
                f"google_genai, zai, deepseek, moonshot"
            )

    def list_available_models(self) -> dict[str, list[dict[str, str]]]:
        """List all available models grouped by provider.

        Returns:
            Dict mapping provider names to lists of model info dicts.
            Each model info dict contains: id, name, aliases
        """
        result: dict[str, list[dict[str, str]]] = {}

        models_by_provider = self.config_loader.list_models()

        for provider_name, model_configs in models_by_provider.items():
            result[provider_name] = []
            for model_config in model_configs:
                result[provider_name].append(
                    {
                        "id": model_config.id,
                        "name": model_config.name,
                        "aliases": ", ".join(model_config.aliases),
                    }
                )

        return result
