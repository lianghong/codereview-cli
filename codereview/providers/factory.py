"""Factory for creating provider instances with auto-detection."""

from langchain_core.callbacks import BaseCallbackHandler

from codereview.config.loader import ConfigLoader
from codereview.config.models import AzureOpenAIConfig, BedrockConfig, NVIDIAConfig
from codereview.providers.azure_openai import AzureOpenAIProvider
from codereview.providers.base import ModelProvider
from codereview.providers.bedrock import BedrockProvider
from codereview.providers.nvidia import NVIDIAProvider


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
        self.config_loader = config_loader or ConfigLoader()

    def create_provider(
        self,
        model_name: str,
        temperature: float | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        project_context: str | None = None,
    ) -> ModelProvider:
        """Create provider based on model name.

        Args:
            model_name: Model ID or alias (e.g., "opus", "gpt-5.2-codex")
            temperature: Optional temperature override
            callbacks: Optional list of callback handlers for streaming/progress
            project_context: Optional project README/documentation content

        Returns:
            Instantiated provider (BedrockProvider, AzureOpenAIProvider, or NVIDIAProvider)

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
            return NVIDIAProvider(
                model_config,
                provider_config,
                temperature,
                callbacks=callbacks,
                project_context=project_context,
            )

        else:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Supported providers: bedrock, azure_openai, nvidia"
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
                        "aliases": "\n".join(model_config.aliases),
                    }
                )

        return result
