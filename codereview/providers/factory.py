"""Factory for creating provider instances with auto-detection."""

from importlib import import_module
from typing import NamedTuple

from langchain_core.callbacks import BaseCallbackHandler

from codereview.config import ConfigLoader, get_config_loader
from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    BedrockOpenAIConfig,
    DeepSeekConfig,
    GoogleGenAIConfig,
    MoonshotConfig,
    NVIDIAConfig,
    ProviderConfig,
    ZAIConfig,
)
from codereview.providers.base import ModelProvider


class _ProviderEntry(NamedTuple):
    """How to build one provider: its config type and where its class lives.

    ``module``/``class_name`` are strings rather than the class itself so the
    import stays lazy. Each provider module imports its vendor's LangChain
    client at module scope, so eagerly importing all eight here would pull
    every client package into every run — including ``--list-models``, which
    touches no provider at all.
    """

    config_type: type[ProviderConfig]
    module: str
    class_name: str


# The single source of truth for provider dispatch. Adding a provider means
# adding one row: the config-type guard, the lazy import, the constructor call
# and the "supported providers" list in the error message are all derived from
# it, so they cannot disagree. Previously each provider had a ~15-line
# if/elif branch and the error message hand-listed the eight names — prose that
# a ninth provider would have silently left stale, exactly the drift the
# Provider Setup table in cli.py has tests for.
_PROVIDER_REGISTRY: dict[str, _ProviderEntry] = {
    "bedrock": _ProviderEntry(
        BedrockConfig, "codereview.providers.bedrock", "BedrockProvider"
    ),
    "azure_openai": _ProviderEntry(
        AzureOpenAIConfig,
        "codereview.providers.azure_openai",
        "AzureOpenAIProvider",
    ),
    "nvidia": _ProviderEntry(
        NVIDIAConfig, "codereview.providers.nvidia", "NVIDIAProvider"
    ),
    "google_genai": _ProviderEntry(
        GoogleGenAIConfig,
        "codereview.providers.google_genai",
        "GoogleGenAIProvider",
    ),
    "zai": _ProviderEntry(ZAIConfig, "codereview.providers.zai", "ZAIProvider"),
    "deepseek": _ProviderEntry(
        DeepSeekConfig, "codereview.providers.deepseek", "DeepSeekProvider"
    ),
    "moonshot": _ProviderEntry(
        MoonshotConfig, "codereview.providers.moonshot", "MoonshotProvider"
    ),
    "bedrock_openai": _ProviderEntry(
        BedrockOpenAIConfig,
        "codereview.providers.bedrock_openai",
        "BedrockOpenAIProvider",
    ),
}


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
            An instance of the provider class ``_PROVIDER_REGISTRY`` names for
            the model's provider section.

        Raises:
            ValueError: If model name not found or provider unknown
        """
        # Resolve model name to provider and config
        provider_name, model_config = self.config_loader.resolve_model(model_name)

        try:
            entry = _PROVIDER_REGISTRY[provider_name]
        except KeyError:
            raise ValueError(
                f"Unknown provider: {provider_name}. Supported providers: "
                f"{', '.join(_PROVIDER_REGISTRY)}"
            ) from None

        # Get provider-specific configuration
        provider_config = self.config_loader.get_provider_config(provider_name)

        # The loader picks the config class per provider section, so a mismatch
        # here means loader and factory disagree about a provider — a wiring bug,
        # not bad user input. Check it anyway: every provider's __init__ reads
        # fields off provider_config, so passing the wrong type would surface as
        # an AttributeError deep inside a constructor instead of here.
        if not isinstance(provider_config, entry.config_type):
            raise ValueError(
                f"Expected {entry.config_type.__name__} for {provider_name} "
                f"provider, got {type(provider_config).__name__}"
            )

        provider_class = getattr(import_module(entry.module), entry.class_name)
        provider: ModelProvider = provider_class(
            model_config,
            provider_config,
            temperature,
            callbacks=callbacks,
            project_context=project_context,
        )
        return provider

    def supports_token_streaming(self, model_name: str) -> bool:
        """Whether the provider behind ``model_name`` streams tokens.

        Answered from the provider *class*, without constructing it: the CLI
        needs this before it decides worker count and which callback handler to
        attach, and both of those feed the provider's constructor. Building a
        provider here would also require credentials, which ``--stream`` has no
        business demanding earlier than the run itself.

        Falls back to ``True`` for an unresolvable model so this never becomes
        the thing that fails a run — the real resolution error surfaces from
        ``create_provider`` with its full message.
        """
        try:
            provider_name, _ = self.config_loader.resolve_model(model_name)
            entry = _PROVIDER_REGISTRY[provider_name]
        except ValueError, KeyError:
            return True
        provider_class = getattr(import_module(entry.module), entry.class_name)
        supports: bool = provider_class.supports_token_streaming()
        return supports

    def list_available_models(self) -> dict[str, list[dict[str, str]]]:
        """List all available models grouped by provider.

        Returns:
            Dict mapping provider names to lists of model info dicts.
            Each model info dict contains: id, name, aliases, deprecated_aliases.

            ``aliases`` holds only the current, recommended names — the ones
            worth advertising. Back-compat-only names live in
            ``deprecated_aliases`` so the caller can decide whether to show
            them; see ``ModelConfig.deprecated_aliases``.
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
                        "deprecated_aliases": ", ".join(
                            model_config.deprecated_aliases
                        ),
                    }
                )

        return result
