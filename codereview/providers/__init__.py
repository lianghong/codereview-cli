"""Provider abstraction for multiple LLM backends."""

from codereview.providers.base import ModelProvider, RetryConfig
from codereview.providers.factory import ProviderFactory

__all__ = [
    "ModelProvider",
    "RetryConfig",
    "ProviderFactory",
]


def __getattr__(name: str) -> type:
    """Lazy-load provider classes to avoid importing heavy dependencies at module level."""
    if name == "BedrockProvider":
        from codereview.providers.bedrock import BedrockProvider

        return BedrockProvider
    if name == "AzureOpenAIProvider":
        from codereview.providers.azure_openai import AzureOpenAIProvider

        return AzureOpenAIProvider
    if name == "NVIDIAProvider":
        from codereview.providers.nvidia import NVIDIAProvider

        return NVIDIAProvider
    if name == "GoogleGenAIProvider":
        from codereview.providers.google_genai import GoogleGenAIProvider

        return GoogleGenAIProvider
    if name == "ZAIProvider":
        from codereview.providers.zai import ZAIProvider

        return ZAIProvider
    if name == "DeepSeekProvider":
        from codereview.providers.deepseek import DeepSeekProvider

        return DeepSeekProvider
    if name == "MoonshotProvider":
        from codereview.providers.moonshot import MoonshotProvider

        return MoonshotProvider
    if name == "BedrockOpenAIProvider":
        from codereview.providers.bedrock_openai import BedrockOpenAIProvider

        return BedrockOpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
