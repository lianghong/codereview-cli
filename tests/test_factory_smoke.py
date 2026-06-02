"""Factory smoke tests across every configured model name and alias.

These guard against provider registration / export drift: the kind of bug
where a model is added to ``models.yaml`` (or a provider class ships) but the
factory branch, config-class union, loader branch, or ``__init__`` lazy export
is missed. A model that resolves but cannot instantiate its provider — or a
provider class that cannot be imported from the package root — fails here.

Every LLM client is mocked, so no network or credentials are touched; the
provider env vars are set to dummy values so all provider sections register.
"""

from unittest.mock import MagicMock, patch

import pytest

from codereview.config import get_config_loader
from codereview.providers.factory import ProviderFactory

# Provider env vars (dummy values) so every provider section in the real
# models.yaml registers and is resolvable. Endpoints must look like URLs.
_PROVIDER_ENV = {
    "AZURE_OPENAI_API_KEY": "test-key-1234567890abcdef",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
    "DEEPSEEK_API_KEY": "test-key-1234567890abcdef",
    "GOOGLE_API_KEY": "test-key-1234567890abcdef",
    "KIMI_API_KEY": "test-key-1234567890abcdef",
    "NVIDIA_API_KEY": "test-key-1234567890abcdef",
    "ZAI_API_KEY": "test-key-1234567890abcdef",
    "OPENAI_API_KEY": "test-key-1234567890abcdef",
    "OPENAI_BASE_URL": "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
    "BEDROCK_OPENAI_MODEL_ID": "openai.gpt-5.5",
    "NVIDIA_BASE_URL": "",
}

# Every concrete LLM client class a provider may construct. Patching these
# means create_provider runs the real construction path (so a broken factory
# branch / config mismatch still fails) without touching the network.
_CLIENT_PATCH_TARGETS = [
    "codereview.providers.bedrock.ChatBedrockConverse",
    "codereview.providers.azure_openai.AzureChatOpenAI",
    "codereview.providers.nvidia.ChatNVIDIA",
    "codereview.providers.google_genai.ChatGoogleGenerativeAI",
    "codereview.providers.zai.ChatOpenAI",
    "codereview.providers.deepseek.ChatDeepSeek",
    "codereview.providers.moonshot.ChatMoonshot",
    "codereview.providers.bedrock_openai.ChatOpenAI",
]


@pytest.fixture
def all_provider_env(monkeypatch):
    """Set dummy provider credentials and reset the config singleton."""
    for key, value in _PROVIDER_ENV.items():
        monkeypatch.setenv(key, value)
    # The loader is an @lru_cache singleton; clear it so it re-reads with the
    # env vars set above, and clear again on teardown so other tests get a
    # clean slate.
    get_config_loader.cache_clear()
    yield
    get_config_loader.cache_clear()


def _all_names() -> list[str]:
    """Every model id and alias registered in the real models.yaml."""
    loader = get_config_loader()
    return sorted(loader.get_model_aliases().keys())


def _patched_clients():
    """A context manager stack patching every LLM client to a MagicMock.

    with_structured_output returns a MagicMock too, so both the tool-use and
    prompt-parsing construction paths complete.
    """
    from contextlib import ExitStack

    stack = ExitStack()
    for target in _CLIENT_PATCH_TARGETS:
        mock_cls = stack.enter_context(patch(target))
        instance = MagicMock()
        instance.with_structured_output.return_value = MagicMock()
        mock_cls.return_value = instance
    return stack


def test_every_name_resolves_to_a_provider(all_provider_env):
    """Every id/alias resolves to a (provider, ModelConfig) without error."""
    loader = get_config_loader()
    names = _all_names()
    assert names, "expected the real models.yaml to register models"

    for name in names:
        provider_name, model_config = loader.resolve_model(name)
        assert provider_name
        assert model_config.id


def test_every_name_instantiates_a_provider(all_provider_env):
    """create_provider succeeds for every id and alias.

    Catches: missing factory branch, config-class type mismatch, missing
    loader branch, and missing __init__ lazy export — for any registered name.
    """
    factory = ProviderFactory()
    names = _all_names()

    with _patched_clients():
        for name in names:
            provider = factory.create_provider(name)
            assert provider is not None, f"create_provider returned None for {name!r}"
            # Smoke the common contract surface every provider must expose.
            assert callable(provider.analyze_batch)
            assert callable(provider.get_pricing)
            assert callable(provider.validate_credentials)


def test_id_and_aliases_resolve_to_same_config(all_provider_env):
    """An alias must resolve to the same provider+config as its primary id."""
    loader = get_config_loader()
    alias_to_id = loader.get_model_aliases()

    for name, primary_id in alias_to_id.items():
        name_provider, name_cfg = loader.resolve_model(name)
        id_provider, id_cfg = loader.resolve_model(primary_id)
        assert name_provider == id_provider
        assert name_cfg.id == id_cfg.id


def test_every_registered_provider_name_is_constructible(all_provider_env):
    """At least one model per registered provider instantiates.

    Ensures every provider section in models.yaml has a working factory path,
    independent of how many models each exposes.
    """
    loader = get_config_loader()
    factory = ProviderFactory()

    # Map provider -> one representative model id.
    rep_by_provider: dict[str, str] = {}
    for name, primary_id in loader.get_model_aliases().items():
        if name != primary_id:
            continue  # only primary ids
        provider_name, _ = loader.resolve_model(name)
        rep_by_provider.setdefault(provider_name, name)

    assert set(rep_by_provider) == set(loader._providers), (
        "every registered provider should have at least one resolvable model"
    )

    with _patched_clients():
        for provider_name, model_id in rep_by_provider.items():
            provider = factory.create_provider(model_id)
            assert provider is not None, f"{provider_name} via {model_id!r} failed"
