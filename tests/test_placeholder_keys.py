"""Placeholder API-key rejection: shared helper + per-provider contract.

CLAUDE.md contract: the placeholder set must include the exact strings the
README tells users to export — matched case-insensitively after .strip() —
so a copied-and-not-replaced placeholder fails fast at --validate instead of
401'ing on the first real call.

These tests lock that contract for every provider that validates an API key,
via the shared helper in mixins.py (single source of truth for the generic
placeholders) plus each provider's README-documented string.
"""

import os
from unittest.mock import patch

import pytest

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockOpenAIConfig,
    DeepSeekConfig,
    GoogleGenAIConfig,
    ModelConfig,
    MoonshotConfig,
    NVIDIAConfig,
    PricingConfig,
    ZAIConfig,
)
from codereview.providers.mixins import (
    is_blank,
    is_https_url,
    is_placeholder_api_key,
    is_short_api_key,
    require_https,
)

# ---------------------------------------------------------------------------
# Shared helper unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "placeholder",
        "your-api-key-here",
        "your-api-key",
        "  PLACEHOLDER  ",  # strip + case-insensitive
        "Your-API-Key-Here",
    ],
)
def test_generic_placeholders_rejected(key):
    assert is_placeholder_api_key(key)


@pytest.mark.parametrize(
    "key,extra",
    [
        ("your-deepseek-key", ("your-deepseek-key", "your-deepseek-api-key-here")),
        ("  Your-Moonshot-Key ", ("your-moonshot-key",)),
    ],
)
def test_provider_specific_placeholders_rejected(key, extra):
    assert is_placeholder_api_key(key, extra)


def test_real_key_accepted():
    assert not is_placeholder_api_key("sk-abc123def456ghi789jkl012")
    assert not is_placeholder_api_key(
        "nvapi-x9y8z7w6v5u4t3s2r1q0", ("nvapi-your-key-here",)
    )


# ---------------------------------------------------------------------------
# The other two normalization helpers validate_credentials shares
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "   ", "\t", "\n", " \t\n ", None])
def test_is_blank_covers_whitespace_and_none(value):
    """Whitespace-only is "no credential", not a credential.

    Pydantic's ``min_length=1`` accepts "   " and truthiness accepts it too, so
    a bare ``not api_key`` let it through — the reason a whitespace-only key
    used to pass every --validate check.
    """
    assert is_blank(value)


@pytest.mark.parametrize("value", ["x", "sk-abc", "  padded  "])
def test_is_blank_accepts_any_real_value(value):
    assert not is_blank(value)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.z.ai/api/paas/v4/",
        "HTTPS://api.z.ai/api/paas/v4/",  # RFC 3986: schemes are case-insensitive
        "Https://api.deepseek.com",
        "  https://api.moonshot.cn/v1  ",
    ],
)
def test_is_https_url_accepts_every_valid_spelling(url):
    assert is_https_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://api.z.ai/v4/",
        "HTTP://api.z.ai/v4/",
        "ftp://example.com",
        "api.z.ai/v4/",  # scheme-relative: not HTTPS
        "",
        "   ",
        # Not a scheme — "https:/" and "https" alone must not pass, or a typo
        # would be read as secure.
        "https:/api.z.ai",
        "https",
        # Scheme present but no host: nothing to connect to. These satisfy a
        # bare startswith("https://") test, which is why --validate used to
        # report them as a green check and the run then failed at client
        # construction or on the first request.
        "https://",
        "https:// ",
        "https:///v1",
        "https://:8443/v1",
        "https://[::1",  # malformed IPv6 authority: urlsplit raises
    ],
)
def test_is_https_url_rejects_everything_else(url):
    assert not is_https_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.z.ai/v4/",
        "https://[::1]/v1",  # bracketed IPv6 authority
        "https://user@api.example.com/v1",  # userinfo@host
        "https://api.example.com:8443/v1",
    ],
)
def test_is_https_url_accepts_hosts_string_surgery_would_mangle(url):
    """A real host must be recognized however the authority is spelled."""
    assert is_https_url(url)


def test_is_https_url_is_the_predicate_require_https_enforces():
    """The two must agree by construction, not by two similar implementations.

    They had drifted — ``require_https`` lowercased, the providers' inline
    ``startswith`` did not — so an uppercase-scheme URL built a working client
    that --validate then reported as a hard failure.
    """
    for url in ("HTTPS://host/v1", "https://host/v1", "http://host/v1", "https://"):
        if is_https_url(url):
            assert require_https(url, "base_url") == url
        else:
            with pytest.raises(ValueError, match="must use HTTPS"):
                require_https(url, "base_url")


def test_require_https_returns_the_value_it_validated():
    """The client must receive the string the predicate actually tested.

    ``is_https_url`` strips before testing, so returning the caller's raw value
    handed a padded ``"  https://host  "`` straight to the HTTP client — the
    validated value and the used value were different strings.
    """
    assert require_https("  https://api.example.com/v1  ", "base_url") == (
        "https://api.example.com/v1"
    )
    # No padding: unchanged, so callers comparing against their input still pass.
    assert require_https("https://api.example.com/v1", "base_url") == (
        "https://api.example.com/v1"
    )


@pytest.mark.parametrize("key", ["", "short", "x" * 19, "  " + "x" * 18 + "  "])
def test_is_short_api_key_flags_implausible_lengths(key):
    """Strips first, so padding can't push a short key past the threshold."""
    assert is_short_api_key(key)


@pytest.mark.parametrize("key", ["x" * 20, "sk-abc123def456ghi789jkl012"])
def test_is_short_api_key_accepts_plausible_lengths(key):
    assert not is_short_api_key(key)


# ---------------------------------------------------------------------------
# Per-provider contract: the EXACT string the README documents must hard-fail
# --validate. README export lines:
#   AZURE_OPENAI_API_KEY="your-api-key"        (README.md:132)
#   NVIDIA_API_KEY="nvapi-your-key-here"       (README.md:182)
#   GOOGLE_API_KEY="your-api-key-here"         (README.md:240)
#   DEEPSEEK_API_KEY="your-deepseek-key"       (README.md:265)
#   ZAI_API_KEY="your-zai-key"                 (README.md:289)
#   KIMI_API_KEY="your-moonshot-key"           (README.md:315)
# ---------------------------------------------------------------------------


def _model_config(**overrides):
    defaults = dict(
        id="test",
        full_id="vendor/test-model",
        name="Test Model",
        pricing=PricingConfig(input_per_million=1.0, output_per_million=5.0),
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


@pytest.fixture(autouse=True)
def _skip_connection_tests():
    os.environ["CODEREVIEW_SKIP_CONNECTION_TEST"] = "1"
    yield
    os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)


def _assert_placeholder_fails(provider):
    result = provider.validate_credentials()
    assert result.valid is False, (
        "README-documented placeholder key must hard-fail --validate "
        f"(provider={result.provider})"
    )


@pytest.mark.parametrize("key", ["your-zai-key", "  Your-ZAI-Key  "])
def test_zai_rejects_readme_placeholder(key):
    from codereview.providers.zai import ZAIProvider

    config = ZAIConfig(api_key=key)
    with patch("codereview.providers.zai.ChatOpenAI"):
        _assert_placeholder_fails(ZAIProvider(_model_config(), config))


@pytest.mark.parametrize("key", ["your-api-key", "  Your-API-Key  "])
def test_azure_rejects_readme_placeholder(key):
    from codereview.providers.azure_openai import AzureOpenAIProvider

    config = AzureOpenAIConfig(
        endpoint="https://test.openai.azure.com",
        api_key=key,
        api_version="2024-01-01",
    )
    with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
        _assert_placeholder_fails(
            AzureOpenAIProvider(_model_config(deployment_name="test-deploy"), config)
        )


@pytest.mark.parametrize("key", ["nvapi-your-key-here", "  NVAPI-Your-Key-Here "])
def test_nvidia_rejects_readme_placeholder(key):
    from codereview.providers.nvidia import NVIDIAProvider

    config = NVIDIAConfig(api_key=key)
    with patch("codereview.providers.nvidia.ChatNVIDIA"):
        _assert_placeholder_fails(NVIDIAProvider(_model_config(), config))


@pytest.mark.parametrize("key", ["your-api-key-here", " Your-API-Key-Here "])
def test_google_rejects_readme_placeholder(key):
    from codereview.providers.google_genai import GoogleGenAIProvider

    config = GoogleGenAIConfig(api_key=key)
    with patch("codereview.providers.google_genai.ChatGoogleGenerativeAI"):
        _assert_placeholder_fails(GoogleGenAIProvider(_model_config(), config))


@pytest.mark.parametrize("key", ["your-deepseek-key", " Your-DeepSeek-Key "])
def test_deepseek_rejects_readme_placeholder(key):
    from codereview.providers.deepseek import DeepSeekProvider

    config = DeepSeekConfig(api_key=key)
    with patch("codereview.providers.deepseek.ChatDeepSeek"):
        _assert_placeholder_fails(DeepSeekProvider(_model_config(), config))


@pytest.mark.parametrize("key", ["your-moonshot-key", " Your-Moonshot-Key "])
def test_moonshot_rejects_readme_placeholder(key):
    from codereview.providers.moonshot import MoonshotProvider

    config = MoonshotConfig(api_key=key)
    with patch("codereview.providers.moonshot.ChatMoonshot"):
        _assert_placeholder_fails(MoonshotProvider(_model_config(), config))


@pytest.mark.parametrize(
    "key",
    [
        "your-bedrock-api-key-here",
        " Your-Bedrock-API-Key-Here ",
        # The exact string README.md's export line documents, angle brackets
        # included (README.md:341) — the CLAUDE.md contract requires it.
        "<your-amazon-bedrock-api-key>",
        " <Your-Amazon-Bedrock-API-Key> ",
        # Same string with the brackets stripped, as a user pasting from prose
        # would leave it.
        "your-amazon-bedrock-api-key",
    ],
)
def test_bedrock_openai_rejects_placeholder(key):
    from codereview.providers.bedrock_openai import BedrockOpenAIProvider

    config = BedrockOpenAIConfig(
        api_key=key,
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
    )
    with patch("codereview.providers.bedrock_openai.ChatOpenAI"):
        _assert_placeholder_fails(BedrockOpenAIProvider(_model_config(), config))


# ---------------------------------------------------------------------------
# Drift guard: scrape the README's own export lines
#
# The per-provider tests above hardcode the documented placeholder, so they go
# stale the moment README.md's export line is reworded. This derives the strings
# from the README itself: every `export <PROVIDER>_API_KEY="..."` value must be
# rejected by the provider that reads that env var. Adding a provider or
# rewording an export line fails here until the placeholder set catches up.
# ---------------------------------------------------------------------------

# Env var -> the provider-specific `extra` tuple its validate_credentials passes
# to is_placeholder_api_key. Bedrock's SigV4 path has no API key, so it is absent.
_README_KEY_EXTRAS: dict[str, tuple[str, ...]] = {
    "AZURE_OPENAI_API_KEY": (),
    "NVIDIA_API_KEY": ("nvapi-your-key-here",),
    "GOOGLE_API_KEY": (),
    "DEEPSEEK_API_KEY": ("your-deepseek-key", "your-deepseek-api-key-here"),
    "ZAI_API_KEY": ("your-zai-api-key-here", "your-zai-key"),
    "KIMI_API_KEY": ("your-moonshot-key", "your-moonshot-api-key-here"),
    "OPENAI_API_KEY": (
        "your-bedrock-api-key-here",
        "<your-amazon-bedrock-api-key>",
        "your-amazon-bedrock-api-key",
    ),
}


def _readme_export_placeholders() -> list[tuple[str, str]]:
    """Return (env_var, placeholder_value) for each README export line."""
    import re
    from pathlib import Path

    readme = Path(__file__).resolve().parent.parent / "README.md"
    pattern = re.compile(r'^export ([A-Z0-9_]*API_KEY)="([^"]+)"', re.MULTILINE)
    return pattern.findall(readme.read_text())


def test_every_readme_documented_placeholder_is_rejected():
    """Each `export *_API_KEY="..."` value in README.md must hard-fail --validate.

    CLAUDE.md contract: "The placeholder set must include the exact strings the
    README tells users to export" — so a copied-and-not-replaced placeholder
    fails fast at --validate instead of 401'ing on the first real call.
    """
    found = _readme_export_placeholders()
    assert found, "no export *_API_KEY lines matched in README.md; regex is wrong"

    unknown = [var for var, _ in found if var not in _README_KEY_EXTRAS]
    assert not unknown, (
        f"README documents {unknown} but _README_KEY_EXTRAS has no entry. Add the "
        "provider's `extra` tuple here (and to its validate_credentials)."
    )

    accepted = [
        f'{var}="{value}"'
        for var, value in found
        if not is_placeholder_api_key(value, _README_KEY_EXTRAS[var])
    ]
    assert not accepted, (
        "README-documented placeholder(s) accepted as a real key: "
        f"{accepted}. Add each to the provider's is_placeholder_api_key extras."
    )


# ---------------------------------------------------------------------------
# Table-driven credential validation: every provider × every bad-value axis
#
# The placeholder axis above is per-provider prose. These cover the remaining
# axes as one table so a new provider inherits all of them by adding one row,
# rather than by someone remembering to write four more tests:
#
#   blank      — "", "   ", "\t" (Pydantic's min_length=1 accepts whitespace)
#   malformed  — a cleartext or hostless URL
#   normalized — an uppercase scheme and a padded key, both of which must be
#                ACCEPTED, because the constructor accepts them
#
# The asymmetry is the point: the first two must hard-fail, the third must pass.
# A check that rejects a working config is as much a bug as one that accepts a
# broken one — --validate telling a user their good setup is broken sends them
# debugging a non-problem.
# ---------------------------------------------------------------------------

# provider key -> (module patch target, config builder, url field name or None)
_VALIDATION_PROVIDERS: dict[str, tuple[str, object, str | None]] = {
    "zai": (
        "codereview.providers.zai.ChatOpenAI",
        lambda **kw: ZAIConfig(api_key=kw.pop("api_key"), **kw),
        "base_url",
    ),
    "moonshot": (
        "codereview.providers.moonshot.ChatMoonshot",
        lambda **kw: MoonshotConfig(api_key=kw.pop("api_key"), **kw),
        "base_url",
    ),
    "deepseek": (
        "codereview.providers.deepseek.ChatDeepSeek",
        lambda **kw: DeepSeekConfig(api_key=kw.pop("api_key"), **kw),
        "api_base",
    ),
    "bedrock_openai": (
        # base_url is required here (no default): its models' endpoint is
        # region-specific, so the config has nothing sensible to fall back to.
        "codereview.providers.bedrock_openai.ChatOpenAI",
        lambda **kw: BedrockOpenAIConfig(
            api_key=kw.pop("api_key"),
            base_url=kw.pop(
                "base_url", "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
            ),
            **kw,
        ),
        "base_url",
    ),
    "nvidia": (
        "codereview.providers.nvidia.ChatNVIDIA",
        lambda **kw: NVIDIAConfig(api_key=kw.pop("api_key"), **kw),
        "base_url",
    ),
    "google_genai": (
        "codereview.providers.google_genai.ChatGoogleGenerativeAI",
        lambda **kw: GoogleGenAIConfig(api_key=kw.pop("api_key"), **kw),
        None,
    ),
    "azure_openai": (
        "codereview.providers.azure_openai.AzureChatOpenAI",
        lambda **kw: AzureOpenAIConfig(
            api_key=kw.pop("api_key"),
            endpoint=kw.pop("endpoint", "https://test.openai.azure.com"),
            api_version="2024-01-01",
        ),
        "endpoint",
    ),
}

_PROVIDER_CLASSES = {
    "zai": ("codereview.providers.zai", "ZAIProvider"),
    "moonshot": ("codereview.providers.moonshot", "MoonshotProvider"),
    "deepseek": ("codereview.providers.deepseek", "DeepSeekProvider"),
    "bedrock_openai": ("codereview.providers.bedrock_openai", "BedrockOpenAIProvider"),
    "nvidia": ("codereview.providers.nvidia", "NVIDIAProvider"),
    "google_genai": ("codereview.providers.google_genai", "GoogleGenAIProvider"),
    "azure_openai": ("codereview.providers.azure_openai", "AzureOpenAIProvider"),
}

_REAL_KEY = "sk-real-key-1234567890abcdef"


def _make_provider(provider_key, **config_kwargs):
    """Build a provider with its client patched, returning the instance."""
    import importlib

    target, build_config, _ = _VALIDATION_PROVIDERS[provider_key]
    module_path, class_name = _PROVIDER_CLASSES[provider_key]
    provider_cls = getattr(importlib.import_module(module_path), class_name)

    config = build_config(**config_kwargs)
    model_config = (
        _model_config(deployment_name="test-deploy")
        if provider_key == "azure_openai"
        else _model_config()
    )
    with patch(target):
        return provider_cls(model_config, config)


@pytest.mark.parametrize("provider_key", sorted(_VALIDATION_PROVIDERS))
@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n ", "  \t\n"])
def test_blank_api_key_hard_fails_for_every_provider(provider_key, blank):
    """A blank or whitespace-only key must fail --validate, attributed to the key.

    Pydantic's ``min_length=1`` accepts "   ", and a whitespace-only string is
    truthy, so a bare ``not api_key`` reported every check green and deferred the
    failure to a 401 on the first real call.
    """
    if blank == "":
        # Pydantic rejects "" at construction — the earlier, better failure.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_provider(provider_key, api_key=blank)
        return

    provider = _make_provider(provider_key, api_key=blank)
    result = provider.validate_credentials()

    assert result.valid is False, (
        f"{provider_key}: whitespace-only API key {blank!r} passed --validate. "
        f"Checks: {result.checks}"
    )
    assert any("API Key" in name and not passed for name, passed, _ in result.checks), (
        f"{provider_key}: the failure must be attributed to the API Key check, "
        f"got {result.checks}"
    )


_MALFORMED_URLS = (
    "http://insecure.example.com/v1",  # cleartext
    "https://",  # scheme only, no host
    "https:///v1",  # empty authority
)

# Azure's endpoint is typed ``HttpUrl``, so Pydantic parses and *rewrites* it
# before the provider sees a string: "https:///v1" becomes "https://v1/" — a
# well-formed URL naming host "v1". There is no hostless value the provider
# could reject, because one can't reach it. (Bare "https://" is still covered:
# Pydantic rejects it outright with "empty host".) Same reason Azure is excluded
# from test_validate_accepts_any_url_the_constructor_accepted.
_URL_CASES_NOT_REACHABLE = {("azure_openai", "https:///v1")}


def _malformed_url_cases():
    for provider_key, (_, _, url_field) in sorted(_VALIDATION_PROVIDERS.items()):
        if not url_field:
            continue
        for bad_url in _MALFORMED_URLS:
            if (provider_key, bad_url) in _URL_CASES_NOT_REACHABLE:
                continue
            yield pytest.param(provider_key, bad_url, id=f"{provider_key}-{bad_url}")


@pytest.mark.parametrize(("provider_key", "bad_url"), list(_malformed_url_cases()))
def test_malformed_url_fails_closed_for_every_url_taking_provider(
    provider_key, bad_url
):
    """A cleartext or hostless URL must fail at construction, before any client.

    ``require_https`` runs from ``_create_model`` (called by ``__init__``), so
    this is enforced even for a caller that never runs --validate: the API key
    must not reach a client pointed at ``http://`` (CWE-319). A hostless
    ``https://`` names no server, so it can't work either — failing here beats
    a green --validate followed by a connection error mid-run.
    """
    _, _, url_field = _VALIDATION_PROVIDERS[provider_key]

    with pytest.raises((ValueError, Exception)) as exc_info:
        _make_provider(provider_key, api_key=_REAL_KEY, **{url_field: bad_url})

    message = str(exc_info.value).lower()
    assert "https" in message or "url" in message, (
        f"{provider_key}: {bad_url!r} was rejected, but not for a URL reason: "
        f"{exc_info.value!r}"
    )


@pytest.mark.parametrize("provider_key", sorted(_VALIDATION_PROVIDERS))
def test_padded_api_key_is_accepted_after_normalization(provider_key):
    """Surrounding whitespace must not make a real key fail --validate.

    Every shared predicate strips before testing, so a key pasted with a
    trailing newline is the same credential. Rejecting it would report a working
    configuration as broken.
    """
    provider = _make_provider(provider_key, api_key=f"  {_REAL_KEY}\n")
    result = provider.validate_credentials()

    assert result.valid is True, (
        f"{provider_key}: a padded but real API key was rejected. "
        f"Checks: {result.checks}"
    )


def test_every_api_key_provider_is_covered_by_the_validation_table():
    """A new key-taking provider must join the table above.

    Reflects over the provider package: any provider whose config carries an
    ``api_key`` needs the blank/malformed/normalized axes, not just the
    placeholder one. Bedrock's SigV4 path has no API key and is skipped.
    """
    import importlib
    import inspect
    from pathlib import Path

    from codereview.providers.base import ModelProvider

    providers_dir = Path(__file__).resolve().parent.parent / "codereview" / "providers"
    key_taking = set()
    for path in sorted(providers_dir.glob("*.py")):
        if path.stem in {"__init__", "base", "mixins", "factory"}:
            continue
        module = importlib.import_module(f"codereview.providers.{path.stem}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                not issubclass(obj, ModelProvider)
                or obj is ModelProvider
                or obj.__module__ != module.__name__
            ):
                continue
            # A provider validates an api_key iff its config class declares one.
            hints = inspect.signature(obj.__init__).parameters
            config_param = hints.get("provider_config")
            if config_param is None:
                continue
            annotation = config_param.annotation
            fields = getattr(annotation, "model_fields", {})
            if "api_key" in fields:
                key_taking.add(path.stem)

    assert key_taking, "no api_key-taking providers found; the scan is broken"
    missing = sorted(key_taking - set(_VALIDATION_PROVIDERS))
    assert not missing, (
        f"provider(s) {missing} take an api_key but have no row in "
        "_VALIDATION_PROVIDERS. Add one so the blank / malformed-URL / "
        "normalized axes apply to them too."
    )
