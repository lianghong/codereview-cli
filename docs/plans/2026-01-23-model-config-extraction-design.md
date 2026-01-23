# Model Configuration Extraction Design

**Date:** 2026-01-23
**Status:** Approved
**Author:** Design Session with User

## Overview

Extract model parameters, pricing, and configuration into external YAML files to facilitate expansion and maintenance. Add Azure OpenAI provider support alongside existing AWS Bedrock, with automatic provider detection.

## Goals

1. **Easy Model Addition** - Add new models by editing YAML, not code
2. **External Configuration** - Non-developers can update model configs
3. **Better Organization** - Separate pricing, inference params, and metadata
4. **Provider Extensibility** - Easy to add new providers (Azure OpenAI first)
5. **Backward Compatibility** - Existing code continues working

## Provider Strategy

- **Bedrock:** Claude, Mistral, Minimax, Kimi, Qwen models
- **Azure OpenAI:** GPT models (starting with GPT-5.2 Codex)
- **Future:** OpenAI API, Google Vertex AI, etc.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│  - Single --model flag                                       │
│  - Auto-detects provider                                     │
│  - --list-models to discover options                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   ProviderFactory                            │
│  - Resolves model name → (provider, config)                 │
│  - Creates appropriate provider instance                     │
└────────────────────┬────────────────────────────────────────┘
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│ BedrockProvider  │  │ AzureOpenAI      │
│                  │  │ Provider         │
└──────────────────┘  └──────────────────┘
           │                   │
           └─────────┬─────────┘
                     ▼
           ┌──────────────────┐
           │ ModelProvider    │
           │ (Abstract Base)  │
           └──────────────────┘
```

### Configuration Flow

```
models.yaml
    ↓
ConfigLoader (parse + validate)
    ↓
Pydantic Models (type-safe)
    ↓
ProviderFactory (resolve model name)
    ↓
Concrete Provider (Bedrock/Azure)
    ↓
LangChain Client (API calls)
```

## Configuration Structure

### YAML File: `codereview/config/models.yaml`

```yaml
# Model pricing and configuration
# Prices are per million tokens

providers:
  bedrock:
    region: us-west-2  # Default region
    models:
      - id: opus
        full_id: global.anthropic.claude-opus-4-5-20251101-v1:0
        name: Claude Opus 4.5
        aliases: [claude-opus]
        pricing:
          input_per_million: 5.00
          output_per_million: 25.00
        inference_params:
          default_temperature: 0.1

      - id: sonnet
        full_id: global.anthropic.claude-sonnet-4-5-20250929-v1:0
        name: Claude Sonnet 4.5
        aliases: [claude-sonnet]
        pricing:
          input_per_million: 3.00
          output_per_million: 15.00
        inference_params:
          default_temperature: 0.1

      # ... other Bedrock models (haiku, minimax, mistral, kimi, qwen)

  azure_openai:
    endpoint: "${AZURE_OPENAI_ENDPOINT}"  # From env var
    api_key: "${AZURE_OPENAI_API_KEY}"    # From env var
    api_version: "2026-01-14"
    models:
      - id: gpt-5.2-codex
        deployment_name: gpt-5.2-codex  # Azure deployment name
        name: GPT-5.2 Codex
        aliases: [gpt52, codex]
        pricing:
          input_per_million: 1.75
          output_per_million: 14.00
        inference_params:
          default_temperature: 0.0
          default_top_p: 0.95
          max_output_tokens: 16000
```

**Features:**
- Single file for all providers
- Environment variable substitution (`${VAR}`)
- Provider-level defaults
- Model-level overrides
- Human-readable comments

## Data Models

### Pydantic Schemas (`codereview/config/models.py`)

```python
class PricingConfig(BaseModel):
    input_per_million: float = Field(gt=0)
    output_per_million: float = Field(gt=0)

class InferenceParams(BaseModel):
    default_temperature: float | None = Field(None, ge=0, le=2)
    default_top_p: float | None = Field(None, ge=0, le=1)
    default_top_k: int | None = Field(None, ge=0)
    max_output_tokens: int | None = Field(None, gt=0)

class ModelConfig(BaseModel):
    id: str  # Short name for CLI
    name: str  # Display name
    aliases: list[str] = Field(default_factory=list)
    pricing: PricingConfig
    inference_params: InferenceParams = Field(default_factory=InferenceParams)

    # Provider-specific fields
    full_id: str | None = None  # Bedrock
    deployment_name: str | None = None  # Azure

class BedrockConfig(ProviderConfig):
    region: str = "us-west-2"

class AzureOpenAIConfig(ProviderConfig):
    endpoint: str
    api_key: str
    api_version: str
```

**Benefits:**
- Automatic validation (pricing > 0, temperature 0-2)
- Type hints for IDE support
- Self-documenting schema
- Extensible per-provider

## Provider Abstraction

### Base Interface (`codereview/providers/base.py`)

```python
class ModelProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def analyze_batch(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        max_retries: int = 3,
    ) -> CodeReviewReport:
        """Analyze a batch of files."""
        pass

    @abstractmethod
    def get_model_display_name(self) -> str:
        """Get human-readable model name."""
        pass

    def reset_state(self) -> None:
        """Reset token counters."""
        pass

    def estimate_cost(self) -> dict[str, float]:
        """Calculate cost from token usage."""
        pass
```

### Concrete Implementations

**BedrockProvider** (`codereview/providers/bedrock.py`)
- Uses `ChatBedrockConverse` from LangChain
- Handles Bedrock-specific retry logic (ThrottlingException)
- Extracts token usage from `usage_metadata`

**AzureOpenAIProvider** (`codereview/providers/azure_openai.py`)
- Uses `AzureChatOpenAI` from LangChain
- Handles OpenAI rate limits (RateLimitError)
- Extracts token usage from `response_metadata.token_usage`

## Provider Factory

### Auto-Detection (`codereview/providers/factory.py`)

```python
class ProviderFactory:
    def create_provider(
        self,
        model_name: str,
        temperature: float | None = None,
    ) -> ModelProvider:
        """Create provider based on model name lookup."""
        # 1. Resolve model name → (provider, ModelConfig)
        provider_name, model_config = self.config_loader.resolve_model(model_name)

        # 2. Get provider config
        provider_config = self.config_loader.get_provider_config(provider_name)

        # 3. Instantiate appropriate provider
        if provider_name == "bedrock":
            return BedrockProvider(model_config, provider_config, temperature)
        elif provider_name == "azure_openai":
            return AzureOpenAIProvider(model_config, provider_config, temperature)
```

**Resolution Logic:**
1. Check primary model ID
2. Check model aliases
3. Return provider + config or raise ValueError

## CLI Changes

### Simplified Interface

**Before:**
```bash
codereview ./src --model-id global.anthropic.claude-opus-4-5-20251101-v1:0 --aws-region us-west-2
```

**After:**
```bash
codereview ./src --model opus  # Auto-detects Bedrock
codereview ./src --model gpt-5.2-codex  # Auto-detects Azure
```

### New Features

**List available models:**
```bash
codereview --list-models

Available Models
┌──────────────┬─────────────────┬──────────────┬─────────────────┐
│ ID           │ Name            │ Provider     │ Aliases         │
├──────────────┼─────────────────┼──────────────┼─────────────────┤
│ opus         │ Claude Opus 4.5 │ bedrock      │ claude-opus     │
│ sonnet       │ Claude Sonnet   │ bedrock      │ claude-sonnet   │
│ gpt-5.2-codex│ GPT-5.2 Codex   │ azure_openai │ gpt52, codex    │
└──────────────┴─────────────────┴──────────────┴─────────────────┘

Usage: codereview ./src --model <id>
```

## Integration with CodeAnalyzer

### Refactored Analyzer (`codereview/analyzer.py`)

```python
class CodeAnalyzer:
    def __init__(
        self,
        model_name: str = "opus",
        temperature: float | None = None,
        provider_factory: ProviderFactory | None = None,
    ):
        """Initialize with provider auto-detection."""
        self.factory = provider_factory or ProviderFactory()
        self.provider = self.factory.create_provider(model_name, temperature)

    def analyze_batch(self, ...) -> CodeReviewReport:
        """Delegate to provider."""
        return self.provider.analyze_batch(...)
```

**Changes:**
- Constructor takes `model_name` instead of `model_id` + `region`
- All logic delegated to provider
- Properties expose provider state (tokens, cost)
- Maintains backward compatibility

## File Structure

```
codereview/
├── config/
│   ├── __init__.py
│   ├── models.py           # Pydantic data models (NEW)
│   ├── loader.py           # ConfigLoader (NEW)
│   └── models.yaml         # Model definitions (NEW)
├── providers/
│   ├── __init__.py         # (NEW)
│   ├── base.py            # ModelProvider ABC (NEW)
│   ├── bedrock.py         # BedrockProvider (NEW)
│   ├── azure_openai.py    # AzureOpenAIProvider (NEW)
│   └── factory.py         # ProviderFactory (NEW)
├── analyzer.py            # Refactored (MODIFIED)
├── cli.py                 # Updated CLI (MODIFIED)
├── config.py              # Legacy support (MODIFIED)
└── ... (other files unchanged)
```

## Implementation Plan

### Phase 1: Configuration System (No Breaking Changes)
1. Create `models.yaml` with all Bedrock models + Azure OpenAI
2. Create Pydantic data models in `config/models.py`
3. Create `ConfigLoader` in `config/loader.py`
4. Add tests for config loading and validation
5. Verify YAML parsing, env var expansion, validation

### Phase 2: Provider Abstraction (Parallel)
6. Create `providers/base.py` with `ModelProvider` ABC
7. Create `providers/bedrock.py` (extract from current `CodeAnalyzer`)
8. Create `providers/azure_openai.py` (new Azure support)
9. Create `providers/factory.py` for provider selection
10. Add provider tests with mocked LangChain clients

### Phase 3: Refactor CodeAnalyzer (Backward Compatible)
11. Refactor `analyzer.py` to use `ProviderFactory`
12. Keep old constructor working with deprecation warning
13. Update tests to new structure
14. Verify all 100 tests pass

### Phase 4: Update CLI (User-Facing)
15. Update CLI with simplified `--model` option
16. Add `--list-models` flag
17. Update help text and examples
18. Test with both Bedrock and Azure models

### Phase 5: Documentation & Cleanup
19. Update `CLAUDE.md` with new architecture
20. Add Azure OpenAI setup instructions to README
21. Create migration guide for users
22. Mark old `config.py` constants as deprecated
23. Plan removal of legacy code in next major version

## Dependencies

### New Packages Required

```toml
# pyproject.toml additions
[project]
dependencies = [
    # ... existing deps ...
    "pyyaml>=6.0",              # YAML parsing
    "langchain-openai>=0.1.0",  # Azure OpenAI support
]
```

### Environment Variables

**Azure OpenAI (new):**
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
```

**AWS Bedrock (unchanged):**
- Uses existing AWS credentials (CLI, env vars, IAM role)

## Backward Compatibility

### No Breaking Changes

- Existing `CodeAnalyzer(model_id="...")` works via legacy wrapper
- Old model IDs resolve correctly through alias mapping
- All 100 existing tests pass without modification
- CLI maintains existing `--model` behavior
- New features are opt-in

### Legacy Support

Keep `config.py` with deprecated constants:
```python
# Build SUPPORTED_MODELS from YAML for backward compat
SUPPORTED_MODELS = {
    "global.anthropic.claude-opus-4-5-20251101-v1:0": {...},
    # ... etc
}

MODEL_ALIASES = {
    "opus": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    # ... etc
}
```

Mark with deprecation warnings for future removal.

## Testing Strategy

### New Test Coverage

**Config Loader Tests:**
- YAML parsing and validation
- Environment variable expansion
- Model resolution by ID and alias
- Error handling for missing/invalid configs

**Provider Tests:**
- Factory creates correct provider type
- Bedrock provider analyze_batch
- Azure provider analyze_batch
- Token tracking and cost estimation
- Retry logic for rate limiting
- Error handling

**Integration Tests:**
- End-to-end with both providers
- CLI with --list-models
- CLI with --model for each provider
- Backward compatibility with old API

### Test Fixtures

Use temporary YAML files with test models:
```python
@pytest.fixture
def mock_config_loader():
    test_yaml = """
    providers:
      bedrock:
        models:
          - id: test-opus
            full_id: test.opus
            ...
      azure_openai:
        models:
          - id: test-gpt
            deployment_name: test-gpt
            ...
    """
    # Write to temp file and load
```

### Mocking Strategy

- Mock LangChain clients (`ChatBedrockConverse`, `AzureChatOpenAI`)
- Mock API responses with token metadata
- No real API calls in tests
- Verify provider selection logic
- Test error handling paths

## Success Criteria

✅ Single YAML file for all model configs
✅ Easy to add new models (edit YAML only)
✅ Easy to add new providers (implement interface)
✅ Auto-detection of provider from model name
✅ Azure OpenAI support for GPT-5.2 Codex
✅ Backward compatible with existing code
✅ All 100 tests pass
✅ Type-safe with Pydantic validation
✅ Environment variable support for secrets
✅ Clean separation of concerns

## Future Enhancements

### Potential Additions (Out of Scope)

- OpenAI API provider (non-Azure)
- Google Vertex AI provider
- Anthropic API direct (non-Bedrock)
- Model capability detection (context window, vision, etc.)
- Cost budgeting and alerts
- Model performance benchmarking
- A/B testing between models
- Provider failover/fallback logic

### Extensibility Points

The design allows easy addition of:
- New providers (implement `ModelProvider` interface)
- New models (add to YAML)
- Provider-specific features (subclass `ModelProvider`)
- Custom configuration sources (implement `ConfigLoader` interface)

## Migration Guide for Users

### For CLI Users

**No changes required!** Existing commands work:
```bash
codereview ./src --model opus  # Still works
```

**New convenience:**
```bash
codereview --list-models  # Discover all options
codereview ./src --model gpt-5.2-codex  # Use Azure OpenAI
```

### For Library Users

**Old way (still works):**
```python
from codereview.analyzer import CodeAnalyzer

analyzer = CodeAnalyzer(
    model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
    region="us-west-2"
)
```

**New way (recommended):**
```python
from codereview.analyzer import CodeAnalyzer

analyzer = CodeAnalyzer(model_name="opus")
```

### For Contributors

**Adding a new Bedrock model:**
1. Edit `codereview/config/models.yaml`
2. Add model entry under `providers.bedrock.models`
3. Done! No code changes needed.

**Adding a new provider:**
1. Create `codereview/providers/newprovider.py`
2. Implement `ModelProvider` interface
3. Add to factory in `providers/factory.py`
4. Add provider config to `models.yaml`
5. Add tests

## Risks & Mitigations

### Risk: YAML Syntax Errors

**Mitigation:**
- Pydantic validation catches errors at load time
- Clear error messages with field names
- Schema validation in CI/CD
- Example YAML in documentation

### Risk: Environment Variable Not Set

**Mitigation:**
- ConfigLoader checks required env vars
- Helpful error messages with var names
- Fallback to defaults where sensible
- Documentation lists required variables

### Risk: Provider API Changes

**Mitigation:**
- Provider abstraction isolates changes
- Version pin LangChain dependencies
- Tests catch breaking changes
- Provider-specific retry logic

### Risk: Backward Compatibility Break

**Mitigation:**
- Legacy wrapper maintains old API
- Deprecation warnings, not removal
- All existing tests must pass
- Gradual migration path

## Conclusion

This design achieves all goals:

1. ✅ **Easy expansion** - Add models via YAML editing
2. ✅ **External config** - Non-developers can update YAML
3. ✅ **Better organization** - Clear separation of concerns
4. ✅ **Provider extensibility** - Clean interface for new providers
5. ✅ **Backward compatible** - Existing code works unchanged

The provider adapter pattern balances flexibility with simplicity. Azure OpenAI support adds GPT-5.2 Codex at competitive pricing ($1.75 input vs $5.00 for Opus). The configuration-driven approach makes future model additions trivial.

Ready for implementation!
