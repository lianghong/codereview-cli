# Migration Guide: v0.1.x → v0.2.0

This guide helps you migrate from version 0.1.x to 0.2.0, which introduces the provider abstraction architecture and Azure OpenAI support.

## Overview of Changes

Version 0.2.0 introduces:
- **Provider Abstraction**: Unified interface for AWS Bedrock and Azure OpenAI
- **Simplified CLI**: Use short model names like `opus` instead of full model IDs
- **Azure OpenAI Support**: Use GPT models alongside Claude
- **YAML Configuration**: External configuration file for models and providers
- **Multi-Language Static Analysis**: Support for Python, Go, Shell, C++, Java, JavaScript, TypeScript

## Breaking Changes

### None! 🎉

Version 0.2.0 maintains **full backward compatibility**. Existing code will continue working with deprecation warnings.

## CLI Changes

### Model Selection (Recommended)

**Before (v0.1.x):**
```bash
codereview ./src --model-id global.anthropic.claude-opus-4-5-20251101-v1:0
codereview ./src --model-id global.anthropic.claude-sonnet-4-5-20250929-v1:0
codereview ./src --aws-region us-east-1
```

**After (v0.2.0) - Recommended:**
```bash
codereview ./src --model opus
codereview ./src --model sonnet
# Region is now configured in models.yaml, not as CLI option
```

**Key Changes:**
- `--model-id` → `--model` (short names)
- `--aws-region` removed (configured in `config/models.yaml`)
- `--aws-profile` still supported for credential selection

### New Features

**List Available Models:**
```bash
codereview --list-models
```

**Use Azure OpenAI:**
```bash
codereview ./src --model gpt-5.2-codex
```

## Library API Changes

### CodeAnalyzer (Recommended)

**Before (v0.1.x):**
```python
from codereview.analyzer import CodeAnalyzer

analyzer = CodeAnalyzer(
    model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
    region="us-west-2",
    temperature=0.1,
)
```

**After (v0.2.0) - Recommended:**
```python
from codereview.analyzer import CodeAnalyzer

analyzer = CodeAnalyzer(
    model_name="opus",
    temperature=0.1,
)
```

**Key Changes:**
- `model_id` + `region` → `model_name` (single parameter)
- Provider auto-detected from model name
- All Bedrock configuration in `config/models.yaml`

### Legacy API Still Works

**Current code (v0.1.x) - Still Works:**
```python
analyzer = CodeAnalyzer(
    model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
    region="us-west-2",
)
```

Shows deprecation warning but continues to function correctly.

## Configuration

### AWS Bedrock (Unchanged)

No changes needed for AWS credentials:
```bash
aws configure
# or
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Azure OpenAI (New)

If using Azure OpenAI, set environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
```

See [README.md](../README.md#azure-openai-configuration) for detailed setup.

## Model Names Mapping

| Old (v0.1.x) | New (v0.2.0) | Aliases |
|--------------|--------------|---------|
| `global.anthropic.claude-opus-4-5-20251101-v1:0` | `opus` | `claude-opus` |
| `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | `sonnet` | `claude-sonnet` |
| `global.anthropic.claude-haiku-4-5-20251001-v1:0` | `haiku` | `claude-haiku` |
| `minimax.minimax-m2` | `minimax` | `minimax-m2` |
| `mistral.mistral-large-3-675b-instruct` | `mistral` | `mistral-large` |
| `moonshot.kimi-k2-thinking` | `kimi` | `kimi-k2` |
| `qwen.qwen3-coder-480b-a35b-v1:0` | `qwen` | `qwen-coder` |

## Testing

If you have tests mocking AWS Bedrock, update them:

**Before (v0.1.x):**
```python
from unittest.mock import patch

with patch("codereview.analyzer.ChatBedrockConverse") as mock_bedrock:
    analyzer = CodeAnalyzer(model_id="...", region="us-west-2")
```

**After (v0.2.0) - Recommended:**
```python
from unittest.mock import Mock, patch

with patch("codereview.analyzer.ProviderFactory") as mock_factory:
    mock_provider = Mock()
    mock_factory.return_value.create_provider.return_value = mock_provider
    analyzer = CodeAnalyzer(model_name="opus")
```

**Benefit:** Provider abstraction makes tests simpler and more maintainable.

## New Features to Explore

### 1. Azure OpenAI Support

```python
# Use GPT models via Azure OpenAI
analyzer = CodeAnalyzer(model_name="gpt-5.2-codex")
```

### 2. Multi-Language Static Analysis

```bash
# Run static analysis for Python, Go, Shell, C++, Java, JS/TS
codereview ./src --static-analysis
```

### 3. List Available Models

```bash
codereview --list-models
```

Output shows all models grouped by provider:
```
Available Models
┌───────────────┬───────────────────┬──────────────┬──────────────────┐
│ ID            │ Name              │ Provider     │ Aliases          │
├───────────────┼───────────────────┼──────────────┼──────────────────┤
│ opus          │ Claude Opus 4.5   │ bedrock      │ opus, claude-opus│
│ gpt-5.2-codex │ GPT-5.2 Codex     │ azure_openai │ gpt52, codex     │
...
```

### 4. Enhanced Configuration

**Edit `config/models.yaml` to:**
- Add custom models
- Adjust pricing
- Configure inference parameters
- Set provider-specific options

Example:
```yaml
providers:
  bedrock:
    region: us-west-2
    models:
      - id: custom-model
        full_id: provider.model-name
        name: Custom Model
        aliases: [custom]
        pricing:
          input_per_million: 1.0
          output_per_million: 5.0
```

## Migration Checklist

### For CLI Users

- [ ] Update commands to use `--model` instead of `--model-id`
- [ ] Remove `--aws-region` from commands (configured in models.yaml)
- [ ] Test with `codereview --list-models` to see available models
- [ ] (Optional) Try Azure OpenAI models if you have access

### For Library Users

- [ ] Update `CodeAnalyzer(model_id=..., region=...)` to `CodeAnalyzer(model_name=...)`
- [ ] Run tests to ensure no regressions
- [ ] Address deprecation warnings
- [ ] (Optional) Update tests to mock `ProviderFactory` instead of `ChatBedrockConverse`

### For Contributors

- [ ] Read updated [CLAUDE.md](../CLAUDE.md) for new architecture
- [ ] Familiarize with provider abstraction pattern
- [ ] Review `config/models.yaml` structure
- [ ] Understand `ProviderFactory` and `ModelProvider` interface

## Troubleshooting

### Deprecation Warnings

If you see:
```
DeprecationWarning: The 'model_id' and 'region' parameters are deprecated.
Use 'model_name' instead.
```

**Solution:** Update to new API:
```python
# Old
analyzer = CodeAnalyzer(model_id="...", region="...")

# New
analyzer = CodeAnalyzer(model_name="opus")
```

### Model Not Found

If you see:
```
ValueError: Unknown model: my-model-id
```

**Solution:** Use short name or list available models:
```bash
codereview --list-models
```

Then use a listed model ID or alias:
```bash
codereview ./src --model opus
```

### Configuration File Not Found

If you see:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../models.yaml'
```

**Cause:** The `config/models.yaml` file is missing from your installation.

**Solution:** Reinstall the package:
```bash
uv pip install -e .
```

Or verify `config/models.yaml` exists in your `codereview/config/` directory.

### Azure Provider Errors

If you see:
```
ValidationError: AZURE_OPENAI_ENDPOINT
```

**Cause:** Azure environment variables not set.

**Solution:** Set required environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
```

## Getting Help

- **Documentation:** [CLAUDE.md](../CLAUDE.md), [README.md](../README.md)
- **Issues:** Open an issue on GitHub
- **Questions:** Check existing issues or start a discussion

## Summary

**Key Takeaways:**
✅ Backward compatible - existing code continues working
✅ Simplified API - use short model names
✅ Multi-provider support - AWS Bedrock and Azure OpenAI
✅ Enhanced configuration - external YAML file
✅ Better testing - mock at provider level

**Recommended Actions:**
1. Update CLI commands to use `--model` with short names
2. Update library code to use `model_name` parameter
3. Explore new features (Azure OpenAI, static analysis, --list-models)
4. Read updated documentation for architecture changes

Welcome to v0.2.0! 🚀
