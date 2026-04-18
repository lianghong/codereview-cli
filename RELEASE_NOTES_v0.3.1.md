# Release Notes - Version 0.3.1

**Release Date:** April 18, 2026  
**Status:** Stable Release  
**Breaking Changes:** None

---

## 🎉 Highlights

### Claude Opus 4.7 Support
The latest reasoning model from Anthropic is now available with enhanced capabilities:
- **Adaptive Thinking** - Dynamically allocates reasoning resources
- **32K Output Tokens** - Larger responses for comprehensive reviews
- **Enhanced Reasoning** - Improved code understanding and analysis
- **Regional Availability** - US East (N. Virginia) and Asia Pacific (Tokyo)

### Bug Fixes
- Fixed `--no-color` flag to work consistently across all CLI commands
- Improved error handling for reasoning models that don't support temperature

---

## 📦 What's New

### Added Features

#### 1. Claude Opus 4.7 Integration
```bash
# Now the default model
codereview ./src

# Explicit usage
codereview ./src --model opus4.7
codereview ./src --model claude-opus-4.7
```

**Model Specifications:**
- **Model ID:** `us.anthropic.claude-opus-4-7`
- **Aliases:** `opus4.7`, `claude-opus-4.7`, `opus-4.7`, `claude-opus-47`
- **Type:** Reasoning model (no temperature parameter)
- **Max Output:** 32,000 tokens
- **Context Window:** 200,000 tokens
- **Pricing:** $5.00/M input, $25.00/M output
- **Capabilities:** code_review, security_analysis, reasoning, computer_use, adaptive_thinking

#### 2. PEP 758 Documentation
Added clarifying comments to exception handlers to explain Python 3.14+ syntax:
```python
# PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
except OSError, RuntimeError:
    pass
```

**Benefits:**
- Clearer intent for contributors
- Prevents misinterpretation as Python 2 syntax
- Improves code maintainability

---

## 🐛 Bug Fixes

### 1. --no-color Flag Consistency
**Issue:** `--no-color` flag was ignored by `--list-models` and `--validate` commands

**Fix:** 
- Removed module-level Console instance
- Console now created early and passed to all helper functions
- All CLI output now respects user-specified flags

**Before:**
```bash
codereview --list-models --no-color  # Still had ANSI codes
```

**After:**
```bash
codereview --list-models --no-color  # Clean plain text output
```

### 2. Reasoning Model Temperature Handling
**Issue:** Claude Opus 4.7 returned error: "temperature is deprecated for this model"

**Fix:**
- Bedrock provider now conditionally omits temperature parameter
- Automatically detects reasoning models (config has no temperature)
- Prevents ValidationException errors

**Technical Implementation:**
```python
# Only add temperature if model supports it
if self.temperature is not None:
    model_kwargs["temperature"] = self.temperature
```

---

## 🔄 Changes

### Default Model Update
- **Previous:** Claude Opus 4.6 (`opus`)
- **Current:** Claude Opus 4.7 (`opus4.7`)
- **Note:** All existing model aliases continue to work

### Documentation Updates
- **CLAUDE.md:** Added "Recent Updates" section
- **README.md:** Added "What's New" section
- **CHANGELOG.md:** Comprehensive v0.3.1 changelog
- **Model Tables:** Updated with Opus 4.7 specifications

---

## 🧪 Quality Assurance

### Test Results
```
✅ 311/311 tests passing (100%)
✅ Zero security vulnerabilities (Bandit: 5,656 lines)
✅ Zero linting issues (Ruff)
✅ Zero type errors (Mypy: 22 files)
✅ Zero dead code (Vulture: 80% confidence)
```

### Manual Testing
```bash
✅ codereview --list-models
✅ codereview --list-models --no-color
✅ codereview --validate --model opus4.7
✅ codereview --model opus4.7 --dry-run
✅ codereview /path/to/code --model opus4.7
```

---

## 📚 Upgrade Guide

### Installation

#### New Installation
```bash
git clone https://github.com/yourusername/codereview-cli.git
cd codereview-cli
uv venv --python 3.14
uv pip install -e .
```

#### Upgrading from v0.3.0
```bash
cd codereview-cli
git pull
uv pip install -e .
```

### Migration

**No breaking changes** - all existing functionality continues to work.

#### Using the New Default (Opus 4.7)
```bash
# Automatic - uses new default
codereview ./src
```

#### Staying on Opus 4.6
```bash
# Explicit model selection
codereview ./src --model opus
```

#### Testing the New Model
```bash
# Dry run to see cost estimate
codereview ./src --model opus4.7 --dry-run

# Validate credentials
codereview --validate --model opus4.7
```

---

## 📊 Performance Comparison

### Claude Opus 4.7 vs 4.6

| Feature | Opus 4.6 | Opus 4.7 |
|---------|----------|----------|
| **Reasoning** | Standard | Enhanced with adaptive thinking |
| **Max Output** | 128K tokens | 32K tokens |
| **Temperature** | Supported (0.1) | Not supported (reasoning model) |
| **Model Type** | Standard | Reasoning model |
| **Pricing** | $5/$25 per M tokens | $5/$25 per M tokens |
| **Availability** | Global regions | US East, Asia Pacific Tokyo |

### When to Use Opus 4.7
- ✅ Complex code analysis requiring deep reasoning
- ✅ Security vulnerability detection
- ✅ Architectural review and design insights
- ✅ Multi-file dependency analysis
- ✅ Production-ready code requiring high quality reviews

### When to Use Opus 4.6
- ✅ Large codebases requiring 128K output tokens
- ✅ When global region availability is needed
- ✅ When temperature control is desired

---

## 🔧 Technical Details

### Files Modified (9)

#### Configuration Files
1. **models.yaml** - Added Opus 4.7 configuration, updated default
2. **pyproject.toml** - Version bumped to 0.3.1
3. **CLAUDE.md** - Added recent updates section, model specifications
4. **README.md** - Added "What's New" section
5. **CHANGELOG.md** - Comprehensive v0.3.1 changes

#### Source Code
6. **cli.py** - Fixed console parameter passing
7. **bedrock.py** - Temperature handling for reasoning models
8. **callbacks.py** - PEP 758 comments (2 locations)
9. **azure_openai.py** - PEP 758 comment (1 location)
10. **readme_finder.py** - PEP 758 comments (3 locations)
11. **static_analysis.py** - PEP 758 comments (2 locations)

### Key Code Changes

#### Bedrock Provider Enhancement
```python
# New: Conditional temperature handling
if model_config.inference_params is None:
    self.temperature = 0.1  # Default
elif model_config.inference_params.temperature is not None:
    self.temperature = model_config.inference_params.temperature
else:
    self.temperature = None  # Reasoning model

# Build kwargs and only add temperature if supported
model_kwargs = {...}
if self.temperature is not None:
    model_kwargs["temperature"] = self.temperature
```

#### CLI Console Fix
```python
# New: Create console early and pass to helpers
con = _create_console(quiet=quiet, no_color=no_color)

if list_models:
    display_available_models(con)  # Now receives console
    
if validate:
    validate_provider_credentials(model_name, aws_profile, con)
```

---

## 🌐 Additional Resources

### Documentation
- [AWS Blog: Claude Opus 4.7 Announcement](https://aws.amazon.com/blogs/aws/introducing-anthropics-claude-opus-4-7-model-in-amazon-bedrock/)
- [Anthropic Documentation](https://docs.anthropic.com/)
- [PEP 758: Unparenthesized Exceptions](https://peps.python.org/pep-0758/)

### Getting Help
- **GitHub Issues:** Report bugs or request features
- **Documentation:** See CLAUDE.md for developer guidance
- **Examples:** Check `examples/` directory for usage patterns

---

## 🙏 Acknowledgments

Special thanks to:
- **Anthropic** for releasing Claude Opus 4.7
- **AWS Bedrock** team for quick model availability
- **Contributors** who reported the `--no-color` flag issue
- **Community** for continued feedback and support

---

## 🔮 What's Next

### Planned for v0.3.2
- Performance optimizations for large codebases
- Additional static analysis tool integrations
- Enhanced error messages and debugging

### Future Roadmap
- Support for more LLM providers
- Custom prompt templates
- Integration with CI/CD platforms
- VS Code extension

---

## 📝 Changelog Summary

```
[0.3.1] - 2026-04-18

Added:
  ✅ Claude Opus 4.7 support (reasoning model)
  ✅ PEP 758 clarification comments

Fixed:
  ✅ --no-color flag consistency
  ✅ Temperature handling for reasoning models

Changed:
  ✅ Default model updated to Opus 4.7
  ✅ Documentation updates

Technical:
  ✅ 311 tests passing
  ✅ Zero security issues
  ✅ Zero linting errors
```

---

**For the complete changelog, see [CHANGELOG.md](CHANGELOG.md)**

**For migration details, see the Migration Guide section above**

**For developer documentation, see [CLAUDE.md](CLAUDE.md)**
