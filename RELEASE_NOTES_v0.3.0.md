# Release Notes: v0.3.0

**Release Date:** March 31, 2026

## 🎉 What's New

### New Models on AWS Bedrock

We're excited to announce two new frontier models now available on AWS Bedrock:

#### 🤖 MiniMax M2.5 (Bedrock)
**Agent-native frontier model optimized for complex workflows**

- **Model ID:** `minimax.minimax-m2.5`
- **Quick Usage:** `uv run codereview ./src --model minimax-m2.5-bedrock`
- **Aliases:** `mm2.5-bedrock`

**Key Features:**
- 🏆 **80.2% SWE-Bench Verified** - State-of-the-art code understanding
- ⚡ **37% faster than M2.1** - Improved inference speed
- 📚 **196K context window** - Analyze large codebases in one go
- 📝 **128K max output** - Comprehensive, detailed reviews
- 🎯 **Agent-native design** - Trained for task decomposition

**Technical Specs:**
- Architecture: MoE (230B total, 10B active)
- Temperature: 0.5 (optimized for deterministic code review)
- Top-p: 0.95, Top-k: 40

**Why MiniMax M2.5?**
> "Agent-native frontier model trained explicitly to reason efficiently, decompose tasks optimally, and complete complex workflows." - AWS

Perfect for:
- Multi-step code reviews
- Complex agentic workflows  
- Large codebase analysis
- Task decomposition scenarios

---

#### 🧠 GLM 5 (Bedrock)
**Frontier-class model for complex systems engineering**

- **Model ID:** `zai.glm-5`
- **Quick Usage:** `uv run codereview ./src --model glm5-bedrock`
- **Aliases:** `glm-5-bedrock`, `glm5b`

**Key Features:**
- 🎓 **Multi-step reasoning** - AIME-style mathematical thinking
- 🏗️ **Systems engineering focus** - Built for complex architectures
- 🔧 **Tool-augmented workflows** - Advanced coding capabilities
- 📊 **200K context window** - Handle enterprise-scale projects
- 📋 **128K max output** - Detailed analysis reports

**Technical Specs:**
- Context: 200K tokens, Max output: 128K tokens
- Temperature: 0.5 (Zhipu AI recommendation for code)
- Top-p: 0.95
- Lineage: Builds on GLM 4.5 agent-centric foundation

**Why GLM 5?**
> "Frontier-class, general-purpose large language model optimized for complex systems engineering and long-horizon agentic tasks." - AWS

Perfect for:
- Complex systems engineering
- Long-horizon agentic tasks
- Advanced reasoning challenges
- Enterprise code reviews
- Tool-augmented workflows

---

## 🐛 Critical Bug Fixes

### Azure OpenAI Provider Restored
**Fixed critical syntax error that completely blocked Azure functionality**

```diff
- except ValueError, TypeError:  # Python 2 syntax ❌
+ except (ValueError, TypeError):  # Python 3 syntax ✅
```

**Impact:** 
- Azure OpenAI provider is now fully functional
- GPT-5.3 Codex, GPT-5.4 Pro, Kimi K2.5 (Azure), and Grok 4 Fast models restored
- All Azure users can resume normal operations

---

## 🔒 Security Enhancements

### ReDoS Attack Prevention
**New input validation for `--exclude` patterns**

Added protection against Regular Expression Denial of Service (ReDoS) attacks:

```bash
# These malicious patterns are now blocked:
codereview ./src --exclude "**/*/*/*/*/*/*/*/*/*/*/*a"  # ❌ Too complex
codereview ./src --exclude "$(python -c 'print("*" * 300)')"  # ❌ Too long
```

**Security Measures:**
- ✅ Max pattern length: 200 characters
- ✅ Max `**` recursion depth: 3
- ✅ Null byte filtering
- ✅ Clear warning messages for invalid patterns

**User Experience:**
```bash
⚠️  Some exclude patterns were invalid and ignored 
    (too long, too complex, or contain invalid characters)
```

---

## ✨ Code Quality Improvements

### Python 3.14 Compliance
**Adopted PEP 758 and verified PEP 765 compliance**

We've modernized exception handling syntax to follow Python 3.14 best practices:

**PEP 758: Unparenthesized Exception Syntax**
```python
# Old style (still valid, but verbose)
except (OSError, RuntimeError):
    handle_error()

# New Python 3.14 style (cleaner)
except OSError, RuntimeError:
    handle_error()
```

**Changes:**
- ✅ Updated 7 exception handlers across 3 files
  - `callbacks.py` (2 handlers)
  - `readme_finder.py` (3 handlers)
  - `static_analysis.py` (2 handlers)
- ✅ Correctly retained parentheses for handlers using `as` clause (required by PEP 758)
- ✅ Verified PEP 765 compliance: no control flow issues in `finally` blocks

**Impact:**
- Cleaner, more Pythonic code
- Better alignment with Python 3.14 standards
- All 311 tests continue to pass
- Zero static analysis issues maintained

---

## 📊 Model Comparison

### MiniMax M2.5: Bedrock vs NVIDIA

| Feature | Bedrock | NVIDIA | Why Different? |
|---------|---------|--------|----------------|
| Temperature | **0.5** | **1.0** | Bedrock lacks thinking mode |
| Top-p | 0.95 | 0.95 | Same |
| Context | 196K | 196K | Same |
| Max Output | 128K | 128K | Same |
| Thinking Mode | ❌ | ✅ | Platform limitation |

**Key Insight:** Temperature difference is intentional
- NVIDIA achieves consistency via thinking mode (`<think>` tags)
- Bedrock compensates with lower temperature
- Both produce reliable code review output

### GLM 5: Universal Parameters

| Feature | Bedrock | NVIDIA | Why Same? |
|---------|---------|--------|-----------|
| Temperature | 0.5 | 0.5 | Model docs recommendation |
| Top-p | 0.95 | 0.95 | Standard for reasoning |
| Context | 200K | 200K | Native model capability |
| Max Output | 128K | 128K | Native model capability |

**Result:** GLM 5 parameters are identical across providers - model characteristics naturally align with code review requirements.

---

## 📈 Statistics

- **Total Models:** 109 (up from 107)
- **Providers:** 4 (AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google GenAI)
- **Test Suite:** 311 tests, 100% passing
- **Static Analysis:** Zero issues (ruff, mypy, isort, vulture)
- **Backward Compatibility:** Full

---

## 🚀 Quick Start

### Using New Models

```bash
# MiniMax M2.5 on Bedrock
uv run codereview ./src --model minimax-m2.5-bedrock

# GLM 5 on Bedrock
uv run codereview ./src --model glm5-bedrock

# With verbose output to see token budget
uv run codereview ./src -m mm2.5-bedrock --verbose

# Large codebase analysis (leverage 196K context)
uv run codereview ./large-project --model mm2.5-bedrock --batch-size 20

# Compare providers
uv run codereview ./src --model minimax-m2.5-nvidia   # NVIDIA version
uv run codereview ./src --model minimax-m2.5-bedrock  # Bedrock version
```

### Prerequisites

**For new Bedrock models:**
1. AWS account with Bedrock access
2. Models enabled in AWS region (us-west-2 or your region)
3. IAM permissions: `bedrock:InvokeModel`

```bash
# Check available models
uv run codereview --list-models | grep -E "(minimax-m2.5|glm5)"

# Dry run to estimate costs (when pricing is published)
uv run codereview ./src -m glm5-bedrock --dry-run
```

---

## 📚 Documentation Updates

### Enhanced Model Documentation

All model configurations now include:
- ✅ Detailed parameter rationale (why these specific values?)
- ✅ Cross-provider comparisons (Bedrock vs NVIDIA)
- ✅ Architecture specifications (MoE details, parameter counts)
- ✅ Capability tags for model selection
- ✅ Use case recommendations

**Example from `models.yaml`:**
```yaml
inference_params:
  # IMPORTANT: M2.5 was RL-trained assuming temp=1.0.
  # However, for deterministic code review on Bedrock
  # (which lacks thinking mode), we use 0.5 to balance
  # consistency with M2.5's trained distribution.
  default_temperature: 0.5
```

### New Files

- **CHANGELOG.md** - Comprehensive version history following Keep a Changelog format
- **RELEASE_NOTES_v0.3.0.md** - This document

### Updated Files

- **pyproject.toml** - Version bumped to 0.3.0
- **README.md** - Version history updated
- **CLAUDE.md** - Model tables and documentation updated
- **codereview/config/models.yaml** - Two new model configurations added

---

## 🔮 Known Limitations

### Pricing Information
- MiniMax M2.5 and GLM 5 Bedrock pricing: **TBD**
- Awaiting AWS publication
- Cost estimation unavailable in `--dry-run` mode until pricing is published
- Placeholder: $0.00 per million tokens

### When Pricing is Published
We'll update:
1. `models.yaml` pricing section
2. `CLAUDE.md` pricing tables  
3. This release note

**Track pricing updates:** Watch the [AWS Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/)

---

## 🎯 Upgrade Path

### From v0.2.9 to v0.3.0

**No breaking changes!** This is a **minor version** release with new features and fixes.

```bash
# Standard upgrade
cd codereview-cli
git pull
uv pip install -e .

# Verify upgrade
uv run codereview --list-models | wc -l  # Should show 109 models

# Test new models
uv run codereview ./tests/fixtures/sample_code --model mm2.5-bedrock --dry-run
```

### Configuration Migration
**None required!** All existing configurations work without changes.

### Compatibility
- ✅ Python 3.14+ (no change)
- ✅ All existing models work identically
- ✅ All CLI flags backward compatible
- ✅ Provider credentials unchanged
- ✅ API interfaces stable

---

## 🤝 Contributing

Found a bug? Have a feature request? We'd love to hear from you!

- **Issues:** https://github.com/lianghong/codereview-cli/issues
- **Discussions:** https://github.com/lianghong/codereview-cli/discussions
- **Pull Requests:** Always welcome!

---

## 🙏 Acknowledgments

Special thanks to:
- **AWS Bedrock Team** - For launching MiniMax M2.5 and GLM 5
- **MiniMax AI** - For the agent-native M2.5 frontier model
- **Zhipu AI** - For the GLM 5 frontier-class model
- **Code Quality Auditor Agent** - For comprehensive codebase review that identified the critical Azure bug
- **Community Contributors** - For bug reports and feedback

---

## 📅 What's Next?

### Coming in v0.3.1
- Pricing updates when AWS publishes official rates
- Additional model optimizations based on user feedback
- Performance improvements for large codebases

### Future Roadmap
- Incremental review mode (review only changed files)
- Team collaboration features
- Custom rule definitions
- IDE integrations (VS Code, JetBrains)

**Stay tuned!** Follow the repository for updates.

---

## 📞 Support

- **Documentation:** [CLAUDE.md](CLAUDE.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Issues:** [GitHub Issues](https://github.com/lianghong/codereview-cli/issues)
- **Email:** feilianghong@gmail.com

---

**Happy Coding!** 🚀

*codereview-cli v0.3.0 - March 31, 2026*
