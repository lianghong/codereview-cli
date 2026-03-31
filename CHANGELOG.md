# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-31

### Added
- **MiniMax M2.5 (Bedrock)**: Agent-native frontier model via AWS Bedrock
  - Model ID: `minimax.minimax-m2.5`
  - Aliases: `minimax-m2.5-bedrock`, `mm2.5-bedrock`
  - Architecture: MoE (230B total, 10B active parameters)
  - Context: 196K tokens, Max output: 128K tokens
  - 80.2% SWE-Bench Verified, 37% faster than M2.1
  - Temperature: 0.5 (optimized for code review without thinking mode)
  - Optimized for task decomposition and complex workflows
- **GLM 5 (Bedrock)**: Frontier-class model for systems engineering via AWS Bedrock
  - Model ID: `zai.glm-5`
  - Aliases: `glm5-bedrock`, `glm-5-bedrock`, `glm5b`
  - Context: 200K tokens, Max output: 128K tokens
  - Temperature: 0.5 (per Zhipu AI recommendations for structured tasks)
  - Optimized for complex systems engineering and long-horizon agentic tasks
  - Multi-step reasoning, AIME-style math, advanced coding, tool-augmented workflows

### Fixed
- **Critical: Azure Provider Syntax Error**: Fixed Python 2 style exception handling (`except ValueError, TypeError:` → `except (ValueError, TypeError):`) that completely blocked Azure OpenAI provider functionality
- **Security: ReDoS Prevention**: Added input validation for user-provided `--exclude` patterns to prevent Regular Expression Denial of Service attacks
  - Max pattern length: 200 characters
  - Max `**` recursion depth: 3
  - Disallow null bytes and malicious patterns
  - Invalid patterns are filtered with warning message

### Changed
- **Model Configuration**: Updated `models.yaml` with comprehensive parameter documentation
  - Added detailed rationale for temperature settings (MiniMax M2.5: why Bedrock uses 0.5 vs NVIDIA's 1.0)
  - Documented thinking mode availability differences between providers
  - Architecture specifications and capability tags for model selection
- **Documentation**: Updated CLAUDE.md with new model information
  - Added MiniMax M2.5 and GLM 5 to model lists and pricing tables
  - Updated supported models count (109 total models)
  - Enhanced parameter documentation with cross-provider comparisons
- **Python 3.14 Compliance**: Adopted PEP 758 unparenthesized exception syntax
  - Updated 7 exception handlers to use modern syntax: `except E1, E2:` instead of `except (E1, E2):`
  - Files updated: `callbacks.py` (2), `readme_finder.py` (3), `static_analysis.py` (2)
  - Correctly retained parentheses for exception handlers using `as` clause (required by PEP 758)
  - Verified PEP 765 compliance: no control flow issues in `finally` blocks
  - All 311 tests pass with Python 3.14.2

### Technical Details
- Total models: 109 (up from 107)
- All 311 tests passing
- Zero static analysis issues (ruff, mypy, isort, vulture)
- Full backward compatibility maintained

### Provider Comparison
**MiniMax M2.5: Bedrock vs NVIDIA**
| Parameter | Bedrock | NVIDIA | Reason |
|-----------|---------|--------|--------|
| Temperature | 0.5 | 1.0 | Bedrock lacks thinking mode |
| Top-p | 0.95 | 0.95 | Same |
| Context | 196K | 196K | Same |
| Thinking Mode | ❌ | ✅ | Platform limitation |

**GLM 5: Bedrock vs NVIDIA**
| Parameter | Bedrock | NVIDIA | Reason |
|-----------|---------|--------|--------|
| Temperature | 0.5 | 0.5 | Model docs recommendation |
| Top-p | 0.95 | 0.95 | Same |
| Context | 200K | 200K | Same |

### Notes
- MiniMax M2.5 and GLM 5 Bedrock pricing TBD (awaiting AWS publication)
- Parameter research based on AWS blog announcement, NVIDIA configurations, and model documentation
- Temperature differences for MiniMax M2.5 are architectural (thinking mode availability), not arbitrary

## [0.2.9] - 2026-03-20

### Added
- **Mistral Small 4 Model**: Added Mistral Small 4 119B via NVIDIA NIM
  - MoE architecture with 256K context, 16K max output
  - Prompt-based JSON parsing (no tool use support)
- **MiniMax M2.5 Model (NVIDIA)**: Added MiniMax M2.5 via NVIDIA NIM
  - 80.2% SWE-Bench Verified
  - 192K context, 128K output
  - Interleaved thinking mode
  - 37% faster than M2.1
- **Prompt-Based JSON Parsing**: Fallback parsing for models without tool use support
  - DeepSeek-R1, Mistral Small 4
  - Maintains structured output reliability

### Fixed
- **Oversized File Handling**: Files exceeding token budget now skipped with warning instead of creating doomed batches
- **Batch Failure Handling**: Clear error messages when all batches fail (rate limits, auth errors)
  - Partial failures now warn that results are incomplete
  - No more misleading "0 issues found" reports
- **Grok 4 Fast Context Fix**: Corrected context window from 2M to 128K to match Azure deployment limit

### Improved
- **Retry Backoff**: Enhanced retry logic for Azure OpenAI and Google GenAI providers
  - 5 retries with longer backoff (10s/20s/40s/60s/60s progression)
  - Total wait time: ~190 seconds
  - Azure respects `Retry-After` headers
- **Plain Text Suggestions**: Improvement Suggestions section renders as plain text without box-drawing characters for clean copy-paste

### Upgraded
- **Dependencies**: Updated to latest versions
  - langchain-aws 1.3.1
  - langsmith 0.7.7
  - google-genai 1.65.0
  - openai 2.24.0
  - websockets 16.0
  - isort 8.0.0
  - ruff 0.15.4
  - mypy 1.19.1

### Testing
- 311 tests passing

## [0.2.8] - 2026-03-15

### Added
- Token-budget-aware batching for efficient context window utilization
- Step 3.5 Flash model via NVIDIA NIM
- GLM-5 model via NVIDIA NIM

### Changed
- Improved file batching logic with token estimation
- Enhanced error messages for provider issues

## [0.2.7] - 2026-03-10

### Added
- Qwen3.5 397B model support via NVIDIA NIM
- Kimi K2.5 model support (Bedrock, Azure, NVIDIA)
- DeepSeek V3.2 model support (Bedrock, NVIDIA)

### Fixed
- Provider credential validation improvements
- Rate limit handling for multiple providers

## [0.2.6] - 2026-03-05

### Added
- Google Generative AI provider (Gemini 3.1 Pro, Gemini 3 Pro, Gemini 3 Flash)
- `--no-color` flag for copy-paste friendly output
- README context discovery with auto-confirmation

### Changed
- Enhanced structured output with `method="json_schema"` for Google GenAI
- Improved retry logic with adaptive backoff

## [0.2.5] - 2026-02-28

### Added
- NVIDIA NIM provider support
- Devstral 2 123B model (72.2% SWE-Bench Verified)
- MiniMax M2, M2.1 models via NVIDIA
- GLM 4.7 model via NVIDIA

### Changed
- Parallel static analysis execution for faster performance
- Improved token estimation for batching

## [0.2.4] - 2026-02-20

### Added
- Azure OpenAI provider support
- GPT-5.3 Codex and GPT-5.4 Pro models
- Grok 4 Fast Reasoning model via Azure
- Responses API support for GPT models

### Fixed
- Rate limit handling for Azure OpenAI
- Retry-After header respect

## [0.2.3] - 2026-02-15

### Added
- Qwen3 Coder 480B model (Bedrock)
- DeepSeek-R1 model (Bedrock)
- MiniMax M2.1 model (Bedrock)

### Changed
- Enhanced prompt engineering for code review
- Improved category normalization

## [0.2.2] - 2026-02-10

### Added
- GLM 4.7 and GLM 4.7 Flash models (Bedrock)
- Context window configuration per model
- Token budget calculation with safety margins

## [0.2.1] - 2026-02-05

### Fixed
- Pydantic V2 compatibility issues
- Category validation for non-Claude models

### Changed
- Enhanced error messages with actionable suggestions

## [0.2.0] - 2026-02-01

### Added
- Multi-provider support (AWS Bedrock foundation)
- Claude Opus 4.6, Sonnet 4.6, Haiku 4.5
- Structured output with Pydantic V2
- Rich terminal UI
- Markdown and JSON export
- Static analysis integration

### Changed
- Migrated from direct Anthropic API to LangChain
- Improved batching logic
- Enhanced retry mechanisms

## [0.1.0] - 2026-01-15

### Added
- Initial release
- Basic code review functionality with Claude
- Python, Go, Shell Script support
- Terminal output
- File scanning and filtering

---

## Release Notes Format

Each release includes:
- **Added**: New features and capabilities
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed in future releases
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes and corrections
- **Security**: Security vulnerability fixes

## Versioning Strategy

- **Major version (X.0.0)**: Breaking changes, major architectural updates
- **Minor version (0.X.0)**: New features, model additions, provider additions
- **Patch version (0.0.X)**: Bug fixes, security patches, documentation updates

---

**Maintained by:** lianghong  
**Repository:** https://github.com/lianghong/codereview-cli  
**License:** MIT
