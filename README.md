# Code Review CLI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> AI-powered code review tool with multiple LLM providers (AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google Generative AI)

## 🎉 What's New (Unreleased)

- ✅ **3 new providers**: DeepSeek direct API (`deepseek-v4-pro`, `deepseek-v4-flash`), Z.AI (`zhipuai/glm-5.1`), Moonshot/Kimi (`kimi-k2.6`). 7 providers total now.
- ✅ **GPT-5.4 (Azure)** — frontier reasoning model, 1.05M context, default Azure model
- ✅ **DeepSeek-V4-Pro (Azure)** — 1M context with prompt-based JSON parsing for tool-use-less Foundry deployments (`supports_tool_use: false` now wired into the Azure provider)
- ✅ **`--tool-timeout`** — override the static-analysis subprocess timeout (default 120s) for slow C++/mypy runs
- ✅ **`--include-hidden`** — opt-in scanning of `.github/scripts`, `.config/`, etc.
- ✅ **Reproducible static analysis** — file lists sorted before truncation so CI runs are deterministic (locked in by regression test)
- ✅ **Accurate issue counts** — ruff/mypy/bandit summary-line parsing replaces the old substring-match heuristic
- ✅ **Supply-chain hardening** — static-analysis tools resolved via `shutil.which()`; binaries inside the analyzed directory are refused (gofmt cache-bypass also fixed)
- ✅ **AWS error redaction** — STS/Bedrock validation errors no longer leak SCP fragments or IAM policy details
- ✅ **All 368 tests passing**, ruff/format/mypy clean

A LangChain-based CLI tool that provides comprehensive, intelligent code reviews for Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript projects using Claude, GPT-5.4, Gemini, DeepSeek-V4-Pro, Kimi K2.6, GLM-5.1, and other leading models through AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google Generative AI, DeepSeek, Z.AI, and Moonshot.

## Features

- **Multi-Provider Support** (7 providers): AWS Bedrock (Claude, Minimax, Kimi, Qwen), Azure OpenAI (GPT-5.4, GPT-5.4 Pro, DeepSeek-V4-Pro, Kimi K2.5), NVIDIA NIM (Mistral, MiniMax, Kimi, Qwen, DeepSeek-V4-Pro, GLM-5/5.1, Step), Google GenAI (Gemini 3.1 Pro / 3 Pro / 3 Flash), DeepSeek direct (V4-Pro, V4-Flash), Z.AI (GLM-5.1), and Moonshot direct (Kimi K2.6)
- **AI-Powered Analysis**: Leverages Claude Opus 4.8, GPT-5.4, DeepSeek-V4-Pro, Kimi K2.6, GLM-5.1, Gemini 3.1 Pro, and other leading models for deep code understanding
- **Multi-Language Support**: Reviews Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript codebases
- **Smart Batching**: Automatically groups files for efficient token usage
- **Structured Output**: Get categorized issues with severity levels and actionable suggestions
- **Static Analysis Integration**: Combine AI review with ruff, mypy, black, eslint, and other tools
- **Architectural Review**: Detects boundary violations, coupling issues, and layering leaks
- **Operational Readiness**: Checks for missing error handling, timeouts, and observability gaps
- **Testing Quality**: Identifies test anti-patterns and coverage gaps
- **Terminal UI**: Rich, colorful terminal output with progress indicators (`--no-color` for copy-paste friendly output)
- **Markdown/JSON Export**: Generate shareable reports in Markdown or JSON format for CI/CD
- **Error Handling**: Robust retry logic with exponential backoff for API rate limits
- **Flexible Configuration**: Customize file size limits, exclusion patterns, and provider settings

## Installation

### Prerequisites

- Python 3.14+
- **At least one of the following:**
  - AWS account with Bedrock access (Claude, Minimax, Kimi, Qwen models)
  - Azure OpenAI resource with model deployment (GPT-5.4, GPT-5.4 Pro, DeepSeek-V4-Pro, Kimi K2.5) — `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
  - NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) — `NVIDIA_API_KEY` (Mistral, MiniMax, Kimi, Qwen, DeepSeek-V4-Pro, GLM-5/5.1, Step; free tier available)
  - Google API key from [AI Studio](https://aistudio.google.com/apikey) — `GOOGLE_API_KEY` (Gemini 3.1 Pro / 3 Pro / 3 Flash)
  - DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com/api_keys) — `DEEPSEEK_API_KEY` (V4-Pro, V4-Flash)
  - Z.AI API key from [z.ai](https://z.ai) — `ZAI_API_KEY` (GLM-5.1, international)
  - Moonshot/Kimi API key from [platform.moonshot.cn](https://platform.moonshot.cn) — `KIMI_API_KEY` (Kimi K2.6; international keys from `platform.moonshot.ai` work too — override `base_url`)

### Install with uv (recommended)

```bash
# Clone the repository
git clone https://github.com/lianghong/codereview-cli.git
cd codereview-cli

# Create virtual environment
uv venv --python 3.14

# Install the package
uv pip install -e .
```

### Install with pip

```bash
pip install -e .
```

## AWS Configuration

### 1. Configure AWS Credentials

Choose one of the following methods:

**Option A: AWS CLI**
```bash
aws configure
```

**Option B: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

**Option C: AWS Profile**
```bash
codereview /path/to/code --aws-profile your-profile
```

### 2. Enable Bedrock Access

1. Go to AWS Console > Bedrock
2. Navigate to "Model access" in your region
3. Request access to "Anthropic Claude Opus 4.6"
4. Wait for approval (usually instant for supported regions)

### 3. Verify IAM Permissions

Ensure your IAM user/role has the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-opus-*"
    }
  ]
}
```

## Azure OpenAI Configuration (Alternative to AWS)

Azure OpenAI provides access to GPT-5.4, GPT-5.4 Pro, DeepSeek-V4-Pro, and Kimi K2.5 via Microsoft Azure AI Foundry.

### 1. Set Environment Variables

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### 2. Deploy Models in Azure AI Foundry

1. Create an Azure OpenAI resource in Azure Portal
2. Deploy models from Azure AI Foundry catalog:
   - **GPT-5.4** (deployment name: `gpt-5.4`) - Frontier reasoning model, 1.05M context, 128K output
   - **GPT-5.4 Pro** (deployment name: `gpt-5.4-pro`) - Deeper reasoning variant, 1.05M context
   - **DeepSeek-V4-Pro** (deployment name: `DeepSeek-V4-Pro`) - 1M context, chain-of-thought reasoning (no tool calling on Foundry)
   - **Kimi K2.5** (deployment name: `Kimi-K2.5`) - Moonshot AI's multimodal MoE model
3. Note your deployment name, endpoint, and API key

### 3. Use Azure Models

```bash
# GPT-5.4 - Flagship reasoning model, 1.05M context (default Azure model)
codereview /path/to/code --model gpt

# GPT-5.4 Pro - Deeper reasoning variant, 1.05M context
codereview /path/to/code --model gpt-pro

# DeepSeek-V4-Pro - 1M context, prompt-based JSON parsing
codereview /path/to/code --model dsv4-azure

# Kimi K2.5 - Multimodal MoE, 256K context
codereview /path/to/code --model kimi-azure
```

### 4. Test Connection

```bash
codereview --list-models  # Should show Azure models
```

**Note:** Azure OpenAI models require you to deploy them in your Azure resource first. The deployment names in your configuration must match your actual Azure deployments. Kimi K2.5 and DeepSeek-V4-Pro are available as "Direct from Azure" models in the Azure AI Foundry catalog. DeepSeek-V4-Pro on Foundry doesn't support tool calling, so the provider falls back to prompt-based JSON parsing automatically (`supports_tool_use: false` in `models.yaml`).

## NVIDIA NIM Configuration (Alternative Provider)

NVIDIA NIM provides access to Mistral Small 4, Mistral Medium 3.5, MiniMax M2.7, Kimi K2.6, Qwen3 Coder, Qwen3.5, DeepSeek-V4-Pro/Flash, GLM-5.1, Step 3.5/3.7 Flash, and more — with a free tier for development.

### 1. Get API Key

1. Visit [NVIDIA Build](https://build.nvidia.com/explore/discover)
2. Sign in or create an account
3. Generate an API key (format: `nvapi-xxxxx...`)

### 2. Set Environment Variable

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

### 3. Use NVIDIA Models

```bash
# Mistral Small 4 - MoE model with 256K context
codereview /path/to/code --model mistral-small

# Mistral Medium 3.5 128B - 77.6% SWE-Bench, per-request reasoning_effort
codereview /path/to/code --model mistral-medium

# DeepSeek-V4-Pro - 1M context, three reasoning modes (free on NVIDIA)
codereview /path/to/code --model dsv4-nvidia

# DeepSeek-V4-Flash - 1M context, fast/cheap sibling of V4-Pro (free on NVIDIA)
codereview /path/to/code --model dsv4-flash-nvidia

# Qwen3 Coder - Ultra-large coding model with thinking mode
codereview /path/to/code --model qwen-nvidia

# Qwen3.5 - Next-gen Qwen reasoning model with thinking mode (262K context)
codereview /path/to/code --model qwen3.5

# MiniMax M2.7 - Agent-native model with thinking mode (56.22% SWE-Pro, 204K context)
# (NVIDIA retired the M2.5 endpoint 2026-05-12; minimax-m2.5/mm25 now route here)
codereview /path/to/code --model minimax-m2.7

# Kimi K2.6 - 262K context, thinking mode
# (NVIDIA shut down the K2.5 endpoint 2026-05-20; kimi-k2.5/kimi25 now route here)
codereview /path/to/code --model kimi-k2.6

# GLM-5.1 - Zhipu reasoning model
# (NVIDIA deprecated glm5 2026-04-20; glm5/glm-5 now route to GLM-5.1)
codereview /path/to/code --model glm51

# Step 3.5 Flash - Cost-efficient reasoning, fast
codereview /path/to/code --model step-3.5-flash

# Step 3.7 Flash - Newer 256K multimodal sibling with reasoning levels
codereview /path/to/code --model step-3.7-flash
```

**Note:** NVIDIA NIM models are currently in free tier. No charges apply during the preview period. Models with thinking mode enabled (MiniMax M2.7, Qwen3.5, DeepSeek-V4-Pro/Flash, Qwen3 Coder, GLM-5.1, Kimi K2.6) provide deeper reasoning for complex code analysis.

## Google Generative AI Configuration (Alternative Provider)

Google Generative AI provides access to Gemini 3.1 Pro and Gemini 3 models with 1M token context windows.

### 1. Get API Key

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Create an API key

### 2. Set Environment Variable

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 3. Use Gemini Models

```bash
# Gemini 3.1 Pro - Most advanced reasoning model (1M context)
# (Google shut down gemini-3-pro 2026-03-09; gemini-3-pro now routes here)
codereview /path/to/code --model gemini-3.1-pro

# Gemini 3 Flash - Fast and cost-efficient (1M context)
codereview /path/to/code --model gemini-3-flash
```

## DeepSeek Direct API Configuration (Alternative Provider)

DeepSeek's direct API exposes V4-Pro and V4-Flash via an OpenAI-compatible endpoint. Both models support tool calling and structured output natively.

### 1. Get API Key

Sign up at [platform.deepseek.com](https://platform.deepseek.com/api_keys) and create an API key.

### 2. Set Environment Variable

```bash
export DEEPSEEK_API_KEY="your-deepseek-key"
```

### 3. Use DeepSeek Models

```bash
# DeepSeek V4-Pro - flagship, 1M context
codereview /path/to/code --model deepseek-v4-pro

# DeepSeek V4-Flash - cost-efficient, 1M context, 12x cheaper input
codereview /path/to/code --model deepseek-v4-flash
```

## Z.AI (Zhipu) Configuration (Alternative Provider)

Z.AI is Zhipu's international platform exposing GLM-5.1 (long-horizon coding model, 203K context) via an OpenAI-compatible endpoint. The CLI integrates via `langchain-openai`'s `ChatOpenAI` with a custom base URL — no langchain-community dependency.

### 1. Get API Key

Sign up at [z.ai](https://z.ai) and create an API key.

### 2. Set Environment Variable

```bash
export ZAI_API_KEY="your-zai-key"
```

### 3. Use Z.AI Models

```bash
# GLM-5.1 - long-horizon coding, 203K context
codereview /path/to/code --model zhipuai/glm-5.1
codereview /path/to/code --model zai-glm  # short alias
```

## Moonshot AI (Kimi) Configuration (Alternative Provider)

Moonshot's direct API exposes Kimi K2.6 (1T MoE, 32B active, 256K context, agentic-coding optimized) via the dedicated `langchain-moonshot` package.

**Two separate platforms with separate accounts/keys:**
- `platform.moonshot.cn` — Chinese platform, default in this CLI (matches `KIMI_API_KEY` naming convention)
- `platform.moonshot.ai` — International platform; override `base_url` if your key is from here

### 1. Get API Key

Sign up at [platform.moonshot.cn](https://platform.moonshot.cn) (or `.ai` for international) and create an API key.

### 2. Set Environment Variable

```bash
export KIMI_API_KEY="your-moonshot-key"
```

### 3. Use Moonshot Models

```bash
# Kimi K2.6 - 1T MoE, 256K context
codereview /path/to/code --model kimi-k2.6
codereview /path/to/code --model kimi  # short alias (canonical)
```

If your key is from the international platform (`platform.moonshot.ai`), override the endpoint in `codereview/config/models.yaml`:
```yaml
moonshot:
  base_url: "https://api.moonshot.ai/v1"
```

## Usage

### Basic Usage

```bash
# Uses Claude Opus 4.8 by default
codereview /path/to/your/codebase
```

### Choose Your Model

```bash
# List all available models
codereview --list-models

# AWS Bedrock Models (Claude family)
codereview /path/to/code --model fable5    # Claude Fable 5 (Mythos-class, 1M context)
codereview /path/to/code --model opus4.8   # Claude Opus 4.8 (latest, default, 1M context)
codereview /path/to/code --model opus4.7   # Claude Opus 4.7 (reasoning, 200K context)
codereview /path/to/code --model opus      # Claude Opus 4.6
codereview /path/to/code --model sonnet    # Claude Sonnet 4.6 (balanced)
codereview /path/to/code --model haiku     # Claude Haiku 4.5 (fastest)

# AWS Bedrock (other providers)
codereview /path/to/code --model kimi-k2.5-bedrock  # Kimi K2.5 (262K context)
codereview /path/to/code --model qwen-bedrock       # Qwen3 Coder 480B
codereview /path/to/code --model qwen-next-bedrock  # Qwen3 Coder Next (80B MoE)
codereview /path/to/code --model minimax-m2.5-bedrock  # MiniMax M2.5 (196K context)
codereview /path/to/code --model glm5-bedrock       # GLM 5 (Bedrock)

# Azure OpenAI Models
codereview /path/to/code --model gpt              # GPT-5.4 (1.05M context, frontier reasoning)
codereview /path/to/code --model gpt-pro          # GPT-5.4 Pro (deeper reasoning variant)
codereview /path/to/code --model dsv4-azure       # DeepSeek-V4-Pro (1M context, no tool use)
codereview /path/to/code --model kimi-azure       # Kimi K2.5 (256K context)

# NVIDIA NIM Models (free tier)
codereview /path/to/code --model mistral-small      # Mistral Small 4 119B
codereview /path/to/code --model mistral-medium     # Mistral Medium 3.5 128B (77.6% SWE-Bench)
codereview /path/to/code --model minimax-m2.7       # MiniMax M2.7 (thinking, agent-native)
codereview /path/to/code --model dsv4-nvidia        # DeepSeek-V4-Pro on NVIDIA (free)
codereview /path/to/code --model dsv4-flash-nvidia  # DeepSeek-V4-Flash on NVIDIA (free, 1M context, fast)
codereview /path/to/code --model qwen-nvidia        # Qwen3 Coder 480B (thinking)
codereview /path/to/code --model qwen3.5            # Qwen3.5 397B (262K context)
codereview /path/to/code --model glm51              # GLM-5.1 (744B MoE; supersedes deprecated glm5)
codereview /path/to/code --model kimi-nvidia-26     # Kimi K2.6 on NVIDIA (free; supersedes K2.5)
codereview /path/to/code --model step-3.5-flash     # Step 3.5 Flash
codereview /path/to/code --model step-3.7-flash     # Step 3.7 Flash (256K, multimodal, newer)

# Google Generative AI Models
codereview /path/to/code --model gemini-3.1-pro     # Gemini 3.1 Pro (1M context)
codereview /path/to/code --model gemini-3-flash     # Gemini 3 Flash (fast, cheap)

# DeepSeek Direct API
codereview /path/to/code --model deepseek-v4-pro    # Flagship, 1M context
codereview /path/to/code --model deepseek-v4-flash  # 12x cheaper input, 1M context

# Z.AI (Zhipu international)
codereview /path/to/code --model zhipuai/glm-5.1    # Long-horizon coding, 203K context
codereview /path/to/code --model zai-glm            # Short alias

# Moonshot direct API (Kimi)
codereview /path/to/code --model kimi-k2.6          # Canonical, 256K context, 1T MoE
codereview /path/to/code --model kimi               # Short alias

# Short aliases work too
codereview /path/to/code -m haiku
codereview /path/to/code -m gpt
codereview /path/to/code -m kimi
```

**Model Comparison:**

| Model | Provider | Use Case | Input $/M | Output $/M |
|-------|----------|----------|-----------|------------|
| Opus 4.8 | AWS Bedrock | Latest reasoning, default model, 1M context | $5.00 | $25.00 |
| Opus 4.7 | AWS Bedrock | Reasoning, 200K context | $5.00 | $25.00 |
| Opus 4.6 | AWS Bedrock | Highest quality, critical reviews | $5.00 | $25.00 |
| Sonnet 4.6 | AWS Bedrock | Balanced performance and cost | $3.00 | $15.00 |
| Haiku 4.5 | AWS Bedrock | Fast, economical, large codebases | $1.00 | $5.00 |
| Kimi K2.5 (Bedrock) | AWS Bedrock | 262K context, MoE | TBD | TBD |
| MiniMax M2.5 (Bedrock) | AWS Bedrock | 196K context, agent-native | TBD | TBD |
| Qwen3 Coder 480B | AWS Bedrock | Ultra-large coding model | TBD | TBD |
| GPT-5.4 | Azure OpenAI | Frontier reasoning, 1.05M context, default Azure | $2.50 | $15.00 |
| GPT-5.4 Pro | Azure OpenAI | Deeper reasoning, hardest problems | $30.00 | $180.00 |
| DeepSeek-V4-Pro (Azure) | Azure OpenAI | 1M context, prompt-based JSON (no tool use on Foundry) | $1.74 | $3.48 |
| Kimi K2.5 (Azure) | Azure OpenAI | Multimodal MoE, 256K context | $0.60 | $3.00 |
| Mistral Small 4 | NVIDIA NIM | 256K context, MoE architecture | Free* | Free* |
| Mistral Medium 3.5 | NVIDIA NIM | 128B dense, 256K context, reasoning_effort, 77.6% SWE-Bench | Free* | Free* |
| MiniMax M2.7 | NVIDIA NIM | 204K context, 128K output, thinking mode, agent-native (supersedes retired M2.5) | Free* | Free* |
| DeepSeek-V4-Pro (NVIDIA) | NVIDIA NIM | 1M context, three reasoning modes | Free* | Free* |
| DeepSeek-V4-Flash (NVIDIA) | NVIDIA NIM | 1M context, fast/cheap sibling of V4-Pro | Free* | Free* |
| Qwen3 Coder (NIM) | NVIDIA NIM | Ultra-large coding, thinking mode | Free* | Free* |
| Qwen3.5 397B | NVIDIA NIM | Next-gen Qwen, thinking mode, 262K context | Free* | Free* |
| GLM-5.1 | NVIDIA NIM | 744B MoE, 131K context, thinking (supersedes deprecated GLM-5) | Free* | Free* |
| Kimi K2.6 | NVIDIA NIM | 262K context, thinking mode (supersedes retired K2.5) | Free* | Free* |
| Step 3.5 Flash | NVIDIA NIM | Cost-efficient reasoning | Free* | Free* |
| Step 3.7 Flash | NVIDIA NIM | 256K multimodal, reasoning levels (newer) | Free* | Free* |
| Gemini 3.1 Pro | Google GenAI | Most advanced reasoning, 1M context (supersedes retired 3 Pro) | $2.00 | $12.00 |
| Gemini 3 Flash | Google GenAI | Fast and cheap, 1M context | $0.50 | $3.00 |
| **DeepSeek-V4-Pro** | **DeepSeek direct** | **1M context, three reasoning modes, tool calling** | **$1.74** | **$3.48** |
| **DeepSeek-V4-Flash** | **DeepSeek direct** | **1M context, 12x cheaper input than V4-Pro** | **$0.14** | **$0.28** |
| **GLM-5.1 (Z.AI)** | **Z.AI direct** | **Long-horizon coding, 203K context, function calling** | **$1.40** | **$4.40** |
| **Kimi K2.6** | **Moonshot direct** | **1T MoE, 32B active, 256K context, agentic** | **$0.60** | **$2.50** |
| Qwen3 Coder (Bedrock) | AWS Bedrock | Ultra-large model, deep analysis | $0.22 | $1.40 |
| Qwen3 Coder Next (Bedrock) | AWS Bedrock | Ultra-sparse MoE, 70%+ SWE-bench | $0.50 | $1.20 |
| GLM 5 (Bedrock) | AWS Bedrock | Zhipu next-gen reasoning | TBD | TBD |

*NVIDIA NIM models are currently in free preview tier. Models with thinking mode use interleaved reasoning for deeper code analysis. Several Bedrock models display "TBD" until AWS publishes official pricing — the CLI renders unpriced models as `Estimated cost: TBD` instead of `$0.0000`.

### Export Reports

```bash
# Export to Markdown (default)
codereview /path/to/code --output review-report.md

# Export to JSON for CI/CD pipelines
codereview /path/to/code --output review-report.json --format json
```

### Filter by Severity

```bash
# Show only critical and high severity issues
codereview /path/to/code --severity high
```

### Limit Files

```bash
# Analyze only first 50 files
codereview /path/to/code --max-files 50
```

### Custom File Size Limit

```bash
# Only analyze files under 20KB
codereview /path/to/code --max-file-size 20
```

### Exclude Patterns

```bash
# Exclude test files and specific directories
codereview /path/to/code --exclude "**/tests/**" --exclude "**/deprecated/**"
```

### Run Static Analysis

Combine AI review with static analysis tools (runs in parallel for speed):

```bash
# Run with all available static analysis tools
codereview /path/to/code --static-analysis

# Combine with specific model
codereview /path/to/code --model sonnet --static-analysis --output comprehensive-review.md
```

**Supported Static Analysis Tools:**
- **Python:** ruff (linter + format check), mypy (type checker), black (formatter), isort (import sorter), vulture (dead code finder), bandit (security scanner)
- **Go:** golangci-lint (meta-linter), go vet (static analyzer), gofmt (formatter), gosec (security)
- **Shell:** shellcheck, bashate
- **C++:** clang-tidy, cppcheck, clang-format
- **Java:** checkstyle
- **JavaScript/TypeScript:** eslint, prettier, tsc, npm-audit

**Output includes:**
- Tool pass/fail status
- Accurate issue counts (ruff/mypy/bandit parsed from summary lines, not substring guessing)
- Detailed output for failed checks
- Integrated into Markdown reports

**Notes:**
- Only installed tools are run; resolved via `shutil.which()` and rejected if they resolve inside the analyzed directory (supply-chain defense).
- Tools run in parallel via `ThreadPoolExecutor` (≤8 workers).
- File lists for tools that need explicit paths are sorted before truncating to `MAX_FILES_PER_TOOL=500`, so CI runs are reproducible.

**Override the per-tool subprocess timeout** (default 120s) for slow runs:
```bash
codereview /path/to/code --static-analysis --tool-timeout 600
```
Useful for `cppcheck --enable=all` on large C++ repos and `mypy` strict mode on big Python codebases.

### Scan Hidden Directories

By default, directories starting with `.` (`.git`, `.venv`, `.github`, `.config`, etc.) are skipped. Opt in to scan them:

```bash
codereview /path/to/code --include-hidden
```
Useful for reviewing CI scripts under `.github/scripts/` or config under `.config/`.

### Verbose Mode

```bash
# Show detailed progress and error traces
codereview /path/to/code --verbose
```

### Copy-Paste Friendly Output

Disable ANSI color/style codes for terminal output that's safe to copy-paste into other tools:

```bash
# No color mode - strips all ANSI escape codes
codereview /path/to/code --no-color

# Also respects the NO_COLOR environment variable (https://no-color.org/)
NO_COLOR=1 codereview /path/to/code
```

### All Options Combined

```bash
codereview /path/to/code \
  --model sonnet \
  --output report.md \
  --severity medium \
  --max-files 100 \
  --max-file-size 15 \
  --exclude "**/vendor/**" \
  --static-analysis \
  --verbose
```

## Review Categories

The tool identifies issues across 8 categories:

1. **Code Style**: Formatting, naming conventions, code organization
2. **Code Quality**: Complexity, duplication, maintainability
3. **Security**: Vulnerabilities, injection risks, data exposure
4. **Performance**: Inefficiencies, resource usage, optimization opportunities
5. **Best Practices**: Language idioms, design patterns, modern approaches
6. **System Design**: Architecture, modularity, scalability
7. **Testing**: Test coverage, test quality, missing tests
8. **Documentation**: Missing docs, unclear comments, API documentation

## Severity Levels

- **Critical**: Security vulnerabilities, data corruption risks, production blockers
- **High**: Major bugs, performance issues, important best practice violations
- **Medium**: Code quality issues, moderate technical debt, maintenance concerns
- **Low**: Minor improvements, style inconsistencies, nice-to-haves
- **Info**: Suggestions, alternative approaches, educational insights

## Output Format

### Terminal Output

The tool displays:
- File scanning progress
- Batch analysis progress
- Categorized issues with severity badges
- System design insights
- Priority recommendations
- Overall metrics summary

### Markdown Export

Generated reports include:
- Executive summary
- Metrics overview (files analyzed, total issues by severity)
- Detailed issue list with:
  - File paths and line numbers
  - Category and severity
  - Description and rationale
  - Suggested fixes (when applicable)
  - Reference links
- System design insights
- Top recommendations

### JSON Export

For CI/CD integration, use `--format json`:

```bash
codereview ./src --output report.json --format json
```

JSON output includes the full `CodeReviewReport` structure for programmatic consumption:
- Parse issues by severity for quality gates
- Integrate with dashboards and monitoring
- Automate notifications based on findings

## Troubleshooting

### Provider Credentials Not Found

```
Error: AWS credentials not found
Error: Azure OpenAI credentials not found
Error: Google API key not configured
```

**Solutions**:
- **AWS**: Configure credentials using `aws configure` or set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables
- **Azure**: Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` environment variables
- **Google**: Set `GOOGLE_API_KEY` environment variable (get from https://aistudio.google.com/apikey)

### Access Denied

```
Error: AccessDeniedException (AWS)
Error: 401 Unauthorized (Azure)
```

**Solutions**:
- **AWS**: Verify Bedrock access in AWS Console, check IAM permissions include `bedrock:InvokeModel`
- **Azure**: Verify API key is correct and resource is active in Azure Portal

### Model Not Available

```
Error: ResourceNotFoundException (AWS)
Error: DeploymentNotFound (Azure)
```

**Solutions**:
- **AWS**: Model may not be available in your region. Request access in AWS Bedrock Console
- **Azure**: Ensure you have deployed the model in your Azure OpenAI resource. Check deployment name matches configuration

### Rate Limiting

```
Error: ThrottlingException (AWS)
Error: 429 Too Many Requests (Azure)
```

**Solution**: The tool automatically retries with exponential backoff. If issues persist:
- Reduce batch size with `--batch-size 5` (fewer files per API call)
- Reduce total files with `--max-files`
- Use smaller file size limit (`--max-file-size`)
- Wait a few minutes before retrying
- Consider using a different model with higher rate limits

### No Files Found

```
Warning: No files found to review
```

**Reasons**:
- Directory is empty
- All files are excluded by default patterns
- File size limits are too restrictive

**Solution**: Check exclusion patterns and adjust `--max-file-size` if needed.

### Configuration File Not Found

```
Error: models.yaml not found
```

**Solution**: Ensure `codereview/config/models.yaml` exists. If using a custom configuration location, verify the path is correct.

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/lianghong/codereview-cli.git
cd codereview-cli

# Create virtual environment
uv venv --python 3.14

# Install in development mode with dependencies
uv pip install -e .
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_analyzer.py -v

# Run with coverage
uv run pytest tests/ --cov=codereview --cov-report=html
```

### Project Structure

```
codereview-cli/
├── codereview/
│   ├── __init__.py
│   ├── analyzer.py           # LLM-based code analysis
│   ├── batcher.py            # Smart file batching
│   ├── cli.py                # CLI entry point
│   ├── models.py             # Pydantic data models for review output
│   ├── renderer.py           # Terminal and Markdown rendering
│   ├── scanner.py            # File system scanning
│   ├── static_analysis.py    # Static analysis tool integration
│   ├── config/
│   │   ├── __init__.py       # Configuration exports
│   │   ├── models.yaml       # Provider and model configuration
│   │   ├── models.py         # Pydantic models for configuration
│   │   ├── prompts.py        # Code review rules and system prompt
│   │   └── loader.py         # YAML configuration loader
│   └── providers/
│       ├── __init__.py
│       ├── base.py           # ModelProvider abstract base class
│       ├── factory.py        # Provider factory with auto-detection
│       ├── bedrock.py        # AWS Bedrock provider implementation
│       ├── azure_openai.py   # Azure OpenAI provider implementation
│       ├── nvidia.py         # NVIDIA NIM provider implementation
│       └── google_genai.py   # Google GenAI provider implementation
├── tests/
│   ├── test_*.py             # Unit tests (319 tests)
│   └── fixtures/             # Test fixtures
├── docs/
│   ├── usage.md              # Detailed usage guide
│   ├── examples.md           # Example commands and workflows
│   └── static-analysis.md    # Static analysis tool reference
├── pyproject.toml            # Project configuration
├── LICENSE                   # MIT License
├── CLAUDE.md                 # Claude Code instructions
├── CHANGELOG.md              # Version history
└── README.md                 # This file
```

### Code Quality

The codebase follows strict quality standards:

**Code Standards:**
- Python 3.14+ modern syntax
- Type hints throughout
- Pydantic V2 for data validation
- Rich for terminal UI
- Click for CLI interface
- Comprehensive test coverage (319 tests)

**Static Analysis Tools:**
```bash
# Install development tools
uv pip install ruff mypy isort vulture types-PyYAML

# Run all checks
uv run ruff check codereview/ tests/
uv run ruff format --check codereview/ tests/
uv run mypy codereview/ --ignore-missing-imports
uv run isort --check-only codereview/ tests/
uv run vulture codereview/ --min-confidence 80

# Auto-fix formatting
uv run ruff format codereview/ tests/
uv run isort codereview/ tests/
uv run ruff check --fix codereview/ tests/
```

**Quality Requirements:**
- All code must pass: ruff (linting + formatting), mypy (type checking), isort (import sorting), vulture (dead code)
- All tests must pass (311/311)
- Type hints required for public APIs
- No unused imports or variables
- Provider implementations must include `get_pricing()` method
- Python 3.14 compliance: PEP 758 (unparenthesized exceptions) and PEP 765 (no control flow in finally)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. **Ensure code quality checks pass:**
   - Run `uv run pytest tests/ -v` (all tests must pass)
   - Run static analysis tools (ruff, mypy, isort, vulture)
   - See "Code Quality" section above for commands
5. Follow existing code style and architecture patterns
6. Update documentation if adding new features
7. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

For issues, questions, or contributions:
- [Open an issue](https://github.com/lianghong/codereview-cli/issues) on GitHub
- Check the [Usage Guide](docs/usage.md)
- Review [Examples](docs/examples.md)

## Version History

Current release: **v0.4.0** — Claude Opus 4.8 integration (new default, 1M context); model-registry audit (retired dead NVIDIA/Google endpoints with aliases redirected to live successors, fixed Opus 4.7 context/output, refreshed DeepSeek-V4-Pro pricing); added DeepSeek-V4-Flash and Step 3.7 Flash on NVIDIA; LangChain dependency hardening (version caps + pinned community packages); review-prompt improvements (linter-deference gating, Critical/High protected from issue cap, line-number guidance).

Full history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Anthropic Claude](https://www.anthropic.com/), [OpenAI GPT](https://openai.com/), [Google Gemini](https://ai.google.dev/), [xAI Grok](https://x.ai/), and [Mistral AI](https://mistral.ai/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service), [NVIDIA NIM](https://build.nvidia.com/), and [Google AI Studio](https://aistudio.google.com/) for model hosting
- Rich library for beautiful terminal output
- Static analysis tools: ruff, mypy, eslint, golangci-lint, shellcheck, bandit, gosec, and more
