# Code Review CLI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> AI-powered code review tool with multiple LLM providers (AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google Generative AI)

## 🎉 What's New (Unreleased)

- ✅ **DeepSeek-V4.1-Flash on NVIDIA NIM**: `--model dsv41-flash-nvidia`,
  1M context, reasoning, and prompt-based JSON output via NVIDIA's free endpoint.
- ✅ **Claude Opus 5.5 (Bedrock)** — newest Opus tier at $4/$20 per M (below Opus 5's $5.50/$27.50), 1M context, 128K output, always-on adaptive thinking (`--model opus5.5`). **It is now the default model**, replacing Opus 5, and the generation-neutral `opus`/`claude-opus` resolve to it; `opus5`/`claude-opus-5`/`opus-5` keep pointing at Opus 5.
- ⚠️ **Bedrock Claude cost estimates corrected (2026-09-23)** — the `us.` (Geo-US) inference-profile entries were priced at Anthropic's *global* rate, but Bedrock charges a 10% premium on regional/geo endpoints for every Claude model from 4.5 on. Opus 5 is now $5.50/$27.50 and Fable 5 $11/$55, both up 10%. Sonnet 5 goes *down*, to $2.20/$11: its $2/$10 launch price became the standard price when the planned rise to $3/$15 was cancelled. Opus 5.5 and Haiku 4.5 use `global.` profiles and are unchanged
- ✅ **GPT-6 Sol and GPT-6 Luna (Bedrock)** — the mid and low-cost GPT-6 tiers on `bedrock-mantle`, 1M context, 128K output, Responses API (`--model gpt6-sol`, `--model gpt6-luna`). Both are served from **us-east-1 only** (each entry's `region:` handles it), both are tiered above 272K input tokens like Astra, and `gpt6` still means Astra. **Their rates are derived, not published** — AWS had no Sol/Luna price on 2026-09-23, so the entries carry OpenAI's list price ×1.1 (the In-Region premium Astra's published rate shows); check your bill before trusting the estimate
- ✅ **Gemini 3.8 Flash (Google)** — newest Flash generation for long-horizon software engineering and autonomous agents, with 1M context and 64K output (`--model gemini-3.8-flash`, or `gemini-flash`). It starts on prompt-based structured output until a live review proves forced tool use while thinking. Gemini 3.7 Flash was removed on 2026-09-19 as curation, and 3.8 absorbed its generation-neutral names — see the migration table
- ✅ **3 new providers**: DeepSeek direct API (`deepseek-v4-pro`, `deepseek-v4-flash`), Z.AI (`zhipuai/glm-5.3`, `zhipuai/glm-5.3-flash`), Moonshot/Kimi (`kimi-k3`). 8 providers total now (incl. OpenAI-on-Bedrock).
- ✅ **GPT-6 Astra (Bedrock)** — OpenAI's most capable model, GA on the OpenAI-compatible `bedrock-mantle` endpoint 2026-09-08, text + image input, 128K output (`--model gpt6`). Two caveats worth knowing before you switch: it is the first entry with **tiered** pricing — Bedrock bills Astra at $11/$55 per M up to 272K input tokens and **$22/$82.50 above it**, per request, so a batch that crosses the break costs double and `--dry-run` names how many do; and `bedrock-mantle` serves it from **us-west-2 only**, mutually exclusive with GPT-5.6 Sol's us-east-1/us-east-2 — each entry declares its `region:` and the provider rewrites the Region label of your `OPENAI_BASE_URL` per model, so one export reaches both
- ✅ **GPT-5.6 Sol (Bedrock)** — OpenAI's flagship and best coding model, on the OpenAI-compatible `bedrock-mantle` endpoint, 272K context, Responses API (`--model gpt5.6`). It inherits `gpt-bedrock`
- ✅ **GLM-5.3 and GLM-5.3-Flash (Z.AI)** — latest flagship and low-cost multimodal sibling, both with 1M context. `glm` now selects 5.3; Flash costs $0.15/$0.50 per M at list price. Both start on prompt-based structured output because reasoning is always enabled
- ✅ **GPT-5.4 (Azure)** — frontier reasoning model, 1.05M context, default Azure model
- ➖ **Grok 4.3 (Bedrock) was added *and* removed inside this same unreleased cycle**, as was GPT-5.5-on-Bedrock. Both endpoints are live; both entries were cut in the curation pass below. `bedrock_openai` is still not OpenAI-only — Grok needed no provider code, only a YAML entry — so re-adding is a config change. See [Migrating deleted aliases](#migrating-deleted-aliases) for `grok`/`grok-4.3`/`grok43`/`grok-bedrock`, all of which now fail fast.
- ✅ **Registry cleanup (11 entries removed)** — every model probed against its live provider endpoint; superseded, region-unavailable, and dead entries dropped. 30 models remained, 32 with Gemini 3.7 Flash and Kimi K3.
- ⚠️ **Second cleanup pass, 2026-08-29 (5 more entries removed)** — re-probing every entry found **half the NVIDIA NIM roster dead**: `mistralai/mistral-small-4-119b-2603` (EOL 2026-07-27), `qwen/qwen3.5-397b-a17b` (2026-07-27), `mistralai/mistral-medium-3.5-128b` (2026-08-07), `z-ai/glm-5.2` (2026-08-21) and `stepfun-ai/step-3.7-flash` (2026-08-28) all answer **HTTP 410 Gone** with NVIDIA's own end-of-life date. NIM now serves no Mistral, Qwen, GLM or StepFun model at all. **27 models remained.** These failed at invocation time only — the catalog no longer lists them, so neither `--list-models` nor `--validate` (which checks catalog visibility) could have caught it.
- ➖ **Curation pass, 2026-08-29 (9 more entries removed, 27 → 18)** — this one is **not** a dead-endpoint cleanup: **all nine endpoints are live and were probed to confirm it.** Removed: Claude Opus 4.8, Claude Sonnet 4.6, Kimi K2.5 (Bedrock), Qwen3-Coder-Next (Bedrock), MiniMax M2.5 (Bedrock), Kimi K2.6 (NVIDIA), Gemini 3.6 Flash, GPT-5.5 (Bedrock) and Grok 4.3 (Bedrock). What it costs, stated plainly so re-adding is an informed choice: Bedrock's cheapest entry goes from **$0.50/M → $1.00/M** (`haiku`), Bedrock keeps exactly **one** entry on the tool-use structured-output path (`haiku`), and `bedrock_openai`'s cheapest goes from **$1.25/M → $5.00/M** with a narrower window (400K → 272K). Every removal site in `models.yaml` carries a dated comment saying what the probe showed and what was lost.
- ⚠️ **Alias cleanup (94 aliases deleted)** — `--list-models` now advertises only current names. Version-explicit aliases of removed models (`opus4.6`, `opus4.8`, `glm51`, `mm25`, `mm2.5-bedrock`, `kimi25`, `step35`, `gpt5.4-bedrock`, `gpt5.5-bedrock`, `gemini-3.6-flash`, `grok-4.3`, `mistral-small`, `qwen3.5`, `glm52`, `step-3.7-flash`, …) were **deleted**, not redirected: they now fail fast instead of silently resolving to a newer model with different pricing and behavior. Version-neutral names (`glm5`, `kimi-azure`, `gemini-3-flash`, `gpt-bedrock`, …) still resolve and are listed under `--list-models --verbose`; `sonnet`/`claude-sonnet` moved onto Sonnet 5 as fully advertised aliases. Names were deleted rather than migrated where the whole family left the registry (`qwen*`, `grok*`, `step*`) or where a migration would have crossed a free/billed provider boundary. See [Migrating deleted aliases](#migrating-deleted-aliases).
- ✅ **`--fail-on <severity>`** — CI merge gate: exits 2 when issues at that severity or above are found, distinct from exit 1 (the run itself failed). Independent of `--severity`, and applied after the report is written
- ✅ **New `Correctness` category** — logic errors, edge cases, error paths, race conditions, and resource leaks are no longer filed as "Code Quality" next to naming nits
- ✅ **`--tool-timeout`** — override the static-analysis subprocess timeout (default 120s) for slow C++/mypy runs
- 🔒 **`--trust-repo-config`** — static analysis no longer runs mypy/ESLint/Prettier when the *reviewed* repository ships a config that makes them load code from the tree (a mypy `plugins =` entry, a JavaScript `eslint.config.*`). Those tools are skipped with a visible reason; pass the flag to opt back in for a repository you trust
- ✅ **`--include-hidden`** — opt-in scanning of `.github/scripts`, `.config/`, etc.
- ✅ **Reproducible static analysis** — file lists sorted before truncation so CI runs are deterministic (locked in by regression test)
- ✅ **Accurate issue counts** — ruff/mypy/bandit summary-line parsing replaces the old substring-match heuristic
- ✅ **Supply-chain hardening** — static-analysis tools resolved via `shutil.which()`; binaries inside the analyzed directory are refused (gofmt cache-bypass also fixed)
- ✅ **AWS error redaction** — STS/Bedrock validation errors no longer leak SCP fragments or IAM policy details
- ✅ **All 1208 tests passing**, ruff/format/mypy clean

A LangChain-based CLI tool that provides comprehensive, intelligent code reviews for Python, Go, Shell Script, C++, Java, JavaScript, and TypeScript projects using Claude, GPT-5.4, GPT-5.6 Sol, GPT-6 Astra, GPT-6 Sol, GPT-6 Luna, Gemini, DeepSeek-V4-Pro, Kimi K3, GLM-5.3, and other leading models through AWS Bedrock, Azure OpenAI, NVIDIA NIM, Google Generative AI, DeepSeek, Z.AI, and Moonshot.

## Features

- **Multi-Provider Support** (8 providers): AWS Bedrock (Claude, Kimi K3), Azure OpenAI (GPT-5.4, GPT-5.4 Pro), NVIDIA NIM (DeepSeek-V4.1-Flash, GLM-5.3, GLM-5.3-Flash, Kimi K3), Google GenAI (Gemini 3.1 Pro / 3.8 Flash), DeepSeek direct (V4-Pro, V4-Flash), Z.AI (GLM-5.3, GLM-5.3-Flash), Moonshot direct (Kimi K3), and OpenAI-on-Bedrock (GPT-5.6 Sol, GPT-6 Astra via the `bedrock-mantle` OpenAI-compatible endpoint)
- **AI-Powered Analysis**: Leverages Claude Opus 5.5, Claude Opus 5, Claude Sonnet 5, GPT-5.4, GPT-5.6 Sol, GPT-6 Astra, GPT-6 Sol, GPT-6 Luna, DeepSeek-V4-Pro, Kimi K3, GLM-5.3, Gemini 3.1 Pro, Gemini 3.8 Flash, and other leading models for deep code understanding
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
  - AWS account with Bedrock access (Claude Opus 5 / Sonnet 5 / Haiku 4.5 / Fable 5, GLM 5)
  - Azure OpenAI resource with model deployment (GPT-5.4, GPT-5.4 Pro) — `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
  - NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) — `NVIDIA_API_KEY` (DeepSeek-V4.1-Flash, GLM-5.3, GLM-5.3-Flash, Kimi K3; free tier available)
  - Google API key from [AI Studio](https://aistudio.google.com/apikey) — `GOOGLE_API_KEY` (Gemini 3.1 Pro / 3.8 Flash)
  - DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com/api_keys) — `DEEPSEEK_API_KEY` (V4-Pro, V4-Flash)
  - Z.AI API key from [z.ai](https://z.ai) — `ZAI_API_KEY` (GLM-5.3 / 5.3-Flash; international)
  - Moonshot/Kimi API key from [platform.moonshot.cn](https://platform.moonshot.cn) — `KIMI_API_KEY` (Kimi K3; international keys from `platform.moonshot.ai` work too — override `base_url`)
  - Amazon Bedrock API key (bearer token) for OpenAI-on-Bedrock — `OPENAI_API_KEY` + `OPENAI_BASE_URL` (GPT-5.6 Sol, GPT-6 Astra, GPT-6 Sol, GPT-6 Luna via the `bedrock-mantle` OpenAI-compatible endpoint — these models' Regions do not all overlap, but each entry's `region:` resolves that, so one base URL serves them all)

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
3. Request access to "Anthropic Claude Opus 5.5" (the default model)
4. Wait for approval (usually instant for supported regions)

Kimi K3 (`--model kimi-bedrock`) is served only through cross-Region inference profiles — the
entry uses the Global profile `global.moonshotai.kimi-k3` ($3/$15 per M, same as Moonshot
direct). For US data residency, switch its `full_id` to `us.moonshotai.kimi-k3` in
`models.yaml` and its pricing to $3.30/$16.50.

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

Azure OpenAI provides access to GPT-5.4 and GPT-5.4 Pro via Microsoft Azure AI Foundry.

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
3. Note your deployment name, endpoint, and API key

### 3. Use Azure Models

```bash
# GPT-5.4 - Flagship reasoning model, 1.05M context (default Azure model)
codereview /path/to/code --model gpt

# GPT-5.4 Pro - Deeper reasoning variant, 1.05M context
codereview /path/to/code --model gpt-pro
```

### 4. Test Connection

```bash
codereview --list-models  # Should show Azure models
```

**Note:** Azure OpenAI models require you to deploy them in your Azure resource first — unlike the Bedrock/NVIDIA catalogs, an entry only works if a deployment with that exact name exists on your resource. The Kimi K2.5 and DeepSeek-V4-Pro Azure entries were removed in the 2026-07-25 registry cleanup for that reason (`DeploymentNotFound`); their aliases now route to the direct Moonshot and DeepSeek APIs. If you deploy a Foundry model that doesn't support tool calling, add it back with `supports_tool_use: false` in `models.yaml` and the provider falls back to prompt-based JSON parsing automatically.

## NVIDIA NIM Configuration (Alternative Provider)

NVIDIA NIM provides access to [DeepSeek-V4.1-Flash](https://build.nvidia.com/deepseek-ai/deepseek-v4.1-flash), GLM-5.3, GLM-5.3-Flash and Kimi K3 — with a free tier for development.

**NIM endpoints can be retired, and catalog visibility does not prove invocability.** A retired endpoint disappears from the catalog and answers `HTTP 410 Gone` with its end-of-life date. `--list-models` reads local configuration, while `--validate` checks catalog visibility and treats a miss as a warning. The previous DeepSeek V4 endpoints were retired in August–September 2026; V4-Flash-0731 advertised its September 21 sunset in a `deprecation` response header while still returning HTTP 200. V4.1-Flash is a separate endpoint with new aliases. If a NIM model starts failing every batch, check `https://integrate.api.nvidia.com/v1/models` and the completion response headers.

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
# DeepSeek-V4.1-Flash - 552B MoE, 1M context, reasoning enabled by default.
# Full CLI id: deepseek-v4.1-flash-nvidia; also dsv4.1-flash-nvidia.
codereview /path/to/code --model dsv41-flash-nvidia

# GLM-5.3-Flash - the fast, free NIM default: 320B/18B MoE, 1M context, native
# multimodal, always-on reasoning. Bare `glm-flash` stays reserved for Z.AI direct.
codereview /path/to/code --model glm53-flash-nvidia

# GLM-5.3 - Z.ai's 753B/40B text flagship, 1M context, leads CyberGym for
# vulnerability discovery. WARNING: served NVFP4 on GB300 and very slow (a probe
# took ~5 min to return 8 tokens) - prefer the Flash entry above for CI.
codereview /path/to/code --model glm53-nvidia

# Kimi K3 - Moonshot flagship: 2.8T/104B MoE, 1M context, native multimodal,
# always-on thinking. Bare `kimi-k3` stays reserved for the Moonshot direct API.
# (the Kimi K2.6 NIM entry was removed 2026-08-29 - the endpoint is listed but
#  not provisioned for every account; use `--model kimi` for K3 on Moonshot)
codereview /path/to/code --model kimi-nvidia-3
```

**Note:** NVIDIA NIM models are currently in free tier. All four entries use
prompt-based JSON parsing with reasoning enabled. DeepSeek-V4.1-Flash keeps
NVIDIA's sampling and reasoning defaults and requests the example's 262,144-token
output budget within its 1,048,576-token combined context. The other NIM entries
pin `reasoning_effort: high`; reasoning consumes output tokens that the review
report also needs. **`glm53-nvidia` is by far the slowest entry in the registry**
(NVFP4 on GB300; a probe took ~5 minutes to return 8 tokens, and a real review of
a single 7-line file took **19 minutes**, versus 3 minutes for Flash); use
`glm53-flash-nvidia` for CI and high-volume runs.

The retired `dsv4-flash-nvidia` / `dsv4-nvidia` aliases still fail explicitly.
Use `dsv41-flash-nvidia` for the new NVIDIA release, or `dsv4-flash` /
`deepseek-v4-pro` for the billed V4 models on DeepSeek's direct API.

For GLM, Kimi, Qwen, Mistral, MiniMax and DeepSeek-V4-**Pro**, the surviving routes are on other providers: GLM-5.3 and GLM-5.3-Flash via Z.AI direct (`--model glm` / `glm-flash`, which also own `glm5`/`glm-5` since the GLM 5 Bedrock re-host was curated away 2026-09-19), Kimi K3 via Moonshot direct (`--model kimi`), DeepSeek-V4-Pro via DeepSeek direct (`--model deepseek-v4-pro`, billed at $1.32/$3.96). Nothing in this registry replaces the retired Mistral or StepFun endpoints; no Qwen model remains anywhere in it (the Bedrock entry was removed 2026-08-29); and no MiniMax remains either, since NIM end-of-lifed M3 on 2026-09-09 after the Bedrock re-host had already been curated away. Every `qwen*`, `mistral*`, `step*` and `minimax*` spelling now fails fast.

## Google Generative AI Configuration (Alternative Provider)

Google Generative AI provides access to Gemini 3.1 Pro and Gemini 3.8 Flash, both with 1M token context windows.

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

# Gemini 3.8 Flash - latest Flash generation for long-horizon engineering (1M context)
# (owns gemini-flash, plus the removed 3.7 entry's gemini-3-flash,
#  gemini3-flash and g3flash names)
codereview /path/to/code --model gemini-3.8-flash
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

Z.AI is Zhipu's international platform. This registry exposes GLM-5.3 (the current text flagship) and GLM-5.3-Flash (the low-cost native-multimodal sibling), both with 1M-token contexts, through the OpenAI-compatible endpoint.

### 1. Get API Key

Sign up at [z.ai](https://z.ai) and create an API key.

### 2. Set Environment Variable

```bash
export ZAI_API_KEY="your-zai-key"
```

### 3. Use Z.AI Models

```bash
# GLM-5.3 - current flagship, 1M-token context (`glm` tracks this model)
codereview /path/to/code --model zhipuai/glm-5.3
codereview /path/to/code --model glm

# GLM-5.3-Flash - low-cost multimodal sibling, 1M-token context
codereview /path/to/code --model glm-flash
```

## Moonshot AI (Kimi) Configuration (Alternative Provider)

Moonshot's direct API exposes Kimi K3 (2.8T MoE, 104B active, 1M context, always-on thinking, agentic-coding optimized) via the dedicated `langchain-moonshot` package.

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
# Kimi K3 - 2.8T MoE, 1M context
codereview /path/to/code --model kimi-k3
codereview /path/to/code --model kimi  # short alias (canonical)
```

If your key is from the international platform (`platform.moonshot.ai`), override the endpoint in `codereview/config/models.yaml`:
```yaml
moonshot:
  base_url: "https://api.moonshot.ai/v1"
```

## OpenAI-on-Bedrock Configuration (Alternative Provider)

Amazon Bedrock hosts OpenAI's frontier models on an OpenAI-compatible `bedrock-mantle` endpoint. This is a **different path** from the SigV4 `bedrock` provider above: it authenticates with an **Amazon Bedrock API key (a bearer token, not AWS creds)** and is driven with `ChatOpenAI` + a custom `base_url` — no new dependency.

The endpoint is not OpenAI-only — it also serves xAI's Grok, and this provider needs no code change to drive it (the Grok 4.3 entry that proved this was removed in the 2026-08-29 curation pass, not because anything broke). Adding a non-OpenAI `bedrock-mantle` model is a `models.yaml` entry.

### 1. Get an Amazon Bedrock API Key

In the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/home#/api-keys/long-term/create), generate a long-term API key. This is **not** an openai.com key.

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="<your-amazon-bedrock-api-key>"
# Any bedrock-mantle Region works — each model entry declares the Region that
# actually serves it (`region:` in models.yaml) and the provider rewrites the
# Region label of this URL per model. GPT-5.6 Sol is In-Region us-east-1 /
# us-east-2 only, GPT-6 Astra is us-west-2 only, GPT-6 Sol/Luna are us-east-1
# only, and one export now reaches all of them; scheme, host and path still come from here.
export OPENAI_BASE_URL="https://bedrock-mantle.us-east-1.api.aws/openai/v1"
```

### 3. Use OpenAI-on-Bedrock Models

```bash
# GPT-5.6 Sol - OpenAI flagship, best coding model, 272K context (Responses API)
# (the GPT-5.5 and Grok 4.3 entries were removed 2026-08-29; `gpt-bedrock`
#  resolves here, at a higher rate than GPT-5.5 — see the migration table)
codereview /path/to/code --model gpt5.6
codereview /path/to/code --model gpt5.6-bedrock   # or gpt-5.6, gpt5.6-sol-bedrock

# GPT-6 Astra - OpenAI's most capable model (Responses API, text + image in)
# Runs in us-west-2 — the ONLY bedrock-mantle Region for it, and not the one
# Sol runs in; the entry's `region:` handles that, so no re-export is needed.
# Pricing is tiered: a batch over 272K input tokens bills at $22/$82.50 per M
# instead of $11/$55, and --dry-run says which batches those are.
codereview /path/to/code --model gpt6
codereview /path/to/code --model gpt6-bedrock     # or gpt-6, gpt6-astra-bedrock

# GPT-6 Sol / GPT-6 Luna - the mid and low-cost GPT-6 tiers (Responses API)
# us-east-1 only — a third Region, again handled by the entry's `region:`.
# Pricing is DERIVED (OpenAI list x1.1, the In-Region premium Astra carries);
# AWS hadn't published a rate for either on 2026-09-23.
codereview /path/to/code --model gpt6-sol         # or gpt-6-sol, gpt6-sol-bedrock
codereview /path/to/code --model gpt6-luna        # or gpt-6-luna, gpt6-luna-bedrock
```

> **Tip:** GPT-5.6 also ships cheaper tiers — Terra (`openai.gpt-5.6-terra`, $2.50/$15) for balanced everyday work and Luna (`openai.gpt-5.6-luna`, $1/$6) for high-volume/CI. Add either the same way as the Sol entry in `models.yaml` if you want a lower-cost run.

## Usage

### Basic Usage

```bash
# Uses Claude Opus 5.5 by default
codereview /path/to/your/codebase
```

### Choose Your Model

```bash
# List all available models (advertised aliases only)
codereview --list-models

# Also show deprecated aliases — back-compat names that resolve to a successor model
codereview --list-models --verbose

# AWS Bedrock Models (Claude family)
codereview /path/to/code --model fable5    # Claude Fable 5 (Mythos-class, 1M context)
codereview /path/to/code --model opus      # Claude Opus 5.5 (the default model since 2026-09-23; newest Opus, 1M context)
codereview /path/to/code --model opus5     # Claude Opus 5 (the previous default, 1M context)
codereview /path/to/code --model sonnet    # Claude Sonnet 5 (1M context; `sonnet` moved here 2026-08-29)
codereview /path/to/code --model haiku     # Claude Haiku 4.5 (fastest, cheapest Bedrock entry)

# Azure OpenAI Models
codereview /path/to/code --model gpt              # GPT-5.4 (1.05M context, frontier reasoning)
codereview /path/to/code --model gpt-pro          # GPT-5.4 Pro (deeper reasoning variant)

# NVIDIA NIM Models (free tier)
codereview /path/to/code --model glm53-flash-nvidia # GLM-5.3-Flash on NVIDIA (free, 1M context, fast)
codereview /path/to/code --model glm53-nvidia       # GLM-5.3 on NVIDIA (free; 753B MoE - but very slow)
codereview /path/to/code --model kimi-nvidia-3      # Kimi K3 on NVIDIA (free; 2.8T MoE, 1M context)

# Google Generative AI Models
codereview /path/to/code --model gemini-3.1-pro     # Gemini 3.1 Pro (1M context)
codereview /path/to/code --model gemini-3.8-flash   # Gemini 3.8 Flash (latest Flash; `gemini-flash`)

# DeepSeek Direct API
codereview /path/to/code --model deepseek-v4-pro    # Flagship, 1M context
codereview /path/to/code --model deepseek-v4-flash  # 12x cheaper input, 1M context

# Z.AI (Zhipu international)
codereview /path/to/code --model zhipuai/glm-5.3          # Flagship, 1M-token context
codereview /path/to/code --model zhipuai/glm-5.3-flash    # Low-cost multimodal, 1M context
codereview /path/to/code --model glm                      # Flagship short alias

# Moonshot direct API (Kimi)
codereview /path/to/code --model kimi-k3            # Canonical, 1M context, 2.8T MoE
codereview /path/to/code --model kimi               # Short alias

# OpenAI-on-Bedrock (bedrock-mantle OpenAI-compatible endpoint; bearer-key auth)
codereview /path/to/code --model gpt5.6             # GPT-5.6 Sol (OpenAI flagship, best coding model)
codereview /path/to/code --model gpt6               # GPT-6 Astra (OpenAI's most capable; us-west-2 only)
codereview /path/to/code --model gpt6-sol           # GPT-6 Sol (mid tier; us-east-1 only)
codereview /path/to/code --model gpt6-luna          # GPT-6 Luna (cheapest GPT-6; us-east-1 only)

# Short aliases work too
codereview /path/to/code -m haiku
codereview /path/to/code -m gpt
codereview /path/to/code -m kimi
```

**Model Comparison:**

| Model | Provider | Use Case | Input $/M | Output $/M |
|-------|----------|----------|-----------|------------|
| Fable 5 | AWS Bedrock | Mythos-class, always-on thinking, 1M context | $11.00 | $55.00 |
| Opus 5.5 | AWS Bedrock | **Default model.** Newest Opus, always-on adaptive thinking, 1M context; Global cross-Region profile (owns `opus`) | $4.00 | $20.00 |
| Opus 5 | AWS Bedrock | Previous default, best for code review, 1M context | $5.50 | $27.50 |
| Sonnet 5 | AWS Bedrock | Claude 5 gen, near-Opus at Sonnet price, 1M context (owns `sonnet`) | $2.20 | $11.00 |
| Haiku 4.5 | AWS Bedrock | Fast, economical, large codebases — cheapest Bedrock entry, and the only one on the native tool-use path | $1.00 | $5.00 |
| Kimi K3 (Bedrock) | AWS Bedrock | 2.8T MoE / 104B active, 1M context, always-on thinking; Global cross-Region profile, AWS credentials instead of a Moonshot key (owns `kimi-bedrock`) | $3.00 | $15.00 |
| GPT-5.4 | Azure OpenAI | Frontier reasoning, 1.05M context, default Azure | $2.50 | $15.00 |
| GPT-5.4 Pro | Azure OpenAI | Deeper reasoning, hardest problems | $30.00 | $180.00 |
| GLM-5.3-Flash (NVIDIA) | NVIDIA NIM | 320B MoE / 18B active, 1M context, multimodal input, always-on reasoning; the fast free NIM default | Free* | Free* |
| DeepSeek-V4.1-Flash (NVIDIA) | NVIDIA NIM | 552B MoE, 1M context, reasoning; prompt-based JSON output | Free* | Free* |
| GLM-5.3 (NVIDIA) | NVIDIA NIM | 753B MoE / 40B active, 1M context, leads CyberGym for vulnerability discovery; **very slow** (NVFP4 on GB300) | Free* | Free* |
| Kimi K3 (NVIDIA) | NVIDIA NIM | 2.8T MoE / 104B active, 1M context, multimodal input, always-on thinking | Free* | Free* |
| Gemini 3.1 Pro | Google GenAI | Most advanced reasoning, 1M context (supersedes retired 3 Pro) | $2.00 | $12.00 |
| Gemini 3.8 Flash | Google GenAI | Latest Flash: long-horizon software engineering and autonomous agents, 1M context, 64K output (owns `gemini-flash` and the generation-3 names) | $1.50 | $7.50 |
| **DeepSeek-V4-Pro** | **DeepSeek direct** | **1M context, three reasoning modes, tool calling; peak rate shown** | **$1.32** | **$3.96** |
| **DeepSeek-V4-Flash** | **DeepSeek direct** | **1M context, lower-cost sibling; peak rate shown** | **$0.44** | **$1.32** |
| **GLM-5.3 (Z.AI)** | **Z.AI direct** | **Latest text flagship, always-on reasoning, 1M context** | **$1.40** | **$4.40** |
| **GLM-5.3-Flash (Z.AI)** | **Z.AI direct** | **Low-cost multimodal sibling, 1M context (list rate)** | **$0.15** | **$0.50** |
| **Kimi K3** | **Moonshot direct** | **2.8T MoE, 104B active, 1M context, always-on thinking, agentic (owns `kimi`)** | **$3.00** | **$15.00** |
| **GPT-5.6 Sol (Bedrock)** | **OpenAI-on-Bedrock** | **OpenAI flagship, best coding model, 272K context, `bedrock-mantle` endpoint us-east-1/2 (owns `gpt-bedrock`)** | **$4.40** | **$22.00** |
| **GPT-6 Astra (Bedrock)** | **OpenAI-on-Bedrock** | **OpenAI's most capable model, text + image in, 128K output, `bedrock-mantle` endpoint us-west-2 only. 1M context; tiered pricing — a request over 272K input tokens bills $22/$82.50** | **$11.00** | **$55.00** |
| **GPT-6 Sol (Bedrock)** | **OpenAI-on-Bedrock** | **Mid GPT-6 tier, 1M context, 128K output, `bedrock-mantle` us-east-1 only; tiered above 272K ($4.40/$16.50). Derived rate — AWS unpublished** | **$2.20** | **$11.00** |
| **GPT-6 Luna (Bedrock)** | **OpenAI-on-Bedrock** | **Cheapest GPT-6 tier, 1M context, 128K output, `bedrock-mantle` us-east-1 only; tiered above 272K ($0.22/$0.825). Derived rate — AWS unpublished** | **$0.11** | **$0.55** |

*NVIDIA NIM models are currently in free preview tier. Models with thinking mode use interleaved reasoning for deeper code analysis. Several Bedrock models display "TBD" until AWS publishes official pricing — the CLI renders unpriced models as `Estimated cost: TBD` instead of `$0.0000`.

Nine rows left this table on 2026-08-29 (Opus 4.8, Sonnet 4.6, Kimi K2.5-on-Bedrock, Qwen3 Coder Next, MiniMax M2.5-on-Bedrock, Kimi K2.6-on-NVIDIA, Gemini 3.6 Flash, GPT-5.5-on-Bedrock, Grok 4.3-on-Bedrock) — **all still live upstream**, removed as curation. The two that changed a floor: Qwen3 Coder Next was the cheapest Bedrock entry at $0.50/$1.20, and Grok 4.3 the cheapest `bedrock-mantle` one at $1.25/$2.50. Two more rows left on 2026-09-19 for the opposite reason — NVIDIA end-of-lifed MiniMax M3 (2026-09-09) and DeepSeek-V4-Pro-0813 (2026-09-14), both HTTP 410, so the NIM roster is down to three entries. Three more left the same day as **curation**, all three still live upstream: GLM 5 (Bedrock), Gemini 3.7 Flash, and Kimi K2.6 (Moonshot) — K2.6 was replaced in place by **Kimi K3**, which is a 5x price rise ($0.60/$2.50 → $3.00/$15.00, and the old figure was itself wrong; K2.6 really billed $0.95/$4.00), so pin `kimi-k2.6` back from git history if you were budgeting on it. Dropping 3.7 Flash also cost the registry its **only** live proof that a thinking model can keep the native tool-use path. `--list-models` is always authoritative over this table.

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

### Fail the Build on Findings (CI Gate)

`--fail-on` turns the review into a merge gate. It exits **2** when any issue at
the given severity *or above* was found:

```bash
# Fail CI if any Critical or High issue exists
codereview ./src --fail-on high

# Strictest: fail on anything at all
codereview ./src --fail-on info
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Review completed; nothing at or above the `--fail-on` threshold |
| `1` | The **run** failed (no results, bad credentials, API error, unwritable output) |
| `2` | The review **succeeded** and found blocking issues (`--fail-on` tripped) |

`1` and `2` are deliberately distinct: a broken pipeline and a failing code
review need different responses.

Two things to know:

- `--fail-on` is independent of `--severity`. `--severity` only filters what is
  *displayed*; hiding a finding never changes the exit code. `--severity critical
  --fail-on high` still fails on a High issue you never saw.
- The gate is applied **after** the report is written, so a failing build still
  produces its `--output` artifact.

```yaml
# GitHub Actions
- name: AI code review
  run: |
    uv run codereview ./src \
      --fail-on high \
      --output review.json --format json \
      --quiet
- name: Upload review
  if: always()          # the report exists even when the gate fails
  uses: actions/upload-artifact@v4
  with:
    path: review.json
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

### Repo-Supplied Configs That Execute Code

Reviewing an untrusted repository means running linters against files you did not
write — and three of them load *code* from the tree when the repository asks them
to:

| Tool | Config | What it executes |
|---|---|---|
| mypy | `mypy.ini`, `setup.cfg`, `pyproject.toml` with a `plugins =` entry | imports the named Python module |
| ESLint | `eslint.config.{js,mjs,cjs,ts}`, `.eslintrc.{js,cjs,mjs}` | the config *is* JavaScript |
| ESLint / Prettier | `.eslintrc.json`, `.prettierrc`, `package.json` with a `plugins` key | loads the named plugin module |

That code runs with your privileges, in your shell, before any review output
exists. So by default the tool **detects those configs and skips the tool**,
naming what it skipped and why:

```
✗ Skipped mypy: mypy.ini in the analyzed repository would make it load and
  execute code from the tree with your privileges. Its findings are missing
  from this review. Pass --trust-repo-config to run it anyway (only for a
  repository you trust).
```

Detection is on **content, not presence**: an ordinary `pyproject.toml` with a
`[tool.mypy]` section but no `plugins` entry still runs mypy normally — repo
config is what makes linter output match that project's CI, and this project's
own `pyproject.toml` is not flagged. An unreadable or oversized (>512 KB) config
*is* treated as risky, since that is what an attacker would arrange if it
bypassed the check.

Opt back in for a repository you trust:

```bash
codereview /path/to/code --static-analysis --trust-repo-config
```

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

The tool identifies issues across 9 categories:

1. **Correctness**: Logic errors, off-by-one, unhandled edge cases, missing error paths, race conditions, resource leaks — the code returns a wrong result or crashes
2. **Code Style**: Formatting, naming conventions, code organization
3. **Code Quality**: Complexity, duplication, maintainability — the code works but is hard to maintain
4. **Security**: Vulnerabilities, injection risks, data exposure
5. **Performance**: Inefficiencies, resource usage, optimization opportunities
6. **Best Practices**: Language idioms, design patterns, modern approaches
7. **System Design**: Architecture, modularity, scalability
8. **Testing**: Test coverage, test quality, missing tests
9. **Documentation**: Missing docs, unclear comments, API documentation

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

### OpenAI-on-Bedrock: "The model … does not exist"

```
✗ Error on batch 1: OpenAIModelNotFoundError: Error code: 404 - {'error': {'code':
  'not_found_error', 'message': "The model 'openai.gpt-6-astra' does not exist", ...}}
```

Every batch fails with the same 404 and the run ends with `All N batch(es) failed`. The model
id is real — it just doesn't exist *in the Region your endpoint names*, and `bedrock-mantle`
returns 404 rather than redirecting. The entries on this provider have **no Region common to
all of them**: GPT-6 Astra is us-west-2 only, GPT-6 Sol and Luna are us-east-1 only, GPT-5.6 Sol
is In-Region us-east-1 / us-east-2.

`OPENAI_BASE_URL` is provider-wide, so the Region comes from **the model entry**: each carries
`region:` in `models.yaml`, and the provider rewrites the Region label of your configured URL
before building the client. One export reaches both models. If you hit this 404:

- **A model entry is missing `region:`.** Add it — that's the fix, not a re-export.
- **Your `OPENAI_BASE_URL` host has no Region label to rewrite** (a custom gateway, say). The
  provider then uses the URL exactly as configured and logs why at debug level; point it at a
  `bedrock-mantle.<region>.api.aws` host, or set the Region you need directly.
- **The Region in the entry is wrong.** Check the model card's `bedrock-mantle` availability
  table rather than widening it from memory.

Note that `--validate` will *not* catch any of this: it checks key presence and HTTPS only,
and runs no connection test for OpenAI-on-Bedrock. The Region is exercised on first invoke.

### Migrating Deleted Aliases

```
Error: 'opus4.6' is not a valid model
```

The 2026-07-25 and 2026-08-29 alias cleanups **deleted** the version-explicit aliases of
removed models instead of pointing them at a successor. A name that says "4.6" silently
resolving to a two-generations-newer model — with different pricing, different
sampling-parameter support and a different structured-output path — is worse than an error
you can read and fix. **Whether the endpoint is still alive upstream doesn't change this**:
nine of the entries below were removed as curation and their endpoints still answer, but their
version-pinning names are deleted on exactly the same rule.

| Deleted alias(es) | Use instead |
|---|---|
| `opus4.7`, `opus-4.7`, `claude-opus-4.7`, `claude-opus-47`, `opus4.6`, `opus-4.6`, `claude-opus-4.6`, `opus4.8`, `opus-4.8`, `claude-opus-4.8`, `claude-opus-48` | `opus5` — the same $5/$25 list price (billed $5.50/$27.50 on its `us.` profile), context and output as 4.8; `opus` now means Opus 5.5 |
| `sonnet4.6`, `claude-sonnet-4.6` | `sonnet5` — `sonnet`/`claude-sonnet` now resolve there too. Cheaper than 4.6's $3/$15 (Sonnet 5's $2/$10 list, $2.20/$11 on the `us.` profile) with 5x the context; note 4.6 accepted `--temperature` and used the tool-use path, and Sonnet 5 does neither |
| `minimax-m2.7`, `minimax-m2.7-nvidia`, `mm2.7-nvidia`, `mm27`, `minimax-m2.5-nvidia`, `mm2.5-nvidia`, `minimax-m2.5`, `mm25`, `minimax-m2.5-bedrock`, `mm2.5-bedrock`, `minimax-m3`, `minimax-m3-nvidia`, `mm3-nvidia`, `mm3` | **nothing** — no MiniMax model remains anywhere in the registry. NIM end-of-lifed `minimaxai/minimax-m3` on 2026-09-09 and serves no MiniMax at all; the Bedrock re-host was cut as curation on 2026-08-29 while still live. `kimi-nvidia-3` is the closest free NIM stand-in (multimodal, 1M context, thinking) |
| `kimi-k2.5-nvidia`, `kimi-k2.5`, `kimi25`, `kimi-k2.5-bedrock`, `kimi25-bedrock` | `kimi` / `kimi-k3` (Moonshot direct), or `kimi-bedrock` for Kimi K3 on Bedrock. `kimi-azure`, `kimi25-azure` and `kimi-k2.5-azure` still resolve, now to Moonshot's K3 |
| `kimi-k2.6-nvidia`, `kimi-nvidia-26`, `kimi26-nvidia` | `kimi-nvidia-3` (Kimi K3 on NVIDIA, free) or `kimi` (K3 on Moonshot). The NIM K2.6 endpoint is still listed upstream but is not provisioned for every account; K3 is a different generation, so these names don't follow it |
| `kimi-k2.6`, `kimi26` | **nothing** — the Moonshot entry was upgraded to **K3** on 2026-09-19 and these version-explicit names were deleted rather than migrated. K2.6 is a *different, still-live* model at roughly a third of K3's price ($0.95/$4.00 vs $3.00/$15.00); silently pointing them at K3 would swap the model and triple the bill. Use `kimi` / `kimi-k3` deliberately, or restore the K2.6 entry from git history — no provider code is needed |
| `glm5-bedrock`, `glm-5-bedrock`, `glm5b` | `glm` / `zhipuai/glm-5.3` (Z.AI direct), which is a point release ahead, or the free `glm53-nvidia`. The Bedrock re-host was removed 2026-09-19 as **curation** at explicit request — `zai.glm-5` was **not** re-verified as dead, so this is not an upstream EOL. The version-neutral `glm5`/`glm-5` migrated to Z.AI's 5.3 entry; the provider-explicit spellings were deleted, since the suffix names a provider that no longer serves it. Note Z.AI does serve a distinct real `glm-5`, so `glm5` resolving to 5.3 is a deliberate choice, not an identity |
| `gemini-3.7-flash`, `gemini37-flash`, `gemini3.7-flash` | `gemini-3.8-flash` (or `gemini-flash`) — identical $1.50/$7.50, 1M context and 64K output, and it absorbed 3.7's generation-neutral names. Removed 2026-09-19 as **curation** while still live. ⚠️ 3.7 was the registry's only entry that had *won* `supports_tool_use: true` back with a live thinking run, so it was the documented bar for flipping 3.8 or either GLM-5.3 entry; version-explicit names were deleted rather than migrated because 3.8 is on the prompt path |
| `glm51`, `glm51-nvidia`, `glm-5.1`, `glm5.1`, `glm5.1-zai`, `zhipuai/glm-5.1`, `zhipuai/glm-5.2`, `glm-5.2`, `glm5.2`, `glm5.2-zai`, `glm52`, `glm52-nvidia`, `glm5.2-nvidia`, `glm-5.2-nvidia`, `glm5-nvidia` | `glm` / `zhipuai/glm-5.3` (Z.AI direct). Version-explicit 5.2 names were deleted when 5.3 superseded it at identical price and limits; `glm5`/`glm-5` still resolve, now to this same Z.AI entry |
| `qwen3.5`, `qwen35`, `qwen3.5-nvidia`, `qwen35-nvidia`, `qwen-nvidia`, `qwen3-nvidia`, `qwen-coder-nvidia`, `qwen-next-bedrock`, `qwen-bedrock`, `qwen-next`, `qwen3-next`, `qwen-coder-next`, `qwen`, `qwen-coder` | **nothing** — no Qwen model remains anywhere in the registry. `qwen-next-bedrock` (the last one, and Bedrock's cheapest entry at $0.50/$1.20) was removed 2026-08-29 while still live; the closest replacements are `haiku` on Bedrock or the free `glm53-flash-nvidia` |
| `grok`, `grok-4.3`, `grok43`, `grok-bedrock`, `grok-4.3-bedrock` | **nothing** — xAI leaves the registry entirely, and resolving a Grok name to an OpenAI model would be a vendor swap. `xai.grok-4.3` is still live on `bedrock-mantle`; re-add a YAML entry (no provider code needed) |
| `mistral-small`, `mistral-small-4`, `mistral-small-nvidia`, `ms4`, `mistral-medium`, `mistral-medium-3.5`, `mistral-medium-nvidia`, `mm35`, `mmed` | **nothing** — NIM retired both endpoints and carries no Mistral successor, so no Mistral model remains in this registry |
| `step35`, `step-3.5-flash`, `step-3.7-flash`, `step-3.7`, `step37`, `step37-nvidia`, `step-flash`, `step-nvidia` | **nothing** — NIM serves no StepFun model any more |
| `gpt5.4-bedrock`, `gpt5.5-bedrock` | `gpt5.6` (GPT-5.6 Sol). `gpt-bedrock` still resolves, now to Sol — but at a **higher** rate than GPT-5.5 ($2.50/$15 → $4.40/$22) and a narrower window (400K → 272K) |
| `gpt5.6-sol`, `sol` | `gpt5.6` |
| `gpt54p` | `gpt54-pro` (or `gpt-pro`) |
| `dsv4pro`, `dsv4f` | `dsv4-pro`, `dsv4-flash` |
| `dsv4-azure` | `deepseek-v4-pro` (the Azure deployment is gone) |
| `dsv4-nvidia`, `ds-v4-nvidia`, `deepseek-v4-nvidia`, `deepseek-v4-pro-nvidia` | `deepseek-v4-pro` (DeepSeek direct, same 1.65T model but **billed** at $1.32/$3.96). NIM end-of-lifed `deepseek-v4-pro-0813` on 2026-09-14. These names were **not** redirected to the Flash-on-NIM entry — Pro and Flash shipped as separate concurrent entries, so `dsv4-nvidia` meant "the Pro one" — and that entry has since been removed as well |
| `dsv4-flash-nvidia`, `ds-v4-flash-nvidia`, `deepseek-v4-flash-nvidia` | `dsv4-flash` (DeepSeek direct, same model but **billed** at $0.44/$1.32) for DeepSeek specifically, or `glm53-flash-nvidia` for the free high-volume NIM slot it vacated. NVIDIA sunset `deepseek-ai/deepseek-v4-flash-0731` on 2026-09-21 — the endpoint still returned HTTP 200 but advertised the date in a `deprecation` response header. **No DeepSeek remains on NIM.** Not redirected: every one of these spells `-nvidia`, and the only home left is billed and on a different provider |
| `g31pro`, `g3pro` | `gemini31-pro` (or `gemini-pro`) |
| `gemini-3.6-flash`, `gemini36-flash`, `gemini3.6-flash`, `g36flash` | `gemini-3.8-flash` — identical $1.50/$7.50, 1M context and 64K output, which is why the generation-3 names `gemini-3-flash`/`gemini3-flash`/`g3flash` migrated (first to 3.7, then to 3.8 when 3.7 was curated away). `gemini-flash` also tracks 3.8 |
| `kimi-moonshot` | `kimi` |

Version-*neutral* aliases (`glm5`, `kimi-azure`, `gemini-3-flash`,
`gpt-bedrock`, …) were kept and still resolve to their successor — run
`codereview --list-models --verbose` to see them. `sonnet`/`claude-sonnet` are the one pair
promoted to fully advertised aliases, because Sonnet 5 genuinely is the current Sonnet.
Three families are deleted outright in the table above: `step*` and `qwen*` have no successor
left to point at, `grok*` would mean a vendor swap, and `qwen*-nvidia` would additionally have
crossed both a provider and a billing boundary.

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
│   ├── test_*.py             # Unit tests (1205 tests)
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
- Comprehensive test coverage (1205 tests)

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

Since v0.4.0 (unreleased) the default moved to **Claude Opus 5**, then to **Claude Opus 5.5** (2026-09-23), and the registry was cut to 18
entries across the two 2026-08-29 passes, so the Opus 4.8 and Step 3.7 Flash entries that release
introduced are gone — see [What's New](#whats-new) and the [CHANGELOG](CHANGELOG.md).

Full history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Anthropic Claude](https://www.anthropic.com/), [OpenAI GPT](https://openai.com/), [Google Gemini](https://ai.google.dev/), [DeepSeek](https://www.deepseek.com/), [Moonshot AI](https://www.moonshot.ai/), and [Z.AI](https://z.ai/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service), [NVIDIA NIM](https://build.nvidia.com/), and [Google AI Studio](https://aistudio.google.com/) for model hosting
- Rich library for beautiful terminal output
- Static analysis tools: ruff, mypy, eslint, golangci-lint, shellcheck, bandit, gosec, and more
