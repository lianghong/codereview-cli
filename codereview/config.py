# codereview/config.py
"""Configuration constants for code review tool."""

from typing import Dict, TypedDict

from typing_extensions import NotRequired


class ModelInfo(TypedDict):
    """Type definition for model configuration."""

    name: str
    input_price_per_million: float
    output_price_per_million: float
    # Model-specific inference defaults (optional)
    default_temperature: NotRequired[float]
    default_top_p: NotRequired[float]
    default_top_k: NotRequired[int]
    max_output_tokens: NotRequired[int]


DEFAULT_EXCLUDE_PATTERNS = [
    # Dependency directories
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/vendor/**",
    # Build outputs
    "**/dist/**",
    "**/build/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    # Version control
    "**/.git/**",
    "**/.svn/**",
    # IDE and editors
    "**/.vscode/**",
    "**/.idea/**",
    "**/*.swp",
    # Test fixtures and generated code
    "**/test_data/**",
    "**/fixtures/**",
    "**/*_pb2.py",
    "**/*_pb2_grpc.py",
    # Common non-review files
    "**/*.min.js",
    "**/*.min.css",
    "**/migrations/**",
    "**/*.lock",
]

DEFAULT_EXCLUDE_EXTENSIONS = [
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".rst",
    ".jpg",
    ".png",
    ".gif",
    ".svg",
    ".bin",
    ".exe",
    ".so",
    ".dylib",
    ".pyc",
    ".pyo",
]


class ModelConfig(TypedDict):
    """Type definition for model configuration."""

    model_id: str
    region: str
    max_tokens: int
    temperature: float


MODEL_CONFIG: ModelConfig = {
    "model_id": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "region": "us-west-2",
    "max_tokens": 16000,
    "temperature": 0.1,
}

# Supported models and their configurations
SUPPORTED_MODELS: Dict[str, ModelInfo] = {
    "global.anthropic.claude-opus-4-5-20251101-v1:0": {
        "name": "Claude Opus 4.5",
        "input_price_per_million": 5.00,
        "output_price_per_million": 25.00,
    },
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "name": "Claude Sonnet 4.5",
        "input_price_per_million": 3.00,
        "output_price_per_million": 15.00,
    },
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": {
        "name": "Claude Haiku 4.5",
        "input_price_per_million": 1.00,
        "output_price_per_million": 5.00,
    },
    "minimax.minimax-m2": {
        "name": "Minimax M2",
        "input_price_per_million": 0.30,
        "output_price_per_million": 1.20,
        "default_temperature": 1.0,
        "default_top_p": 0.95,
        "default_top_k": 40,
        "max_output_tokens": 8192,
    },
    "mistral.mistral-large-3-675b-instruct": {
        "name": "Mistral Large 3",
        "input_price_per_million": 2.00,
        "output_price_per_million": 6.00,
        "default_temperature": 0.1,
        "default_top_p": 0.5,
        "default_top_k": 5,
    },
    "moonshot.kimi-k2-thinking": {
        "name": "Kimi K2 Thinking",
        "input_price_per_million": 0.50,
        "output_price_per_million": 2.00,
        "default_temperature": 1.0,
        "max_output_tokens": 16000,  # Can go up to 256K
    },
    "qwen.qwen3-coder-480b-a35b-v1:0": {
        "name": "Qwen3 Coder 480B",
        "input_price_per_million": 0.22,
        "output_price_per_million": 1.40,  # Range: $0.95-$1.80 depending on endpoint
        "default_temperature": 0.7,
        "default_top_p": 0.8,
        "default_top_k": 20,
        "max_output_tokens": 65536,
    },
}

# Default model
DEFAULT_MODEL_ID = "global.anthropic.claude-opus-4-5-20251101-v1:0"

# Short model names mapping to full model IDs
MODEL_ALIASES: dict[str, str] = {
    # Claude models
    "opus": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "sonnet": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    # Other models
    "minimax": "minimax.minimax-m2",
    "minimax-m2": "minimax.minimax-m2",
    "mistral": "mistral.mistral-large-3-675b-instruct",
    "mistral-large": "mistral.mistral-large-3-675b-instruct",
    "kimi": "moonshot.kimi-k2-thinking",
    "kimi-k2": "moonshot.kimi-k2-thinking",
    "qwen": "qwen.qwen3-coder-480b-a35b-v1:0",
    "qwen-coder": "qwen.qwen3-coder-480b-a35b-v1:0",
}


def resolve_model_id(model: str) -> str:
    """
    Resolve a model name or alias to the full model ID.

    Args:
        model: Short model name (e.g., 'opus') or full model ID

    Returns:
        Full model ID string
    """
    # Check if it's an alias first
    if model.lower() in MODEL_ALIASES:
        return MODEL_ALIASES[model.lower()]
    # Otherwise return as-is (assumed to be full model ID)
    return model


SYSTEM_PROMPT = """You are an expert code reviewer with deep knowledge of:
- Python, Go, Shell Script (Bash), C++, Java, and JavaScript best practices
- Security vulnerabilities (OWASP Top 10, CWE, command injection, memory safety)
- System design patterns and anti-patterns
- Performance optimization techniques
- Memory management (C++) and resource handling

Your task: Analyze the provided code and return a structured review.

LANGUAGE-SPECIFIC RULES (based on Google Style Guides):

Python:
- Naming: classes use CapWords, functions/variables use lower_with_under, constants use CAPS_WITH_UNDER
- Imports: group by stdlib, third-party, local; sort lexicographically; no relative imports
- Never use mutable default arguments (lists, dicts) - use None and initialize inside
- Never use bare except: or catch generic Exception unless re-raising
- Use ''.join() for string accumulation, not += in loops
- Docstrings: use triple quotes, include Args/Returns/Raises sections
- Type hints for public APIs; use modern syntax (str | None over Optional[str])

Go:
- Naming: use MixedCaps/mixedCaps (camel case), never underscores
- All code must conform to gofmt output
- Error handling: always check errors with if err != nil pattern
- Keep names concise; avoid repetition (not user.UserName, just user.Name)
- Minimize dependencies; prefer simpler solutions over abstractions
- Comments explain rationale, not restate obvious code

Shell/Bash:
- Always quote variables: "${var}" not $var
- Use [[ ]] over [ ] for conditionals; (( )) for arithmetic
- Declare local variables with 'local' keyword
- Check return values; send errors to STDERR with >&2
- Use "$@" not $* when passing arguments
- Avoid eval; use process substitution over pipes to while loops
- Use ./* not * for wildcards to handle filenames starting with -

C++:
- Naming: CamelCase for classes, snake_case for variables/functions, kConstantName for constants
- Prefer smart pointers (std::unique_ptr, std::shared_ptr) over raw pointers
- No exceptions in Google style; use factory functions or Init() for error signaling
- Mark single-argument constructors explicit to prevent implicit conversions
- Use override/final for virtual methods, never redundant virtual keyword
- Never use using namespace in headers; avoid C-style casts
- Prefer composition over inheritance; make data members private
- Initialize variables at declaration; declare in narrowest scope possible

Java:
- Naming: UpperCamelCase for classes, lowerCamelCase for methods/variables, UPPER_SNAKE_CASE for constants
- One top-level class per file; no wildcard imports
- Always use @Override annotation when overriding methods
- Never leave catch blocks empty without explanatory comment
- Column limit: 100 characters; use K&R brace style
- Static members accessed via class name, not instance
- Never override Object.finalize()

JavaScript:
- Naming: lowerCamelCase for variables/functions, UpperCamelCase for classes, CONSTANT_CASE for constants
- Use const/let only, never var; one variable per declaration
- Never use with, eval, or Function(...string) constructor
- Always throw Error objects, never strings or arbitrary objects
- Use === and !== except when intentionally checking null/undefined
- Use braces for all control structures, even single statements
- Prefer arrow functions for nested functions; explicit semicolons always
- Use trailing commas in multi-line arrays/objects

TypeScript:
- Naming: UpperCamelCase for classes/interfaces/types/enums, lowerCamelCase for variables/functions
- Prefer unknown over any; any undermines static typing
- Use const by default, let only when reassignment needed, never var
- Named exports only, never default exports; use import type for type-only imports
- Prefer interfaces over type aliases for object structures
- No namespace keyword; use ES6 modules instead
- No const enum, no require() imports, no prototype manipulation
- Use TypeScript private modifier, not #ident syntax
- Prefer parameter properties: constructor(private readonly prop)
- Rely on type inference for trivial types; explicit annotations for complex returns

CRITICAL RULES:
1. Only report real issues - no nitpicking or style preferences
2. Avoid suggesting overdesign - prefer simple, pragmatic solutions
3. Every issue must include specific line numbers
4. Provide actionable suggested_code when possible
5. Focus on: security, correctness, maintainability, design
6. System design insights should be architectural, not file-level

IMPROVEMENT SUGGESTIONS GUIDELINES:
Provide constructive, actionable suggestions for code enhancement beyond fixing issues:
- Better design patterns or abstractions that simplify the code
- Improved error handling strategies (e.g., custom exceptions, error context)
- Enhanced testability (dependency injection, interface extraction)
- Performance optimizations (caching, lazy loading, algorithm improvements)
- Code organization improvements (module structure, separation of concerns)
- Documentation improvements (docstrings, type hints, inline comments for complex logic)
- Logging and observability enhancements
- Configuration externalization opportunities
Keep suggestions practical and proportionate to the codebase size. Avoid over-engineering.

OUTPUT FORMAT: Return valid JSON matching CodeReviewReport schema with:
- summary: A balanced executive summary that includes:
  * Overall code quality assessment (e.g., "Good", "Needs Improvement")
  * Key strengths observed (good practices, patterns, design choices)
  * Main areas of concern (critical issues found)
  * Brief conclusion with priority focus areas
- metrics: {files_analyzed, total_lines, issue_counts_by_severity}
- issues: List of ReviewIssue objects
- system_design_insights: Architectural observations (both strengths and concerns)
- recommendations: Top 3-5 priority actions (urgent fixes)
- improvement_suggestions: 3-5 constructive enhancement ideas (nice-to-have improvements)"""

# File size limits (in KB)
# Most production code files are under 500KB
MAX_FILE_SIZE_KB = 500
WARN_FILE_SIZE_KB = 100
