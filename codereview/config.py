# codereview/config.py
"""Configuration constants for code review tool."""

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
    ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".rst",
    ".jpg", ".png", ".gif", ".svg",
    ".bin", ".exe", ".so", ".dylib",
    ".pyc", ".pyo",
]

MODEL_CONFIG = {
    "model_id": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "region": "us-west-2",
    "max_tokens": 16000,
    "temperature": 0.1,
}

SYSTEM_PROMPT = """You are an expert code reviewer with deep knowledge of:
- Python and Go best practices
- Security vulnerabilities (OWASP Top 10, CWE)
- System design patterns and anti-patterns
- Performance optimization techniques

Your task: Analyze the provided code and return a structured review.

CRITICAL RULES:
1. Only report real issues - no nitpicking or style preferences
2. Avoid suggesting overdesign - prefer simple, pragmatic solutions
3. Every issue must include specific line numbers
4. Provide actionable suggested_code when possible
5. Focus on: security, correctness, maintainability, design
6. System design insights should be architectural, not file-level

OUTPUT FORMAT: Return valid JSON matching CodeReviewReport schema with:
- summary: Executive summary of findings
- metrics: {files_analyzed, total_lines, issue_counts_by_severity}
- issues: List of ReviewIssue objects
- system_design_insights: Architectural observations
- recommendations: Top 3-5 priority actions"""

# File size limits
MAX_FILE_SIZE_KB = 10
WARN_FILE_SIZE_KB = 5
