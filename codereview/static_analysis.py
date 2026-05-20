"""Static analysis integration for code quality tools across multiple languages."""

import json
import logging
import re
import shutil
import subprocess  # nosec B404 - required for running static analysis tools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# Per-tool regexes that pull an authoritative issue count from the tool's
# own summary line. Falling back to substring counting (BASE_INDICATORS
# below) double-counts: ruff prints both per-issue lines and a "Found N
# errors" summary; mypy prints "error:" on each diagnostic *and* a
# trailing "Found N errors in M files".
_TOOL_COUNT_PATTERNS: dict[str, re.Pattern[str]] = {
    "ruff": re.compile(r"^Found (\d+) errors?\.?$", re.MULTILINE),
    "ruff-format": re.compile(r"^(\d+) files? would be reformatted", re.MULTILINE),
    "mypy": re.compile(r"^Found (\d+) errors? in \d+ files?", re.MULTILINE),
}

# mypy emits one diagnostic per line as ``path:line: severity: message``
# (the column is optional); use this when the summary line is absent.
_MYPY_DIAGNOSTIC = re.compile(
    r"^[^:\n]+:\d+(?::\d+)?:\s+(error|warning|note):", re.MULTILINE
)

# Maximum number of files to pass to command line tools.
# Prevents "Argument list too long" errors on large repos.
#
# DESIGN GUARANTEE: when truncation triggers, the file list MUST be sorted
# before slicing. Filesystem walk order is non-deterministic across runs
# (and across hosts), so an unsorted [:N] would silently change the
# analyzed subset between runs and break CI reproducibility. Test:
# tests/test_static_analysis.py::test_truncation_is_deterministic.
MAX_FILES_PER_TOOL = 500

# Issue detection indicators for counting problems in tool output
# Base indicators common to all tools
BASE_INDICATORS = [
    "error",
    "warning",
    "would reformat",
    "would be reformatted",
    "unused",
]

# Language-specific indicators
GO_INDICATORS = [
    "typecheck",
    "deadcode",
    "ineffassign",
    # gosec security indicators (G1xx-G6xx)
    "g101",
    "g102",
    "g103",
    "g104",
    "g105",
    "g106",
    "g107",
    "g108",
    "g109",
    "g110",
    "g201",
    "g202",
    "g203",
    "g204",
    "g301",
    "g302",
    "g303",
    "g304",
    "g305",
    "g306",
    "g307",
    "g401",
    "g402",
    "g403",
    "g404",
    "g501",
    "g502",
    "g503",
    "g504",
    "g505",
    "g601",
    "severity",
    "confidence",
]

SHELL_INDICATORS = [
    # ShellCheck SC codes (SC1000-SC9999)
    "sc1",
    "sc2",
    "sc3",
    "sc4",
    # bashate E codes (E001-E099) - listed as literal prefixes since
    # indicator matching uses substring containment, not regex
    "e001",
    "e002",
    "e003",
    "e004",
    "e005",
    "e006",
    "e010",
    "e011",
    "e012",
    "e040",
    "e041",
    "e042",
    "e043",
    "e044",
    "note",
]

CPP_INDICATORS = ["[performance]", "[portability]", "[style]", "cppcheck"]

JAVA_INDICATORS = ["audit", "violation", "checkstyle"]

JS_TS_INDICATORS = [
    "eslint",
    "prettier",
    "problems found",
    # npm audit indicators
    "vulnerabilities",
]

PYTHON_INDICATORS = [
    # Bandit security indicators (B1xx-B7xx)
    ">> issue:",
    "severity:",
    "confidence:",
    "test_id",
]

# Map language to indicators
LANGUAGE_INDICATORS: dict[str, list[str]] = {
    "go": GO_INDICATORS,
    "shell": SHELL_INDICATORS,
    "cpp": CPP_INDICATORS,
    "java": JAVA_INDICATORS,
    "javascript": JS_TS_INDICATORS,
    "typescript": JS_TS_INDICATORS,
    "python": PYTHON_INDICATORS,
}


class StaticAnalysisSummary(TypedDict):
    """Summary of static analysis results."""

    tools_run: int
    tools_passed: int
    tools_failed: int
    total_issues: int
    passed: bool


@dataclass
class StaticAnalysisResult:
    """Results from static analysis tools."""

    tool: str
    passed: bool
    issues_count: int
    output: str
    errors: list[str]


class StaticAnalyzer:
    """Runs static analysis tools on code across multiple languages."""

    TOOLS = {
        # Python tools
        "ruff": {
            "name": "Ruff",
            "description": "Fast Python linter",
            "command": ["ruff", "check"],
            "language": "python",
        },
        "ruff-format": {
            "name": "Ruff Format",
            "description": "Fast Python code formatter",
            "command": ["ruff", "format", "--check", "--diff"],
            "language": "python",
            "version_command": ["ruff", "version"],
        },
        "mypy": {
            "name": "Mypy",
            "description": "Static type checker",
            "command": ["mypy"],
            "language": "python",
        },
        "black": {
            "name": "Black",
            "description": "Code formatter",
            "command": ["black", "--check", "--diff"],
            "language": "python",
        },
        "isort": {
            "name": "isort",
            "description": "Import sorter",
            "command": ["isort", "--check-only", "--diff"],
            "language": "python",
        },
        "vulture": {
            "name": "Vulture",
            "description": "Dead code finder",
            "command": ["vulture", "--min-confidence", "80"],
            "language": "python",
        },
        "bandit": {
            "name": "Bandit",
            "description": "Python security vulnerability scanner",
            "command": ["bandit", "-r", "-f", "txt"],
            "language": "python",
        },
        # Go tools
        "golangci-lint": {
            "name": "golangci-lint",
            "description": "Go meta-linter",
            "command": ["golangci-lint", "run"],
            "language": "go",
        },
        "go-vet": {
            "name": "go vet",
            "description": "Go static analyzer",
            "command": ["go", "vet"],
            "language": "go",
            "version_command": ["go", "version"],
        },
        "gofmt": {
            "name": "gofmt",
            "description": "Go formatter",
            "command": ["gofmt", "-l"],
            "language": "go",
            "version_command": ["go", "version"],
        },
        "gosec": {
            "name": "gosec",
            "description": "Go security vulnerability scanner",
            "command": ["gosec", "-fmt=text"],
            "language": "go",
        },
        # Shell Script tools
        "shellcheck": {
            "name": "ShellCheck",
            "description": "Shell script static analyzer",
            "command": ["shellcheck"],
            "language": "shell",
        },
        "bashate": {
            "name": "bashate",
            "description": "Shell script style checker (PEP8-like for bash)",
            "command": ["bashate"],
            "language": "shell",
        },
        # C++ tools
        "clang-tidy": {
            "name": "clang-tidy",
            "description": "C++ static analyzer and linter",
            "command": ["clang-tidy", "-checks=*,-llvmlibc-*"],
            "language": "cpp",
        },
        "cppcheck": {
            "name": "cppcheck",
            "description": "C++ static analysis tool",
            "command": ["cppcheck", "--enable=all", "--error-exitcode=1"],
            "language": "cpp",
        },
        "clang-format": {
            "name": "clang-format",
            "description": "C++ code formatter",
            "command": ["clang-format", "--dry-run", "--Werror"],
            "language": "cpp",
        },
        # Java tools
        "checkstyle": {
            "name": "Checkstyle",
            "description": "Java style checker",
            "command": ["checkstyle", "-c", "google_checks.xml"],
            "language": "java",
        },
        # JavaScript/TypeScript tools
        "eslint": {
            "name": "ESLint",
            "description": "JavaScript/TypeScript linter",
            "command": ["eslint"],
            "language": "javascript",
        },
        "prettier": {
            "name": "Prettier",
            "description": "Code formatter for JS/TS/CSS/HTML",
            "command": ["prettier", "--check"],
            "language": "javascript",
        },
        "tsc": {
            "name": "TypeScript Compiler",
            "description": "TypeScript type checker",
            "command": ["tsc", "--noEmit"],
            "language": "typescript",
            "version_command": ["tsc", "--version"],
        },
        "npm-audit": {
            "name": "npm audit",
            "description": "JavaScript/TypeScript dependency vulnerability scanner",
            "command": ["npm", "audit", "--json"],
            "language": "javascript",
            "version_command": ["npm", "--version"],
        },
    }

    # Default per-tool subprocess timeout in seconds. Caller can override via
    # the constructor; CLI exposes this as --tool-timeout. 120s covers most
    # repos but slow runs (cppcheck --enable=all, mypy strict on large
    # codebases) need more.
    DEFAULT_TOOL_TIMEOUT_SECONDS = 120

    def __init__(self, directory: Path, tool_timeout: int | None = None):
        """
        Initialize static analyzer.

        Args:
            directory: Directory to analyze.
            tool_timeout: Per-tool subprocess timeout in seconds. Defaults to
                ``DEFAULT_TOOL_TIMEOUT_SECONDS`` (120). Raise this for slow
                cppcheck/mypy runs on large repos.

        Raises:
            ValueError: If ``directory`` does not exist, is not a directory,
                or contains null bytes; or if ``tool_timeout`` is non-positive.
                Failing fast here surfaces config errors before any tool runs
                and lets callers handle them in one place rather than per-tool.
        """
        if tool_timeout is not None and tool_timeout <= 0:
            raise ValueError(
                f"tool_timeout must be a positive integer, got {tool_timeout}"
            )
        self.tool_timeout = tool_timeout or self.DEFAULT_TOOL_TIMEOUT_SECONDS

        # Resolve path immediately to handle symlinks and normalize
        self.directory = directory.resolve()
        self._validate_directory()
        # Maps tool executable name → absolute resolved path on PATH.
        # Populated by _check_available_tools so run_tool doesn't have to
        # re-resolve (and re-validate) on every invocation.
        self._tool_paths: dict[str, str] = {}
        self.available_tools = self._check_available_tools()

    def _validate_directory(self) -> None:
        """Validate that ``self.directory`` is safe to use, or raise.

        Raises:
            ValueError: with a human-readable reason when the path is
                missing, not a directory, or contains a null byte.
        """
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {self.directory}")
        if not self.directory.is_dir():
            raise ValueError(f"Path is not a directory: {self.directory}")
        if "\x00" in str(self.directory):
            raise ValueError(f"Invalid characters in path: {self.directory!r}")

    def _validate_file_path(self, file_path: Path) -> bool:
        """
        Validate that a file path is within the analysis directory.

        This prevents path traversal attacks where a symlink or crafted
        path could escape the intended directory.

        Args:
            file_path: Path to validate

        Returns:
            True if file is safe to use, False otherwise
        """
        try:
            # Resolve file path to handle symlinks
            resolved_file = file_path.resolve()

            # Check if file is within directory (prevents path traversal)
            # self.directory is already resolved in __init__
            resolved_file.relative_to(self.directory)
            return True
        except ValueError:
            # relative_to raises ValueError if path is not relative to directory
            logging.warning(
                "File path %s is outside analysis directory, skipping", file_path
            )
            return False

    def _filter_safe_files(self, files: list[Path]) -> list[Path]:
        """
        Filter file list to only include files within the analysis directory.

        Args:
            files: List of file paths from rglob

        Returns:
            Filtered list containing only safe paths
        """
        return [f for f in files if self._validate_file_path(f)]

    def _safe_rglob(self, pattern: str) -> list[Path]:
        """Recursively glob for files, handling permission errors gracefully.

        Wraps Path.rglob() to catch OSError (including PermissionError) that
        can occur when encountering directories with restricted permissions.
        Symlinks are skipped to prevent following links that point outside
        the analysis directory (defense-in-depth alongside _validate_file_path).

        Args:
            pattern: Glob pattern (e.g., "*.go", "*.py")

        Returns:
            List of matching paths, or empty list if scanning fails
        """
        try:
            return [p for p in self.directory.rglob(pattern) if not p.is_symlink()]
        except OSError as e:
            logging.warning("Error scanning for %s files: %s", pattern, e)
            return []

    def _safe_rglob_suffixes(self, suffixes: set[str]) -> list[Path]:
        """Single-pass recursive glob filtered by file suffix set.

        Equivalent to calling _safe_rglob once per suffix, but walks the
        tree a single time — useful when a tool accepts many extensions
        (e.g. prettier covers 8 suffixes, C++ covers 5). Symlinks are
        skipped (see _safe_rglob).

        Args:
            suffixes: Set of extensions including the dot (e.g. {".js", ".tsx"}).

        Returns:
            List of matching file paths, or empty list if scanning fails.
        """
        try:
            return [
                p
                for p in self.directory.rglob("*")
                if p.is_file() and not p.is_symlink() and p.suffix in suffixes
            ]
        except OSError as e:
            logging.warning("Error scanning tree for %s: %s", sorted(suffixes), e)
            return []

    def _resolve_tool_binary(self, executable: str) -> str | None:
        """Resolve a tool name to an absolute path on PATH.

        Rejects binaries that resolve inside the analyzed directory: a repo
        could otherwise ship its own ``ruff``/``eslint``/``npm`` (e.g. via
        ``node_modules/.bin``) and have it run with the user's privileges.

        Args:
            executable: Tool name as it appears on PATH (e.g. ``"ruff"``).

        Returns:
            Absolute path to the binary, or None if not found / inside the
            analyzed directory.
        """
        path = shutil.which(executable)
        if path is None:
            return None
        try:
            resolved = Path(path).resolve()
        except OSError:
            return None
        if resolved.is_relative_to(self.directory):
            logging.warning(
                "Refusing to run %s resolved inside analyzed directory: %s",
                executable,
                resolved,
            )
            return None
        return str(resolved)

    def _check_available_tools(self) -> list[str]:
        """Check which static analysis tools are installed and available.

        Resolves each tool via PATH (rejecting binaries inside the analyzed
        directory) and runs its version check. Tools whose version check
        succeeds are considered available; the rest are skipped.

        Returns:
            List of available tool names from TOOLS configuration.
        """
        available = []
        for tool_name, config in self.TOOLS.items():
            version_cmd = list(config.get("version_command", [tool_name, "--version"]))
            resolved = self._resolve_tool_binary(version_cmd[0])
            if resolved is None:
                logging.debug("Tool %s not available on PATH", tool_name)
                continue
            try:
                result = subprocess.run(  # nosec B603 - resolved absolute path
                    [resolved, *version_cmd[1:]],
                    capture_output=True,
                    timeout=5,
                    shell=False,
                )
                if result.returncode == 0:
                    # Cache the resolved exe so run_tool doesn't re-resolve.
                    # The first command token is the tool's primary executable.
                    # When the version probe uses a *different* binary (gofmt
                    # uses `go version`), we must validate the primary
                    # separately — caching the bare name would let
                    # subprocess.run search PATH again at run time, defeating
                    # the in-repo-binary check in _resolve_tool_binary.
                    primary = config["command"][0]
                    if primary == version_cmd[0]:
                        primary_resolved: str | None = resolved
                    else:
                        primary_resolved = self._resolve_tool_binary(primary)
                    if primary_resolved is None:
                        logging.debug(
                            "Tool %s passed version check via %s but primary "
                            "executable %s could not be safely resolved on PATH",
                            tool_name,
                            version_cmd[0],
                            primary,
                        )
                        continue
                    available.append(tool_name)
                    self._tool_paths[primary] = primary_resolved
            except subprocess.TimeoutExpired:
                logging.debug(
                    "Tool %s version check timed out, marking as unavailable",
                    tool_name,
                )
            except PermissionError:
                logging.debug("Tool %s not executable (permission denied)", tool_name)
        return available

    def run_tool(self, tool_name: str) -> StaticAnalysisResult:
        """
        Run a specific static analysis tool.

        Args:
            tool_name: Name of the tool to run

        Returns:
            StaticAnalysisResult with tool output
        """
        if tool_name not in self.TOOLS:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=[f"Unknown tool: {tool_name}"],
            )

        if tool_name not in self.available_tools:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=[f"Tool not installed: {tool_name}"],
            )

        # Directory was validated in __init__; no per-call check needed.
        tool_config = self.TOOLS[tool_name]
        language = str(tool_config.get("language", "python"))
        dir_path = str(self.directory)  # Already resolved in __init__

        # Build command based on tool type
        # Each tool has different CLI conventions for specifying targets:
        base_command = list(tool_config["command"])
        if tool_name == "go-vet":
            # go vet: Uses Go's "./..." pattern to recursively check all packages
            # in the current module. Must run from within the Go module directory.
            # Check if there are any Go files first
            go_files = self._safe_rglob("*.go")
            if not go_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No Go files found",
                    errors=[],
                )
            command = base_command + ["./..."]
            cwd = self.directory
        elif tool_name == "golangci-lint":
            # golangci-lint: Automatically finds Go files when run in a Go module.
            # No path argument needed; it uses the current working directory.
            # Check if there are any Go files first
            go_files = self._safe_rglob("*.go")
            if not go_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No Go files found",
                    errors=[],
                )
            command = base_command
            cwd = self.directory
        elif tool_name == "gosec":
            # gosec: Go security scanner, scans recursively from directory
            go_files = self._safe_rglob("*.go")
            if not go_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No Go files found",
                    errors=[],
                )
            command = base_command + ["./..."]
            cwd = self.directory
        elif tool_name in ("shellcheck", "bashate"):
            # shellcheck/bashate: Do NOT support directory scanning.
            # We must explicitly pass each shell script file as an argument.
            shell_files = self._safe_rglob("*.sh") + self._safe_rglob("*.bash")
            # Validate files are within analysis directory (security measure)
            shell_files = self._filter_safe_files(shell_files)
            if not shell_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No shell scripts found",
                    errors=[],
                )
            if len(shell_files) > MAX_FILES_PER_TOOL:
                logging.warning(
                    "%s: Limiting to %d of %d files to avoid command line length limits",
                    tool_name,
                    MAX_FILES_PER_TOOL,
                    len(shell_files),
                )
                # Sort before truncating: rglob order is filesystem-dependent
                # and would otherwise make the analyzed subset non-reproducible
                # across runs (problematic for CI quality gates).
                shell_files = sorted(shell_files)[:MAX_FILES_PER_TOOL]
            command = base_command + [str(f) for f in shell_files]
            cwd = self.directory
        elif tool_name in ("clang-tidy", "clang-format"):
            # C++ tools that need explicit file list (single tree walk)
            cpp_files = self._safe_rglob_suffixes({".cpp", ".cc", ".cxx", ".h", ".hpp"})
            # Validate files are within analysis directory (security measure)
            cpp_files = self._filter_safe_files(cpp_files)
            if not cpp_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No C++ files found",
                    errors=[],
                )
            if len(cpp_files) > MAX_FILES_PER_TOOL:
                logging.warning(
                    "%s: Limiting to %d of %d files to avoid command line length limits",
                    tool_name,
                    MAX_FILES_PER_TOOL,
                    len(cpp_files),
                )
                cpp_files = sorted(cpp_files)[:MAX_FILES_PER_TOOL]
            command = base_command + [str(f) for f in cpp_files]
            cwd = self.directory
        elif tool_name == "cppcheck":
            # cppcheck accepts directory path
            command = base_command + [dir_path]
            cwd = self.directory
        elif tool_name == "checkstyle":
            # Checkstyle needs explicit Java file list
            java_files = self._safe_rglob("*.java")
            # Validate files are within analysis directory (security measure)
            java_files = self._filter_safe_files(java_files)
            if not java_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No Java files found",
                    errors=[],
                )
            if len(java_files) > MAX_FILES_PER_TOOL:
                logging.warning(
                    "%s: Limiting to %d of %d files to avoid command line length limits",
                    tool_name,
                    MAX_FILES_PER_TOOL,
                    len(java_files),
                )
                java_files = sorted(java_files)[:MAX_FILES_PER_TOOL]
            command = base_command + [str(f) for f in java_files]
            cwd = self.directory
        elif tool_name == "eslint":
            # ESLint accepts the directory itself, but we still need to
            # confirm at least one JS/TS source exists; otherwise eslint
            # exits non-zero and we'd report a spurious failure.
            # Use the single-pass walk for consistency with prettier/clang-tidy.
            if not self._safe_rglob_suffixes({".js", ".jsx", ".ts", ".tsx"}):
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No JavaScript/TypeScript files found",
                    errors=[],
                )
            command = base_command + [dir_path]
            cwd = self.directory
        elif tool_name == "prettier":
            # Prettier needs files to format (single tree walk across 8 suffixes)
            prettier_files = self._safe_rglob_suffixes(
                {".js", ".jsx", ".ts", ".tsx", ".css", ".html", ".json", ".md"}
            )
            # Validate files are within analysis directory (security measure)
            prettier_files = self._filter_safe_files(prettier_files)
            if not prettier_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No files found for Prettier to format",
                    errors=[],
                )
            # Pass explicit files to prettier (--ignore-unknown may not be supported)
            if len(prettier_files) > MAX_FILES_PER_TOOL:
                logging.warning(
                    "%s: Limiting to %d of %d files to avoid command line length limits",
                    tool_name,
                    MAX_FILES_PER_TOOL,
                    len(prettier_files),
                )
                prettier_files = sorted(prettier_files)[:MAX_FILES_PER_TOOL]
            command = base_command + [str(f) for f in prettier_files]
            cwd = self.directory
        elif tool_name == "tsc":
            # TypeScript compiler needs tsconfig.json or explicit files
            tsconfig_path = self.directory / "tsconfig.json"
            if not tsconfig_path.exists():
                # No tsconfig.json - check for TypeScript files
                ts_files = self._safe_rglob("*.ts") + self._safe_rglob("*.tsx")
                # Filter out .d.ts declaration files
                ts_files = [f for f in ts_files if not str(f).endswith(".d.ts")]
                # Validate files are within analysis directory (security measure)
                ts_files = self._filter_safe_files(ts_files)
                if not ts_files:
                    return StaticAnalysisResult(
                        tool=tool_name,
                        passed=True,
                        issues_count=0,
                        output="No TypeScript files found (no tsconfig.json)",
                        errors=[],
                    )
                # Pass explicit files to tsc
                if len(ts_files) > MAX_FILES_PER_TOOL:
                    logging.warning(
                        "%s: Limiting to %d of %d files to avoid command line length limits",
                        tool_name,
                        MAX_FILES_PER_TOOL,
                        len(ts_files),
                    )
                    ts_files = sorted(ts_files)[:MAX_FILES_PER_TOOL]
                command = base_command + [str(f) for f in ts_files]
            else:
                # tsconfig.json exists, tsc will use it
                command = base_command
            cwd = self.directory
        elif tool_name == "npm-audit":
            # npm audit: Requires package.json and node_modules in the directory
            package_json = self.directory / "package.json"
            if not package_json.exists():
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No package.json found",
                    errors=[],
                )
            # npm audit runs from the project directory
            command = base_command
            cwd = self.directory
        elif tool_name == "gofmt":
            # gofmt accepts a directory path but needs Go files to exist
            go_files = self._filter_safe_files(self._safe_rglob("*.go"))
            if not go_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No Go files found",
                    errors=[],
                )
            command = base_command + [dir_path]
            cwd = self.directory
        elif tool_name == "bandit":
            # bandit: Python security scanner, accepts directory path
            # Check for Python files first
            py_files = self._safe_rglob("*.py")
            if not py_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No Python files found",
                    errors=[],
                )
            command = base_command + [dir_path]
            cwd = self.directory
        else:
            # Default (ruff, mypy, black, isort, vulture):
            # These tools accept a directory path as the final argument.
            command = base_command + [dir_path]
            cwd = self.directory

        # Substitute the cached, validated absolute path for the executable.
        # Cache is populated by _check_available_tools, which rejects binaries
        # inside the analyzed directory (supply-chain defense).
        resolved_exe = self._tool_paths.get(command[0])
        if resolved_exe is not None:
            command = [resolved_exe, *command[1:]]

        try:
            result = subprocess.run(  # nosec B603 - resolved absolute path
                command,
                capture_output=True,
                text=True,
                timeout=self.tool_timeout,
                cwd=cwd,
                shell=False,  # Security: prevent command injection
            )

            output = result.stdout + result.stderr

            # Determine pass/fail based on tool
            if tool_name == "gofmt":
                # gofmt -l outputs filenames of unformatted files
                # Empty output = all files formatted correctly
                passed = len(result.stdout.strip()) == 0
            else:
                # Most tools return non-zero when they find issues
                passed = result.returncode == 0

            # Count issues (rough estimation based on output lines)
            issues_count = 0
            if not passed:
                if tool_name == "gofmt":
                    # Count unformatted files
                    issues_count = len(
                        [
                            line
                            for line in result.stdout.strip().split("\n")
                            if line.strip()
                        ]
                    )
                elif tool_name == "npm-audit":
                    # npm audit --json outputs structured JSON; parse it
                    # to get accurate vulnerability counts
                    issues_count = self._count_npm_audit_issues(output)
                else:
                    issues_count = self._count_issues(tool_name, language, output)

            return StaticAnalysisResult(
                tool=tool_name,
                passed=passed,
                issues_count=issues_count,
                output=output.strip(),
                errors=[],
            )

        except subprocess.TimeoutExpired:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=[
                    f"Tool timed out after {self.tool_timeout} seconds "
                    "(raise --tool-timeout for slow runs)"
                ],
            )
        except (OSError, subprocess.SubprocessError) as e:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=[f"Error running tool: {e}"],
            )

    @staticmethod
    def _count_issues(tool_name: str, language: str, output: str) -> int:
        """Count issues from a tool's text output.

        Strategy: prefer an authoritative summary-line regex when the tool
        emits one (ruff/mypy/bandit do); otherwise fall back to counting
        lines that contain a known indicator. The substring fallback is
        crude — a single error line containing both ``error`` and ``warning``
        counts once, and tool summary lines like ``Found 12 errors`` would
        otherwise double-count alongside the per-issue lines.
        """
        pattern = _TOOL_COUNT_PATTERNS.get(tool_name)
        if pattern is not None:
            match = pattern.search(output)
            if match is not None:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass

        # mypy without the summary line (e.g. --no-error-summary or
        # partial output): count diagnostic lines directly.
        if tool_name == "mypy":
            return len(_MYPY_DIAGNOSTIC.findall(output))

        # bandit text output: each finding is introduced by ">> Issue:".
        if tool_name == "bandit":
            return output.count(">> Issue:")

        # Fallback: per-line indicator scan. Imprecise but better than nothing
        # for tools whose output format we haven't characterized.
        indicators = list(BASE_INDICATORS)
        indicators.extend(LANGUAGE_INDICATORS.get(language, []))
        count = 0
        for line in output.split("\n"):
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in indicators):
                count += 1
        return count

    @staticmethod
    def _count_npm_audit_issues(output: str) -> int:
        """Parse npm audit JSON output and return the total vulnerability count.

        Args:
            output: Raw stdout+stderr from `npm audit --json`

        Returns:
            Total number of vulnerabilities found
        """
        try:
            data = json.loads(output)
            # npm audit v2+ includes metadata.vulnerabilities with severity totals
            # Check this first as it provides accurate counts (not just package count)
            if "metadata" in data and isinstance(data.get("metadata"), dict):
                vuln_counts = data["metadata"].get("vulnerabilities", {})
                if isinstance(vuln_counts, dict):
                    return sum(
                        v for v in vuln_counts.values() if isinstance(v, int) and v > 0
                    )
            # Fallback: v2+ package map when metadata is unavailable
            if "vulnerabilities" in data and isinstance(data["vulnerabilities"], dict):
                return sum(
                    1
                    for pkg in data["vulnerabilities"].values()
                    if isinstance(pkg, dict) and pkg.get("severity")
                )
        # PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
        except json.JSONDecodeError, TypeError, AttributeError:
            # Without this log, an HTML error page from a corporate proxy
            # would silently inflate "issue count" to one-per-line. Operators
            # debugging phantom vulnerability counts need to see that JSON
            # parsing failed so they can investigate the raw output.
            logging.warning(
                "npm audit JSON parsing failed; falling back to non-empty "
                "line count. Output preview: %s",
                output[:200].replace("\n", " "),
            )
            return len([line for line in output.split("\n") if line.strip()])
        return 0

    def run_all(self, parallel: bool = True) -> dict[str, StaticAnalysisResult]:
        """
        Run all available static analysis tools.

        Args:
            parallel: Run tools in parallel for faster execution (default: True)

        Returns:
            Dictionary mapping tool names to results
        """
        if not self.available_tools:
            return {}

        if not parallel or len(self.available_tools) == 1:
            # Sequential execution
            return {tool: self.run_tool(tool) for tool in self.available_tools}

        # Parallel execution using ThreadPoolExecutor
        # Cap workers to prevent resource exhaustion in containerized environments
        results = {}
        max_workers = min(len(self.available_tools), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tool = {
                executor.submit(self.run_tool, tool): tool
                for tool in self.available_tools
            }
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    results[tool_name] = future.result()
                except Exception as e:
                    # Log full exception for debugging, store summary in result
                    logging.exception("Unexpected error running tool %s", tool_name)
                    results[tool_name] = StaticAnalysisResult(
                        tool=tool_name,
                        passed=False,
                        issues_count=0,
                        output="",
                        errors=[f"Execution error: {type(e).__name__}: {e}"],
                    )

        # Distinguish "lots of code-quality issues" from "every tool blew up
        # on infrastructure problems" (deadlock, disk full, ulimit, etc.).
        # CI pipelines reading `passed` would otherwise treat both the same.
        if results and all(not r.output and r.errors for r in results.values()):
            logging.error(
                "All %d static analysis tool(s) failed without producing output. "
                "This usually indicates an infrastructure problem (resource limits, "
                "missing system dependencies, or sandbox restrictions) rather than "
                "code-quality issues. Run a single tool with --verbose to debug.",
                len(results),
            )
        return results

    @staticmethod
    def _filter_lines_for_paths(
        output: str, allowed_basenames: set[str]
    ) -> tuple[list[str], int]:
        """Keep only output lines that mention at least one allowed basename.

        Most linters emit one diagnostic per line as ``<path>:<line>:<col>: ...``
        so filename-substring matching is a good-enough, tool-agnostic filter.
        We match on basename rather than full path because tools vary on
        whether they print absolute or relative paths.

        Returns the filtered non-empty lines plus the count of dropped lines.
        Empty lines and lines without any path token are dropped (they're
        usually summary banners that get re-emitted by the section header).
        """
        kept: list[str] = []
        dropped = 0
        for ln in output.splitlines():
            if not ln.strip():
                continue
            if any(name in ln for name in allowed_basenames):
                kept.append(ln)
            else:
                dropped += 1
        return kept, dropped

    @staticmethod
    def condense_for_prompt(
        results: dict[str, "StaticAnalysisResult"],
        max_chars: int = 4000,
        max_lines_per_tool: int = 25,
        only_paths: list[str] | None = None,
    ) -> str:
        """Condense linter results into a short block to inject into LLM prompts.

        The goal is to tell the model what's already been caught so it doesn't
        re-report the same findings. We keep the head of each non-passing
        tool's output (where modern linters emit per-issue lines) and cap the
        total size — passing too much linter detail eats into the file token
        budget and rarely helps the LLM more than a representative sample.

        Args:
            results: Mapping from tool name to its StaticAnalysisResult
            max_chars: Hard cap on the returned string length
            max_lines_per_tool: Truncate each tool's output to this many lines
            only_paths: When set, keep only lines mentioning one of these
                file paths' basenames. Used to slice global linter output
                down to the files in a single batch.

        Returns:
            A formatted block ready to splice into batch context, or an empty
            string if there's nothing useful to inject.
        """
        if not results:
            return ""

        problem_tools = [
            (name, res)
            for name, res in results.items()
            if not res.passed or res.issues_count > 0
        ]
        if not problem_tools:
            return ""

        allowed_basenames: set[str] | None = None
        if only_paths is not None:
            from pathlib import PurePath

            allowed_basenames = {PurePath(p).name for p in only_paths}
            allowed_basenames.discard("")

        sections: list[str] = []
        for name, res in problem_tools:
            if allowed_basenames is not None:
                lines, _ = StaticAnalyzer._filter_lines_for_paths(
                    res.output, allowed_basenames
                )
            else:
                lines = [ln for ln in res.output.splitlines() if ln.strip()]
            if not lines:
                continue
            head = lines[:max_lines_per_tool]
            truncated = len(lines) - len(head)
            count_label = (
                f"{len(lines)} matching line(s)"
                if allowed_basenames is not None
                else f"{res.issues_count} issue(s)"
            )
            section = [f"-- {name} ({count_label}) --"]
            section.extend(head)
            if truncated > 0:
                section.append(f"... {truncated} more line(s) elided")
            sections.append("\n".join(section))

        if not sections:
            return ""

        block = "\n\n".join(sections)
        if len(block) > max_chars:
            block = block[:max_chars] + "\n... (linter output truncated)"
        return block

    @staticmethod
    def get_summary(results: dict[str, StaticAnalysisResult]) -> StaticAnalysisSummary:
        """
        Generate summary of static analysis results.

        Args:
            results: Results from run_all()

        Returns:
            StaticAnalysisSummary with aggregated metrics
        """
        total_issues = sum(r.issues_count for r in results.values())
        tools_passed = sum(1 for r in results.values() if r.passed)
        tools_failed = sum(1 for r in results.values() if not r.passed)

        return {
            "tools_run": len(results),
            "tools_passed": tools_passed,
            "tools_failed": tools_failed,
            "total_issues": total_issues,
            "passed": tools_failed == 0,
        }
