"""Static analysis integration for code quality tools across multiple languages."""

import json
import logging
import subprocess  # nosec B404 - required for running static analysis tools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# Maximum number of files to pass to command line tools
# Prevents "Argument list too long" errors on large repos
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

    def __init__(self, directory: Path):
        """
        Initialize static analyzer.

        Args:
            directory: Directory to analyze
        """
        # Resolve path immediately to handle symlinks and normalize
        self.directory = directory.resolve()
        self.available_tools = self._check_available_tools()

    def _validate_directory(self) -> tuple[bool, str]:
        """
        Validate directory is safe to use.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.directory.exists():
            return False, "Directory does not exist"

        if not self.directory.is_dir():
            return False, "Path is not a directory"

        # Check for null bytes or other suspicious characters in path
        dir_str = str(self.directory)
        if "\x00" in dir_str:
            return False, "Invalid characters in path"

        return True, ""

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

        Args:
            pattern: Glob pattern (e.g., "*.go", "*.py")

        Returns:
            List of matching paths, or empty list if scanning fails
        """
        try:
            return list(self.directory.rglob(pattern))
        except OSError as e:
            logging.warning("Error scanning for %s files: %s", pattern, e)
            return []

    def _check_available_tools(self) -> list[str]:
        """Check which static analysis tools are installed and available.

        Runs version check for each configured tool. Tools that pass version
        check or timeout (may still work with longer timeout) are considered
        available. Tools not found or lacking permissions are excluded.

        Returns:
            List of available tool names from TOOLS configuration.
        """
        available = []
        for tool_name, config in self.TOOLS.items():
            try:
                # Use custom version command if specified, otherwise default to --version
                version_cmd = config.get("version_command", [tool_name, "--version"])
                result = subprocess.run(  # nosec B603 - commands from hardcoded config
                    version_cmd, capture_output=True, timeout=5, shell=False
                )
                # Only mark as available if the command succeeds
                if result.returncode == 0:
                    available.append(tool_name)
            except FileNotFoundError:
                logging.debug("Tool %s not installed", tool_name)
            except subprocess.TimeoutExpired:
                # Version check timed out, but tool may still work with longer timeout
                logging.debug(
                    "Tool %s version check timed out (assuming available)", tool_name
                )
                available.append(tool_name)
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

        # Validate directory before use (security measure)
        is_valid, error_msg = self._validate_directory()
        if not is_valid:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=[f"Invalid directory: {error_msg}"],
            )

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
                    f"{tool_name}: Limiting to {MAX_FILES_PER_TOOL} of "
                    f"{len(shell_files)} files to avoid command line length limits"
                )
                shell_files = shell_files[:MAX_FILES_PER_TOOL]
            command = base_command + [str(f) for f in shell_files]
            cwd = self.directory
        elif tool_name in ("clang-tidy", "clang-format"):
            # C++ tools that need explicit file list
            cpp_files = (
                self._safe_rglob("*.cpp")
                + self._safe_rglob("*.cc")
                + self._safe_rglob("*.cxx")
                + self._safe_rglob("*.h")
                + self._safe_rglob("*.hpp")
            )
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
                    f"{tool_name}: Limiting to {MAX_FILES_PER_TOOL} of "
                    f"{len(cpp_files)} files to avoid command line length limits"
                )
                cpp_files = cpp_files[:MAX_FILES_PER_TOOL]
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
                    f"checkstyle: Limiting to {MAX_FILES_PER_TOOL} of "
                    f"{len(java_files)} files to avoid command line length limits"
                )
                java_files = java_files[:MAX_FILES_PER_TOOL]
            command = base_command + [str(f) for f in java_files]
            cwd = self.directory
        elif tool_name == "eslint":
            # ESLint accepts directory, looks for JS/TS files
            # Use next() on generator to stop at first match instead of collecting all files
            has_js_files = False
            for ext in ("*.js", "*.jsx", "*.ts", "*.tsx"):
                try:
                    next(self.directory.rglob(ext))
                    has_js_files = True
                    break
                except StopIteration, OSError:
                    continue
            if not has_js_files:
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
            # Prettier needs files to format - check for JS/TS/CSS/HTML files
            prettier_files = (
                self._safe_rglob("*.js")
                + self._safe_rglob("*.jsx")
                + self._safe_rglob("*.ts")
                + self._safe_rglob("*.tsx")
                + self._safe_rglob("*.css")
                + self._safe_rglob("*.html")
                + self._safe_rglob("*.json")
                + self._safe_rglob("*.md")
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
                    f"prettier: Limiting to {MAX_FILES_PER_TOOL} of "
                    f"{len(prettier_files)} files to avoid command line length limits"
                )
                prettier_files = prettier_files[:MAX_FILES_PER_TOOL]
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
                        f"tsc: Limiting to {MAX_FILES_PER_TOOL} of "
                        f"{len(ts_files)} files to avoid command line length limits"
                    )
                    ts_files = ts_files[:MAX_FILES_PER_TOOL]
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
            go_files = self._safe_rglob("*.go")
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

        try:
            result = subprocess.run(  # nosec B603 - commands from hardcoded config
                command,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
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
                    # Count error/warning lines using base + language-specific indicators
                    indicators = list(BASE_INDICATORS)
                    indicators.extend(LANGUAGE_INDICATORS.get(language, []))

                    for line in output.split("\n"):
                        line_lower = line.lower()
                        if any(indicator in line_lower for indicator in indicators):
                            issues_count += 1

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
                errors=["Tool timed out after 120 seconds"],
            )
        except (OSError, subprocess.SubprocessError) as e:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=[f"Error running tool: {str(e)}"],
            )

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
            # npm audit v2+ format: {"vulnerabilities": {...}} with per-package entries
            if "vulnerabilities" in data and isinstance(data["vulnerabilities"], dict):
                return len(data["vulnerabilities"])
            # npm audit v1 format: {"metadata": {"vulnerabilities": {"low": N, ...}}}
            metadata = data.get("metadata", {})
            vuln_counts = metadata.get("vulnerabilities", {})
            if isinstance(vuln_counts, dict):
                return sum(
                    v for v in vuln_counts.values() if isinstance(v, int) and v > 0
                )
        except json.JSONDecodeError, TypeError, AttributeError:
            # Fallback: count non-empty lines as a rough estimate
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
        return results

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
