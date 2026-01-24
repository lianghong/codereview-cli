"""Static analysis integration for code quality tools across multiple languages."""

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# Maximum number of files to pass to command line tools
# Prevents "Argument list too long" errors on large repos
MAX_FILES_PER_TOOL = 500


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
        # Shell Script tools
        "shellcheck": {
            "name": "ShellCheck",
            "description": "Shell script static analyzer",
            "command": ["shellcheck"],
            "language": "shell",
        },
        # C++ tools
        "clang-tidy": {
            "name": "clang-tidy",
            "description": "C++ static analyzer and linter",
            "command": ["clang-tidy"],
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
            "command": ["checkstyle", "-c", "/google_checks.xml"],
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
            # Resolve both paths to handle symlinks
            resolved_file = file_path.resolve()
            resolved_dir = self.directory.resolve()

            # Check if file is within directory (prevents path traversal)
            resolved_file.relative_to(resolved_dir)
            return True
        except ValueError:
            # relative_to raises ValueError if path is not relative to directory
            logging.warning(
                f"File path {file_path} is outside analysis directory, skipping"
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

    def _check_available_tools(self) -> list[str]:
        """Check which tools are available."""
        available = []
        for tool_name, config in self.TOOLS.items():
            try:
                # Use custom version command if specified, otherwise default to --version
                version_cmd = config.get("version_command", [tool_name, "--version"])
                result = subprocess.run(
                    version_cmd, capture_output=True, timeout=5, shell=False
                )
                # Only mark as available if the command succeeds
                if result.returncode == 0:
                    available.append(tool_name)
            except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):  # fmt: skip
                pass
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
        language = tool_config.get("language", "python")
        dir_path = str(self.directory)  # Already resolved in __init__

        # Build command based on tool type
        # Each tool has different CLI conventions for specifying targets:
        base_command = list(tool_config["command"])
        if tool_name == "go-vet":
            # go vet: Uses Go's "./..." pattern to recursively check all packages
            # in the current module. Must run from within the Go module directory.
            # Check if there are any Go files first
            go_files = list(self.directory.rglob("*.go"))
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
            go_files = list(self.directory.rglob("*.go"))
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
        elif tool_name == "shellcheck":
            # shellcheck: Does NOT support directory scanning.
            # We must explicitly pass each shell script file as an argument.
            shell_files = list(self.directory.rglob("*.sh")) + list(
                self.directory.rglob("*.bash")
            )
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
                    f"shellcheck: Limiting to {MAX_FILES_PER_TOOL} of "
                    f"{len(shell_files)} files to avoid command line length limits"
                )
                shell_files = shell_files[:MAX_FILES_PER_TOOL]
            command = base_command + [str(f) for f in shell_files]
            cwd = self.directory
        elif tool_name in ("clang-tidy", "clang-format"):
            # C++ tools that need explicit file list
            cpp_files = (
                list(self.directory.rglob("*.cpp"))
                + list(self.directory.rglob("*.cc"))
                + list(self.directory.rglob("*.cxx"))
                + list(self.directory.rglob("*.h"))
                + list(self.directory.rglob("*.hpp"))
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
            java_files = list(self.directory.rglob("*.java"))
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
            # We collect files to check if any exist, but ESLint uses directory path
            js_files = (
                list(self.directory.rglob("*.js"))
                + list(self.directory.rglob("*.jsx"))
                + list(self.directory.rglob("*.ts"))
                + list(self.directory.rglob("*.tsx"))
            )
            if not js_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No JavaScript/TypeScript files found",
                    errors=[],
                )
            if len(js_files) > MAX_FILES_PER_TOOL:
                logging.warning(
                    f"eslint: Found {len(js_files)} JS/TS files, "
                    "ESLint may take longer on large codebases"
                )
            command = base_command + [dir_path]
            cwd = self.directory
        elif tool_name == "prettier":
            # Prettier needs files to format - check for JS/TS/CSS/HTML files
            prettier_files = (
                list(self.directory.rglob("*.js"))
                + list(self.directory.rglob("*.jsx"))
                + list(self.directory.rglob("*.ts"))
                + list(self.directory.rglob("*.tsx"))
                + list(self.directory.rglob("*.css"))
                + list(self.directory.rglob("*.html"))
                + list(self.directory.rglob("*.json"))
                + list(self.directory.rglob("*.md"))
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
                ts_files = list(self.directory.rglob("*.ts")) + list(
                    self.directory.rglob("*.tsx")
                )
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
        elif tool_name == "gofmt":
            # gofmt accepts a directory path but needs Go files to exist
            go_files = list(self.directory.rglob("*.go"))
            if not go_files:
                return StaticAnalysisResult(
                    tool=tool_name,
                    passed=True,
                    issues_count=0,
                    output="No Go files found",
                    errors=[],
                )
            command = base_command + [dir_path]
            cwd = self.directory.parent
        else:
            # Default (ruff, mypy, black, isort, vulture):
            # These tools accept a directory path as the final argument.
            command = base_command + [dir_path]
            cwd = self.directory.parent

        try:
            result = subprocess.run(
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
                else:
                    # Count error/warning lines for all tools
                    indicators = [
                        "error",
                        "warning",
                        "would reformat",
                        "would be reformatted",
                        "unused",
                    ]
                    # Add language-specific indicators
                    if language == "go":
                        indicators.extend(["typecheck", "deadcode", "ineffassign"])
                    elif language == "shell":
                        # ShellCheck uses SC codes
                        indicators.extend(["sc", "note", "info"])
                    elif language == "cpp":
                        # C++ tools indicators
                        indicators.extend(
                            ["style", "performance", "portability", "cppcheck"]
                        )
                    elif language == "java":
                        # Checkstyle indicators
                        indicators.extend(["audit", "violation", "checkstyle"])
                    elif language in ("javascript", "typescript"):
                        # ESLint/Prettier/TSC indicators
                        indicators.extend(
                            ["eslint", "prettier", "ts", "problems found"]
                        )

                    for line in output.split("\n"):
                        if any(indicator in line.lower() for indicator in indicators):
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
                    # Handle unexpected errors from thread execution
                    results[tool_name] = StaticAnalysisResult(
                        tool=tool_name,
                        passed=False,
                        issues_count=0,
                        output="",
                        errors=[f"Execution error: {str(e)}"],
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
