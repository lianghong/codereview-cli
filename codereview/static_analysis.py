"""Static analysis integration for Python code quality tools."""
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class StaticAnalysisResult:
    """Results from static analysis tools."""
    tool: str
    passed: bool
    issues_count: int
    output: str
    errors: List[str]


class StaticAnalyzer:
    """Runs static analysis tools on Python code."""

    TOOLS = {
        "ruff": {
            "name": "Ruff",
            "description": "Fast Python linter",
            "command": ["ruff", "check"],
        },
        "mypy": {
            "name": "Mypy",
            "description": "Static type checker",
            "command": ["mypy"],
        },
        "black": {
            "name": "Black",
            "description": "Code formatter",
            "command": ["black", "--check", "--diff"],
        },
        "isort": {
            "name": "isort",
            "description": "Import sorter",
            "command": ["isort", "--check-only", "--diff"],
        },
    }

    def __init__(self, directory: Path):
        """
        Initialize static analyzer.

        Args:
            directory: Directory to analyze
        """
        self.directory = directory
        self.available_tools = self._check_available_tools()

    def _check_available_tools(self) -> List[str]:
        """Check which tools are available."""
        available = []
        for tool_name in self.TOOLS.keys():
            try:
                subprocess.run(
                    [tool_name, "--version"],
                    capture_output=True,
                    timeout=5
                )
                available.append(tool_name)
            except (FileNotFoundError, subprocess.TimeoutExpired):
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
                errors=[f"Unknown tool: {tool_name}"]
            )

        if tool_name not in self.available_tools:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=[f"Tool not installed: {tool_name}"]
            )

        # Validate directory before use (security measure)
        if not self.directory.exists() or not self.directory.is_dir():
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=["Invalid directory path"]
            )

        tool_config = self.TOOLS[tool_name]
        # Use absolute path to avoid relative path issues
        command = tool_config["command"] + [str(self.directory.resolve())]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=self.directory.parent
            )

            # Most tools return non-zero when they find issues
            passed = result.returncode == 0
            output = result.stdout + result.stderr

            # Count issues (rough estimation based on output lines)
            issues_count = 0
            if not passed:
                # Count error/warning lines
                for line in output.split('\n'):
                    if any(indicator in line.lower() for indicator in
                           ['error', 'warning', 'would reformat', 'would be reformatted']):
                        issues_count += 1

            return StaticAnalysisResult(
                tool=tool_name,
                passed=passed,
                issues_count=issues_count,
                output=output.strip(),
                errors=[]
            )

        except subprocess.TimeoutExpired:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=["Tool timed out after 120 seconds"]
            )
        except Exception as e:
            return StaticAnalysisResult(
                tool=tool_name,
                passed=False,
                issues_count=0,
                output="",
                errors=[f"Error running tool: {str(e)}"]
            )

    def run_all(self) -> Dict[str, StaticAnalysisResult]:
        """
        Run all available static analysis tools.

        Returns:
            Dictionary mapping tool names to results
        """
        results = {}
        for tool_name in self.available_tools:
            results[tool_name] = self.run_tool(tool_name)
        return results

    def get_summary(self, results: Dict[str, StaticAnalysisResult]) -> Dict[str, Any]:
        """
        Generate summary of static analysis results.

        Args:
            results: Results from run_all()

        Returns:
            Summary dictionary
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
