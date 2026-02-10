"""Rich terminal and Markdown output rendering."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codereview.models import CodeReviewReport, ReviewIssue
from codereview.providers.base import ValidationResult
from codereview.static_analysis import StaticAnalysisResult, StaticAnalyzer

# Shared constants for severity display
SEVERITY_ORDER: list[str] = ["Critical", "High", "Medium", "Low", "Info"]

SEVERITY_ICONS: dict[str, str] = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🔵",
    "Info": "⚪",
}


class TerminalRenderer:
    """Renders code review results to Rich terminal."""

    SEVERITY_COLORS = {
        "Critical": "red",
        "High": "bright_red",
        "Medium": "yellow",
        "Low": "blue",
        "Info": "white",
    }

    def __init__(self, console: Console | None = None):
        """Initialize renderer.

        Args:
            console: Rich console for output (creates new one if None)
        """
        self.console = console or Console()

    def render(self, report: CodeReviewReport, min_severity: str = "info") -> None:
        """
        Render full report to terminal.

        Args:
            report: CodeReviewReport to render
            min_severity: Minimum severity level to display (default: "info")
        """
        self.console.print()
        self._render_header()
        self._render_summary(report)
        self._render_issues(report, min_severity)
        self._render_recommendations(report)
        self._render_improvement_suggestions(report)
        self.console.print()

    def _render_header(self) -> None:
        """Render report header panel."""
        self.console.print(
            Panel.fit("[bold cyan]Code Review Report[/bold cyan]", border_style="cyan")
        )

    def _render_summary(self, report: CodeReviewReport) -> None:
        """Render summary section panel."""
        summary = self._strip_variation_selectors(self._format_summary(report))
        self.console.print(Panel(summary, title="Summary", border_style="green"))

    @staticmethod
    def _metrics_to_dict(report: CodeReviewReport) -> dict[str, Any]:
        """Convert report metrics to dictionary."""
        if hasattr(report.metrics, "model_dump"):
            return report.metrics.model_dump(exclude_none=True)
        return report.metrics  # type: ignore[return-value]

    def _format_summary(self, report: CodeReviewReport) -> str:
        """Format summary text."""
        metrics_dict = self._metrics_to_dict(report)
        lines = [
            f"[bold]{report.summary}[/bold]",
            "",
            f"📊 Files analyzed: {metrics_dict.get('files_analyzed', 0)}",
            f"📝 Total lines of code: {metrics_dict.get('total_lines', 0):,}",
            f"🐛 Total issues: {metrics_dict.get('total_issues', 0)}",
        ]
        return "\n".join(lines)

    def _render_issues(
        self, report: CodeReviewReport, min_severity: str = "info"
    ) -> None:
        """Render issues grouped by severity, filtered by minimum severity."""
        if not report.issues:
            self.console.print("[green]✓ No issues found![/green]\n")
            return

        # Get minimum severity index (case-insensitive)
        min_severity_title = min_severity.title()
        if min_severity_title not in SEVERITY_ORDER:
            min_severity_title = "Info"
        min_index = SEVERITY_ORDER.index(min_severity_title)

        # Filter issues by minimum severity
        filtered_issues = [
            issue
            for issue in report.issues
            if issue.severity in SEVERITY_ORDER
            and SEVERITY_ORDER.index(issue.severity) <= min_index
        ]

        if not filtered_issues:
            self.console.print(
                f"[green]✓ No issues at {min_severity_title} severity or above![/green]\n"
            )
            return

        grouped = self._group_by_severity(filtered_issues)

        for severity in SEVERITY_ORDER:
            if severity not in grouped:
                continue

            issues = grouped[severity]
            color = self._get_severity_color(severity)
            icon = SEVERITY_ICONS[severity]

            self.console.print(
                f"\n{icon} [bold {color}]{severity} ({len(issues)})[/bold {color}]"
            )

            for issue in issues:
                self._render_issue(issue)

    def _render_issue(self, issue: ReviewIssue) -> None:
        """Render single issue as a formatted table with prominent labels."""
        color = self._get_severity_color(issue.severity)

        table = Table(
            show_header=False,
            border_style=color,
            box=None,
            padding=(0, 1),
        )
        table.add_column("Key", style="bold cyan", width=10, justify="right")
        table.add_column("Value", overflow="fold")

        table.add_row("[bold cyan]Category[/]", issue.category)
        table.add_row(
            "[bold cyan]File[/]", f"[underline]{issue.file_path}:{issue.line_start}[/]"
        )
        table.add_row("[bold cyan]Issue[/]", f"[bold]{issue.title}[/]")
        table.add_row("[bold cyan]Details[/]", issue.description)
        table.add_row("[bold cyan]Why[/]", issue.rationale)

        if issue.suggested_code:
            table.add_row("[bold cyan]Fix[/]", f"```\n{issue.suggested_code}\n```")

        self.console.print(table)
        self.console.print()

    @staticmethod
    def _strip_variation_selectors(text: str) -> str:
        """Strip Unicode variation selectors that cause Rich panel alignment issues.

        Emoji variation selector U+FE0F causes a mismatch between Rich's cell
        width calculation and actual terminal rendering, resulting in misaligned
        right borders in panels.
        """
        return text.replace("\ufe0f", "").replace("\ufe0e", "")

    def _render_recommendations(self, report: CodeReviewReport) -> None:
        """Render top recommendations panel."""
        if not report.recommendations:
            return

        lines = []
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        content = self._strip_variation_selectors("\n".join(lines))
        self.console.print(
            Panel(content, title="Top Recommendations", border_style="yellow")
        )

    def _render_improvement_suggestions(self, report: CodeReviewReport) -> None:
        """Render improvement suggestions panel without left/right borders for easier copy."""
        if not report.improvement_suggestions:
            return

        lines = []
        for i, suggestion in enumerate(report.improvement_suggestions, 1):
            lines.append(f"{i}. {suggestion}")

        content = self._strip_variation_selectors("\n".join(lines))
        self.console.print(
            Panel(
                content,
                title="💡 Improvement Suggestions",
                border_style="cyan",
                box=box.HORIZONTALS,
            )
        )

    def _group_by_severity(
        self, issues: list[ReviewIssue]
    ) -> dict[str, list[ReviewIssue]]:
        """Group issues by severity level."""
        grouped: dict[str, list[ReviewIssue]] = defaultdict(list)
        for issue in issues:
            grouped[issue.severity].append(issue)
        return grouped

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level."""
        return self.SEVERITY_COLORS.get(severity, "white")


class StaticAnalysisRenderer:
    """Renders static analysis results to Rich terminal."""

    def __init__(self, console: Console | None = None):
        """Initialize renderer.

        Args:
            console: Rich console for output (creates new one if None)
        """
        self.console = console or Console()

    def render(self, results: dict[str, StaticAnalysisResult]) -> None:
        """Render static analysis results to terminal.

        Args:
            results: Dictionary mapping tool names to StaticAnalysisResult
        """
        summary = StaticAnalyzer.get_summary(results)

        # Create summary table
        table = Table(
            title="Static Analysis Results",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Tool", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Issues", justify="right")

        for tool_name, result in results.items():
            tool_config = StaticAnalyzer.TOOLS.get(tool_name, {})
            tool_display = str(tool_config.get("name", tool_name))

            if result.passed:
                status = "[green]✓ Passed[/green]"
            elif result.errors:
                status = "[red]✗ Error[/red]"
            else:
                status = "[yellow]⚠ Issues[/yellow]"

            issues_str = str(result.issues_count) if result.issues_count > 0 else "-"

            table.add_row(tool_display, status, issues_str)

        try:
            self.console.print(table)
            self.console.print()

            # Overall summary
            if summary["passed"]:
                self.console.print(
                    "[green]✓ All static analysis checks passed![/green]\n"
                )
            else:
                self.console.print(
                    f"[yellow]⚠️  {summary['tools_failed']} tool(s) found issues "
                    f"({summary['total_issues']} total)[/yellow]\n"
                )
        except OSError:
            # Handle terminal I/O errors - fall back to ASCII-safe output
            print("\nStatic Analysis Results:")
            for tool_name, result in results.items():
                status = "PASS" if result.passed else "FAIL"
                print(f"  [{status}] {tool_name}: {result.issues_count} issues")
            if summary["passed"]:
                print("\nAll static analysis checks passed!")
            else:
                print(
                    f"\n{summary['tools_failed']} tool(s) found issues "
                    f"({summary['total_issues']} total)"
                )

        # Show details for all tools with output
        self._render_tool_outputs(results)

    def _render_tool_outputs(self, results: dict[str, StaticAnalysisResult]) -> None:
        """Render output details for all tools that have output.

        Args:
            results: Dictionary mapping tool names to StaticAnalysisResult
        """
        for tool_name, result in results.items():
            # Show output for any tool that has output (not just failed ones)
            if result.output:
                tool_config = StaticAnalyzer.TOOLS.get(tool_name, {})
                tool_display = str(tool_config.get("name", tool_name))

                # Determine style based on result status
                if result.passed:
                    title_style = "green"
                    border_style = "green"
                elif result.errors:
                    title_style = "red"
                    border_style = "red"
                else:
                    title_style = "yellow"
                    border_style = "yellow"

                # Show full output for failed/error tools, limited for passed tools
                if result.passed:
                    # Limit output to first 30 lines for passed tools
                    max_lines = 30
                    output_lines = result.output.split("\n")
                    total_lines = len(output_lines)
                    output_preview = "\n".join(output_lines[:max_lines])

                    if total_lines > max_lines:
                        output_preview += (
                            f"\n... ({total_lines - max_lines} more lines)"
                        )
                else:
                    # Show full output for failed/warning tools
                    output_preview = result.output

                    # Prepend error information if available
                    if result.errors:
                        error_msg = f"[red bold]Errors:[/red bold]\n{result.errors}\n\n"
                        output_preview = error_msg + output_preview

                try:
                    self.console.print(
                        Panel(
                            output_preview,
                            title=f"[{title_style}]{tool_display} Output[/{title_style}]",
                            border_style=border_style,
                        )
                    )
                except OSError:
                    # Handle terminal I/O errors (e.g., write blocking)
                    # Fall back to simple print
                    print(f"\n{tool_display} Output:")
                    print(output_preview)


class ValidationRenderer:
    """Renders credential validation results to Rich terminal."""

    def __init__(self, console: Console | None = None):
        """Initialize renderer.

        Args:
            console: Rich console for output (creates new one if None)
        """
        self.console = console or Console()

    def render(self, validation: ValidationResult) -> None:
        """Render validation result to terminal.

        Args:
            validation: ValidationResult from provider.validate_credentials()
        """
        # Create validation table
        table = Table(
            title=f"{validation.provider} Pre-flight Checks",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for name, passed, message in validation.checks:
            status = "[green]✓ Pass[/green]" if passed else "[red]✗ Fail[/red]"
            table.add_row(name, status, message or "-")

        self.console.print(table)

        # Show warnings
        if validation.warnings:
            self.console.print()
            for warning in validation.warnings:
                self.console.print(f"[yellow]⚠️  {warning}[/yellow]")

        # Show suggestions if validation failed
        if not validation.valid and validation.suggestions:
            self.console.print()
            self.console.print("[bold]💡 Suggestions:[/bold]")
            for suggestion in validation.suggestions:
                self.console.print(f"   • {suggestion}")

        # Overall status
        self.console.print()
        if validation.valid:
            self.console.print("[green]✓ All pre-flight checks passed[/green]")
        else:
            self.console.print("[red]✗ Pre-flight validation failed[/red]")
            self.console.print(
                "[yellow]Fix the issues above before running the actual analysis[/yellow]"
            )


class MarkdownExporter:
    """Exports code review reports to Markdown format."""

    # File extension to language mapping for code blocks
    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".go": "go",
        ".sh": "bash",
        ".bash": "bash",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".rs": "rust",
        ".rb": "ruby",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
    }

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        suffix = PurePath(file_path).suffix
        return self.LANGUAGE_EXTENSIONS.get(suffix, "text")

    @staticmethod
    def _metrics_to_dict(report: CodeReviewReport) -> dict[str, Any]:
        """Convert report metrics to dictionary."""
        if hasattr(report.metrics, "model_dump"):
            return report.metrics.model_dump(exclude_none=True)
        return report.metrics  # type: ignore[return-value]

    def export(
        self,
        report: CodeReviewReport,
        output_path: Path | str,
        skipped_files: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Export report to Markdown file.

        Args:
            report: CodeReviewReport to export
            output_path: Path to output Markdown file
            skipped_files: Optional list of (file_path, reason) tuples for files
                that were skipped during scanning or analysis

        Raises:
            RuntimeError: If file write fails
        """
        output_path = Path(output_path)
        content = self._generate_markdown(report, skipped_files)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        except OSError as e:
            raise RuntimeError(f"Failed to write report to {output_path}: {e}") from e

    def _generate_markdown(
        self,
        report: CodeReviewReport,
        skipped_files: list[tuple[str, str]] | None = None,
    ) -> str:
        """Generate Markdown content."""
        sections = [
            self._header(),
            self._summary(report),
            self._metrics(report),
        ]

        metrics_dict = self._metrics_to_dict(report)

        # Add static analysis section if it was run
        if metrics_dict.get("static_analysis_run"):
            sections.append(self._static_analysis(report))

        sections.extend(
            [
                self._issues(report),
                self._system_design(report),
                self._recommendations(report),
                self._improvement_suggestions(report),
            ]
        )

        # Add skipped files section if any files were skipped
        if skipped_files:
            sections.append(self._skipped_files(skipped_files))

        return "\n\n".join(sections)

    def _skipped_files(self, skipped_files: list[tuple[str, str]]) -> str:
        """Generate skipped files section.

        Args:
            skipped_files: List of (file_path, reason) tuples

        Returns:
            Markdown formatted skipped files section
        """
        lines = [
            "## Skipped Files",
            "",
            f"**{len(skipped_files)} file(s)** were not included in this review:",
            "",
        ]

        # Show up to 20 files, then summarize the rest
        max_display = 20
        for file_path, reason in skipped_files[:max_display]:
            lines.append(f"- `{file_path}`: {reason}")

        if len(skipped_files) > max_display:
            remaining = len(skipped_files) - max_display
            lines.append(f"- ... and {remaining} more file(s)")

        return "\n".join(lines)

    def _header(self) -> str:
        """Generate header section."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"# Code Review Report\n\n**Generated:** {now}"

    def _summary(self, report: CodeReviewReport) -> str:
        """Generate summary section."""
        return f"## Executive Summary\n\n{report.summary}"

    def _metrics(self, report: CodeReviewReport) -> str:
        """Generate metrics section."""
        lines = ["## Metrics\n"]

        metrics_dict = self._metrics_to_dict(report)

        # Separate token metrics from other metrics
        token_keys = {"input_tokens", "output_tokens", "total_tokens"}
        regular_metrics = {k: v for k, v in metrics_dict.items() if k not in token_keys}
        token_metrics = {k: v for k, v in metrics_dict.items() if k in token_keys}

        # Display regular metrics first
        for key, value in regular_metrics.items():
            if isinstance(value, int):
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value:,}")
            else:
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

        # Add token usage section if available
        if token_metrics:
            lines.append("\n### Token Usage & Cost\n")

            # Get pricing information from metrics (no hardcoded defaults)
            input_price = metrics_dict.get("input_price_per_million")
            output_price = metrics_dict.get("output_price_per_million")
            model_name = metrics_dict.get("model_name")

            # Display model name if available
            if model_name:
                lines.append(f"**Model:** {model_name}\n")

            for key, value in sorted(token_metrics.items()):
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value:,}")

            # Calculate and display cost only if we have both token counts AND pricing info
            if (
                "input_tokens" in token_metrics
                and "output_tokens" in token_metrics
                and input_price is not None
                and output_price is not None
            ):
                input_cost = (token_metrics["input_tokens"] / 1_000_000) * input_price
                output_cost = (
                    token_metrics["output_tokens"] / 1_000_000
                ) * output_price
                total_cost = input_cost + output_cost
                lines.append(f"- **Estimated Cost:** ${total_cost:.4f} USD")
                lines.append(
                    f"  - Input cost: ${input_cost:.4f} (${input_price:.2f}/M tokens)"
                )
                lines.append(
                    f"  - Output cost: ${output_cost:.4f} (${output_price:.2f}/M tokens)"
                )

        return "\n".join(lines)

    def _static_analysis(self, report: CodeReviewReport) -> str:
        """Generate static analysis section."""
        metrics_dict = self._metrics_to_dict(report)

        if not metrics_dict.get("static_analysis_run"):
            return ""

        lines = ["## Static Analysis\n"]

        tools_passed = metrics_dict.get("static_tools_passed", 0)
        tools_failed = metrics_dict.get("static_tools_failed", 0)
        issues_found = metrics_dict.get("static_issues_found", 0)

        if tools_failed == 0:
            lines.append("✅ All static analysis tools passed!\n")
        else:
            lines.append(f"⚠️ {tools_failed} tool(s) found {issues_found} issue(s)\n")

        lines.append("| Tool | Status | Issues |")
        lines.append("|------|--------|--------|")

        # We don't have individual tool results in metrics, so just show summary
        lines.append(
            f"| All Tools | {tools_passed} passed, {tools_failed} failed | {issues_found} |"
        )

        lines.append("\n**Tools run:** ruff, mypy, black, isort (when available)")
        lines.append(
            "\n*Run with `--static-analysis` flag to see detailed output in terminal.*"
        )

        return "\n".join(lines)

    def _issues(self, report: CodeReviewReport) -> str:
        """Generate issues section."""
        if not report.issues:
            return "## Issues\n\n✅ No issues found!"

        lines = ["## Issues by Severity\n"]

        # Group by severity
        grouped: dict[str, list[ReviewIssue]] = defaultdict(list)
        for issue in report.issues:
            grouped[issue.severity].append(issue)

        # Render each severity group
        for severity in SEVERITY_ORDER:
            if severity not in grouped:
                continue

            icon = SEVERITY_ICONS[severity]
            issues = grouped[severity]

            lines.append(f"### {icon} {severity} ({len(issues)})\n")

            for issue in issues:
                lines.append(self._format_issue(issue))

        return "\n".join(lines)

    def _format_issue(self, issue: ReviewIssue) -> str:
        """Format single issue in Markdown."""
        lines = [
            f"#### [{issue.category}] {issue.title}",
            f"**File:** `{issue.file_path}:{issue.line_start}`",
            f"**Severity:** {issue.severity}\n",
            f"**Description:**\n{issue.description}\n",
            f"**Rationale:**\n{issue.rationale}\n",
        ]

        if issue.suggested_code:
            lang = self._detect_language(issue.file_path)
            lines.append("**Suggested Fix:**")
            lines.append(f"```{lang}\n{issue.suggested_code}\n```\n")

        if issue.references:
            lines.append("**References:**")
            for ref in issue.references:
                lines.append(f"- {ref}")

        lines.append("\n---\n")

        return "\n".join(lines)

    def _system_design(self, report: CodeReviewReport) -> str:
        """Generate system design section."""
        return f"## System Design Insights\n\n{report.system_design_insights}"

    def _recommendations(self, report: CodeReviewReport) -> str:
        """Generate recommendations section."""
        if not report.recommendations:
            return ""

        lines = ["## Top Recommendations\n"]

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def _improvement_suggestions(self, report: CodeReviewReport) -> str:
        """Generate improvement suggestions section."""
        if not report.improvement_suggestions:
            return ""

        lines = ["## 💡 Improvement Suggestions\n"]
        lines.append(
            "*These are constructive enhancement ideas beyond fixing issues:*\n"
        )

        for i, suggestion in enumerate(report.improvement_suggestions, 1):
            lines.append(f"{i}. {suggestion}")

        return "\n".join(lines)
