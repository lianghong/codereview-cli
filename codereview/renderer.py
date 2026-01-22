"""Rich terminal and Markdown output rendering."""
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from codereview.models import CodeReviewReport, ReviewIssue


class TerminalRenderer:
    """Renders code review results to Rich terminal."""

    SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Info"]

    SEVERITY_COLORS = {
        "Critical": "red",
        "High": "bright_red",
        "Medium": "yellow",
        "Low": "blue",
        "Info": "white",
    }

    SEVERITY_ICONS = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🔵",
        "Info": "⚪",
    }

    def __init__(self):
        """Initialize renderer."""
        self.console = Console()

    def render(self, report: CodeReviewReport):
        """
        Render full report to terminal.

        Args:
            report: CodeReviewReport to render
        """
        self.console.print()
        self._render_header()
        self._render_summary(report)
        self._render_issues(report)
        self._render_recommendations(report)
        self.console.print()

    def _render_header(self):
        """Render header."""
        self.console.print(
            Panel.fit(
                "[bold cyan]Code Review Report[/bold cyan]",
                border_style="cyan"
            )
        )

    def _render_summary(self, report: CodeReviewReport):
        """Render summary section."""
        summary = self._format_summary(report)
        self.console.print(Panel(summary, title="Summary", border_style="green"))

    def _format_summary(self, report: CodeReviewReport) -> str:
        """Format summary text."""
        lines = [
            f"[bold]{report.summary}[/bold]",
            "",
            f"📊 Files analyzed: {report.metrics.get('files', 0)}",
            f"🐛 Total issues: {report.metrics.get('issues', 0)}",
        ]
        return "\n".join(lines)

    def _render_issues(self, report: CodeReviewReport):
        """Render issues grouped by severity."""
        if not report.issues:
            self.console.print("[green]✓ No issues found![/green]\n")
            return

        grouped = self._group_by_severity(report.issues)

        for severity in self.SEVERITY_ORDER:
            if severity not in grouped:
                continue

            issues = grouped[severity]
            color = self._get_severity_color(severity)
            icon = self.SEVERITY_ICONS[severity]

            self.console.print(f"\n{icon} [bold {color}]{severity} ({len(issues)})[/bold {color}]")

            for issue in issues:
                self._render_issue(issue)

    def _render_issue(self, issue: ReviewIssue):
        """Render single issue."""
        color = self._get_severity_color(issue.severity)

        table = Table(show_header=False, border_style=color, box=None)
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Category", issue.category)
        table.add_row("File", f"{issue.file_path}:{issue.line_start}")
        table.add_row("Issue", issue.title)
        table.add_row("Details", issue.description)
        table.add_row("Why", issue.rationale)

        if issue.suggested_code:
            table.add_row("Fix", f"```\n{issue.suggested_code}\n```")

        self.console.print(table)
        self.console.print()

    def _render_recommendations(self, report: CodeReviewReport):
        """Render top recommendations."""
        if not report.recommendations:
            return

        lines = []
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        self.console.print(
            Panel(
                "\n".join(lines),
                title="Top Recommendations",
                border_style="yellow"
            )
        )

    def _group_by_severity(self, issues: List[ReviewIssue]) -> Dict[str, List[ReviewIssue]]:
        """Group issues by severity level."""
        grouped = {}
        for issue in issues:
            if issue.severity not in grouped:
                grouped[issue.severity] = []
            grouped[issue.severity].append(issue)
        return grouped

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level."""
        return self.SEVERITY_COLORS.get(severity, "white")


class MarkdownExporter:
    """Exports code review reports to Markdown format."""

    SEVERITY_ICONS = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🔵",
        "Info": "⚪",
    }

    def export(self, report: CodeReviewReport, output_path: Path | str):
        """
        Export report to Markdown file.

        Args:
            report: CodeReviewReport to export
            output_path: Path to output Markdown file
        """
        output_path = Path(output_path)
        content = self._generate_markdown(report)
        output_path.write_text(content)

    def _generate_markdown(self, report: CodeReviewReport) -> str:
        """Generate Markdown content."""
        sections = [
            self._header(),
            self._summary(report),
            self._metrics(report),
            self._issues(report),
            self._system_design(report),
            self._recommendations(report),
        ]

        return "\n\n".join(sections)

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

        # Separate token metrics from other metrics
        token_keys = {"input_tokens", "output_tokens", "total_tokens"}
        regular_metrics = {k: v for k, v in report.metrics.items() if k not in token_keys}
        token_metrics = {k: v for k, v in report.metrics.items() if k in token_keys}

        # Display regular metrics first
        for key, value in regular_metrics.items():
            if isinstance(value, int):
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value:,}")
            else:
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

        # Add token usage section if available
        if token_metrics:
            lines.append("\n### Token Usage & Cost\n")
            for key, value in sorted(token_metrics.items()):
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value:,}")

            # Calculate and display cost if we have token counts
            if 'input_tokens' in token_metrics and 'output_tokens' in token_metrics:
                input_cost = (token_metrics['input_tokens'] / 1_000_000) * 15.00
                output_cost = (token_metrics['output_tokens'] / 1_000_000) * 75.00
                total_cost = input_cost + output_cost
                lines.append(f"- **Estimated Cost:** ${total_cost:.4f} USD")
                lines.append(f"  - Input cost: ${input_cost:.4f} (${15.00}/M tokens)")
                lines.append(f"  - Output cost: ${output_cost:.4f} (${75.00}/M tokens)")

        return "\n".join(lines)

    def _issues(self, report: CodeReviewReport) -> str:
        """Generate issues section."""
        if not report.issues:
            return "## Issues\n\n✅ No issues found!"

        lines = ["## Issues by Severity\n"]

        # Group by severity
        grouped = {}
        for issue in report.issues:
            if issue.severity not in grouped:
                grouped[issue.severity] = []
            grouped[issue.severity].append(issue)

        # Render each severity group
        for severity in ["Critical", "High", "Medium", "Low", "Info"]:
            if severity not in grouped:
                continue

            icon = self.SEVERITY_ICONS[severity]
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
            lines.append(f"**Suggested Fix:**")
            lines.append(f"```python\n{issue.suggested_code}\n```\n")

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
