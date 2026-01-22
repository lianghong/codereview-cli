"""Rich terminal and Markdown output rendering."""
from typing import Dict, List
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
