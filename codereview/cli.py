"""CLI entry point for code review tool."""
import click
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from codereview.scanner import FileScanner
from codereview.batcher import SmartBatcher
from codereview.analyzer import CodeAnalyzer
from codereview.renderer import TerminalRenderer, MarkdownExporter
from codereview.models import CodeReviewReport, ReviewIssue
from codereview.config import SUPPORTED_MODELS
from codereview.static_analysis import StaticAnalyzer

console = Console()


@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output Markdown report to file'
)
@click.option(
    '--severity', '-s',
    type=click.Choice(['critical', 'high', 'medium', 'low', 'info'], case_sensitive=False),
    default='info',
    help='Minimum severity level to display'
)
@click.option(
    '--exclude', '-e',
    multiple=True,
    help='Additional exclusion patterns'
)
@click.option(
    '--max-files',
    type=int,
    help='Maximum number of files to analyze'
)
@click.option(
    '--max-file-size',
    type=int,
    default=10,
    help='Maximum file size in KB (default: 10)'
)
@click.option(
    '--aws-region',
    type=str,
    help='AWS region for Bedrock'
)
@click.option(
    '--aws-profile',
    type=str,
    help='AWS CLI profile to use'
)
@click.option(
    '--model-id',
    type=click.Choice([
        'global.anthropic.claude-opus-4-5-20251101-v1:0',
        'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'global.anthropic.claude-haiku-4-5-20251001-v1:0'
    ], case_sensitive=False),
    default='global.anthropic.claude-opus-4-5-20251101-v1:0',
    help='Claude model to use (default: Opus 4.5)'
)
@click.option(
    '--static-analysis',
    is_flag=True,
    help='Run static analysis tools (ruff, mypy, black, isort)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed progress'
)
def main(
    directory: Path,
    output: Path | None,
    severity: str,
    exclude: tuple,
    max_files: int | None,
    max_file_size: int,
    aws_region: str | None,
    aws_profile: str | None,
    model_id: str,
    static_analysis: bool,
    verbose: bool
):
    """
    Analyze code in DIRECTORY and generate a comprehensive review report.

    Reviews Python and Go files using Claude models via AWS Bedrock.
    """
    try:
        # Get model information
        model_info = SUPPORTED_MODELS.get(model_id)
        model_name = model_info["name"] if model_info else "Unknown"

        console.print(f"\n[bold cyan]🔍 Code Review Tool[/bold cyan]\n")
        console.print(f"📂 Scanning directory: {directory}")
        console.print(f"🤖 Model: {model_name}\n")

        # Step 1: Scan files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)

            scanner = FileScanner(directory, max_file_size_kb=max_file_size)
            files = scanner.scan()

            if max_files:
                files = files[:max_files]

            progress.update(task, completed=True)

        if not files:
            console.print("[yellow]⚠️  No files found to review[/yellow]")
            return

        console.print(f"✓ Found {len(files)} files to review\n")

        # Step 1.5: Run static analysis if requested
        static_results = None
        if static_analysis:
            console.print("[cyan]Running static analysis tools...[/cyan]\n")
            analyzer_static = StaticAnalyzer(directory)

            if not analyzer_static.available_tools:
                console.print("[yellow]⚠️  No static analysis tools found. Install: pip install ruff mypy black isort[/yellow]\n")
            else:
                console.print(f"[cyan]Available tools: {', '.join(analyzer_static.available_tools)}[/cyan]\n")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Running static analysis...", total=None)
                    static_results = analyzer_static.run_all()
                    progress.update(task, completed=True)

                _render_static_analysis_results(static_results)

        # Step 2: Create batches
        batcher = SmartBatcher()
        batches = batcher.create_batches(files)

        console.print(f"📦 Created {len(batches)} batches\n")

        # Step 3: Analyze batches
        analyzer = CodeAnalyzer(region=aws_region, model_id=model_id)
        all_issues = []
        total_files = len(files)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Analyzing code...",
                total=len(batches)
            )

            for i, batch in enumerate(batches, 1):
                if verbose:
                    console.print(f"  Batch {i}/{len(batches)}: {len(batch.files)} files")

                try:
                    report = analyzer.analyze_batch(batch)
                    all_issues.extend(report.issues)

                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    error_msg = e.response.get('Error', {}).get('Message', '')

                    if error_code == 'AccessDeniedException':
                        console.print(f"[red]✗ AWS Access Denied: {error_msg}[/red]")
                        console.print("[yellow]Check that you have access to AWS Bedrock Claude Opus 4.5[/yellow]")
                    elif error_code in ['ThrottlingException', 'TooManyRequestsException']:
                        console.print(f"[red]✗ Rate limit exceeded on batch {i}[/red]")
                        console.print("[yellow]Consider reducing batch size or waiting before retrying[/yellow]")
                    else:
                        console.print(f"[red]✗ AWS Error on batch {i} ({error_code}): {error_msg}[/red]")

                    if verbose:
                        import traceback
                        console.print(traceback.format_exc())

                except Exception as e:
                    console.print(f"[red]✗ Error analyzing batch {i}: {e}[/red]")
                    if verbose:
                        import traceback
                        console.print(traceback.format_exc())

                progress.update(task, advance=1)

        # Step 4: Create final report
        metrics = {
            "files_analyzed": total_files,
            "total_issues": len(all_issues),
            "critical": sum(1 for i in all_issues if i.severity == "Critical"),
            "high": sum(1 for i in all_issues if i.severity == "High"),
            "medium": sum(1 for i in all_issues if i.severity == "Medium"),
            "low": sum(1 for i in all_issues if i.severity == "Low"),
            "input_tokens": analyzer.total_input_tokens,
            "output_tokens": analyzer.total_output_tokens,
            "total_tokens": analyzer.total_input_tokens + analyzer.total_output_tokens,
            "model_name": model_name,
            "input_price_per_million": model_info["input_price_per_million"] if model_info else 0,
            "output_price_per_million": model_info["output_price_per_million"] if model_info else 0,
        }

        # Add static analysis metrics if available
        if static_results:
            static_summary = StaticAnalyzer(directory).get_summary(static_results)
            metrics["static_analysis_run"] = True
            metrics["static_tools_passed"] = static_summary["tools_passed"]
            metrics["static_tools_failed"] = static_summary["tools_failed"]
            metrics["static_issues_found"] = static_summary["total_issues"]
        else:
            metrics["static_analysis_run"] = False

        final_report = CodeReviewReport(
            summary=f"Analyzed {total_files} files and found {len(all_issues)} issues",
            metrics=metrics,
            issues=all_issues,
            system_design_insights="Analysis complete",
            recommendations=_generate_recommendations(all_issues)
        )

        # Display token usage and cost
        console.print(f"\n[cyan]💰 Token Usage & Cost Estimate:[/cyan]")
        console.print(f"   Input tokens:  {analyzer.total_input_tokens:,}")
        console.print(f"   Output tokens: {analyzer.total_output_tokens:,}")
        console.print(f"   Total tokens:  {analyzer.total_input_tokens + analyzer.total_output_tokens:,}")

        # Calculate cost using model's pricing
        if model_info:
            input_price = model_info["input_price_per_million"]
            output_price = model_info["output_price_per_million"]
            input_cost = (analyzer.total_input_tokens / 1_000_000) * input_price
            output_cost = (analyzer.total_output_tokens / 1_000_000) * output_price
            total_cost = input_cost + output_cost
            console.print(f"   [bold]Estimated cost: ${total_cost:.4f}[/bold]")
        console.print()

        # Step 5: Render results
        renderer = TerminalRenderer()
        renderer.render(final_report)

        # Step 6: Export to Markdown if requested
        if output:
            exporter = MarkdownExporter()
            exporter.export(final_report, output)
            console.print(f"\n[green]✓ Report exported to: {output}[/green]\n")

    except NoCredentialsError:
        console.print("\n[red]✗ AWS credentials not found[/red]\n")
        console.print("[yellow]Please configure AWS credentials:[/yellow]")
        console.print("  1. Run: aws configure")
        console.print("  2. Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        console.print("  3. Or use --aws-profile flag\n")
        raise click.Abort()

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        error_msg = e.response.get('Error', {}).get('Message', '')

        console.print(f"\n[red]✗ AWS Error ({error_code}): {error_msg}[/red]\n")

        if error_code == 'AccessDeniedException':
            console.print("[yellow]Troubleshooting:[/yellow]")
            console.print("  1. Ensure you have access to AWS Bedrock in your region")
            console.print("  2. Check that Claude Opus 4.5 model access is enabled")
            console.print("  3. Verify your IAM permissions include 'bedrock:InvokeModel'\n")
        elif error_code == 'ResourceNotFoundException':
            console.print("[yellow]The Claude Opus 4.5 model may not be available in your region.[/yellow]")
            console.print("Try using --aws-region with a supported region (e.g., us-west-2)\n")

        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise click.Abort()

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]\n")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise click.Abort()


def _render_static_analysis_results(results: dict) -> None:
    """Render static analysis results to terminal."""
    from codereview.static_analysis import StaticAnalyzer

    analyzer = StaticAnalyzer(Path("."))  # Dummy instance for getting tool info
    summary = StaticAnalyzer(Path(".")).get_summary(results)

    # Create summary table
    table = Table(title="Static Analysis Results", show_header=True, header_style="bold cyan")
    table.add_column("Tool", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Issues", justify="right")

    for tool_name, result in results.items():
        tool_config = analyzer.TOOLS.get(tool_name, {})
        tool_display = tool_config.get("name", tool_name)

        if result.passed:
            status = "[green]✓ Passed[/green]"
        elif result.errors:
            status = "[red]✗ Error[/red]"
        else:
            status = "[yellow]⚠ Issues[/yellow]"

        issues_str = str(result.issues_count) if result.issues_count > 0 else "-"

        table.add_row(tool_display, status, issues_str)

    console.print(table)
    console.print()

    # Overall summary
    if summary["passed"]:
        console.print("[green]✓ All static analysis checks passed![/green]\n")
    else:
        console.print(f"[yellow]⚠️  {summary['tools_failed']} tool(s) found issues ({summary['total_issues']} total)[/yellow]\n")

    # Show details for failed tools
    for tool_name, result in results.items():
        if not result.passed and not result.errors and result.output:
            tool_config = analyzer.TOOLS.get(tool_name, {})
            tool_display = tool_config.get("name", tool_name)

            # Limit output to first 20 lines
            output_lines = result.output.split('\n')[:20]
            output_preview = '\n'.join(output_lines)

            if len(result.output.split('\n')) > 20:
                output_preview += f"\n... ({len(result.output.split('\n')) - 20} more lines)"

            console.print(Panel(
                output_preview,
                title=f"[yellow]{tool_display} Output[/yellow]",
                border_style="yellow"
            ))
            console.print()


def _generate_recommendations(issues: list[ReviewIssue]) -> list[str]:
    """Generate top recommendations from issues."""
    critical = [i for i in issues if i.severity == "Critical"]
    high = [i for i in issues if i.severity == "High"]

    recommendations = []

    if critical:
        recommendations.append(f"🚨 Address {len(critical)} critical issues immediately")

    if high:
        recommendations.append(f"⚠️  Fix {len(high)} high-priority issues")

    if len(issues) > 10:
        recommendations.append("📊 Consider refactoring to reduce technical debt")

    return recommendations[:5]


if __name__ == '__main__':
    main()
