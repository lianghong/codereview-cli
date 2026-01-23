"""CLI entry point for code review tool."""

import os
import traceback
from pathlib import Path

import click
from botocore.exceptions import ClientError, NoCredentialsError
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from codereview.analyzer import CodeAnalyzer
from codereview.batcher import FileBatcher
from codereview.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    MAX_FILE_SIZE_KB,
    MODEL_ALIASES,
    SUPPORTED_MODELS,
    resolve_model_id,
)
from codereview.models import CodeReviewReport, ReviewIssue
from codereview.renderer import MarkdownExporter, TerminalRenderer
from codereview.scanner import FileScanner
from codereview.static_analysis import StaticAnalyzer

console = Console()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output Markdown report to file",
)
@click.option(
    "--severity",
    "-s",
    type=click.Choice(
        ["critical", "high", "medium", "low", "info"], case_sensitive=False
    ),
    default="info",
    help="Minimum severity level to display",
)
@click.option("--exclude", "-e", multiple=True, help="Additional exclusion patterns")
@click.option("--max-files", type=int, help="Maximum number of files to analyze")
@click.option(
    "--max-file-size",
    type=int,
    default=MAX_FILE_SIZE_KB,
    help=f"Maximum file size in KB (default: {MAX_FILE_SIZE_KB})",
)
@click.option("--aws-region", type=str, help="AWS region for Bedrock")
@click.option("--aws-profile", type=str, help="AWS CLI profile to use")
@click.option(
    "--model",
    "-m",
    "model_name",
    type=click.Choice(list(MODEL_ALIASES.keys()), case_sensitive=False),
    default="opus",
    help="Model to use: opus, sonnet, haiku, minimax, mistral, kimi, qwen (default: opus)",
)
@click.option(
    "--static-analysis",
    is_flag=True,
    help="Run static analysis tools for all supported languages",
)
@click.option(
    "--temperature",
    type=float,
    default=0.1,
    help="Temperature for model inference (default: 0.1)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show files and estimated cost without making API calls",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@click.pass_context
def main(
    ctx: click.Context,
    directory: Path | None,
    output: Path | None,
    severity: str,
    exclude: tuple[str, ...],
    max_files: int | None,
    max_file_size: int,
    aws_region: str | None,
    aws_profile: str | None,
    model_name: str,
    static_analysis: bool,
    temperature: float,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Analyze code in DIRECTORY and generate a comprehensive review report.

    Reviews Python, Go, Shell, C++, Java, JavaScript, and TypeScript files using
    LLM models via AWS Bedrock.
    """
    # Resolve model name to full model ID
    model_id = resolve_model_id(model_name)
    # Show help if no directory provided
    if directory is None:
        click.echo(ctx.get_help())
        ctx.exit(0)

    try:
        # Set AWS profile if provided
        if aws_profile:
            os.environ["AWS_PROFILE"] = aws_profile

        # Get model information
        model_info = SUPPORTED_MODELS.get(model_id)
        model_name = model_info["name"] if model_info else "Unknown"

        console.print("\n[bold cyan]🔍 Code Review Tool[/bold cyan]\n")
        console.print(f"📂 Scanning directory: {directory}")
        console.print(f"🤖 Model: {model_name} (AWS Bedrock)\n")

        # Step 1: Scan files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)

            # Combine default exclusions with user-provided patterns
            exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS) + list(exclude)
            scanner = FileScanner(
                directory,
                exclude_patterns=exclude_patterns,
                max_file_size_kb=max_file_size,
            )
            files = scanner.scan()

            if max_files:
                files = files[:max_files]

            progress.update(task, completed=True)

        if not files:
            console.print("[yellow]⚠️  No files found to review[/yellow]")
            return

        console.print(f"✓ Found {len(files)} files to review")

        # Report files skipped during scanning (e.g., too large)
        if scanner.skipped_files:
            console.print(
                f"[yellow]⚠️  {len(scanner.skipped_files)} file(s) skipped during scan:[/yellow]"
            )
            for skipped_path, reason in scanner.skipped_files[:5]:
                console.print(f"   • {skipped_path.name}: {reason}")
            if len(scanner.skipped_files) > 5:
                console.print(f"   ... and {len(scanner.skipped_files) - 5} more")
        console.print()

        # Step 1.5: Run static analysis if requested
        static_results = None
        if static_analysis:
            console.print("[cyan]Running static analysis tools...[/cyan]\n")
            analyzer_static = StaticAnalyzer(directory)

            if not analyzer_static.available_tools:
                console.print(
                    "[yellow]⚠️  No static analysis tools found. Install: pip install ruff mypy black isort[/yellow]\n"
                )
            else:
                console.print(
                    f"[cyan]Available tools: {', '.join(analyzer_static.available_tools)}[/cyan]\n"
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Running static analysis...", total=None)
                    static_results = analyzer_static.run_all()
                    progress.update(task, completed=True)

                _render_static_analysis_results(static_results)

        # Step 2: Create batches
        batcher = FileBatcher()
        batches = batcher.create_batches(files)

        console.print(f"📦 Created {len(batches)} batches\n")

        # Handle dry-run mode
        if dry_run:
            _render_dry_run(files, batches, model_info, model_name, console)
            return

        # Step 3: Analyze batches
        analyzer = CodeAnalyzer(
            region=aws_region, model_id=model_id, temperature=temperature
        )
        all_issues = []
        all_suggestions = []
        all_design_insights = []
        total_files = len(files)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Analyzing...", total=len(batches))

            for i, batch in enumerate(batches, 1):
                # Build file list description
                file_names = [f.name for f in batch.files]
                if len(file_names) <= 3:
                    files_desc = ", ".join(file_names)
                else:
                    files_desc = (
                        f"{', '.join(file_names[:3])} +{len(file_names) - 3} more"
                    )

                # Update progress with current batch info
                progress.update(
                    task,
                    description=f"[cyan]Batch {i}/{len(batches)}[/cyan] {files_desc}",
                )

                if verbose:
                    console.print(f"  Files: {', '.join(file_names)}")

                try:
                    report = analyzer.analyze_batch(batch)
                    all_issues.extend(report.issues)
                    all_suggestions.extend(report.improvement_suggestions)
                    if report.system_design_insights:
                        all_design_insights.append(report.system_design_insights)

                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    error_msg = e.response.get("Error", {}).get("Message", "")

                    if error_code == "AccessDeniedException":
                        console.print(f"\n[red]✗ AWS Access Denied: {error_msg}[/red]")
                        console.print(
                            "[yellow]Check that you have access to AWS Bedrock Claude Opus 4.5[/yellow]"
                        )
                    elif error_code in [
                        "ThrottlingException",
                        "TooManyRequestsException",
                    ]:
                        console.print(
                            f"\n[red]✗ Rate limit exceeded on batch {i}[/red]"
                        )
                        console.print(
                            "[yellow]Consider reducing batch size or waiting before retrying[/yellow]"
                        )
                    else:
                        console.print(
                            f"\n[red]✗ AWS Error on batch {i} ({error_code}): {error_msg}[/red]"
                        )

                    if verbose:
                        console.print(traceback.format_exc())

                except Exception as e:
                    console.print(f"\n[red]✗ Error analyzing batch {i}: {e}[/red]")
                    if verbose:
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
            "input_price_per_million": (
                model_info["input_price_per_million"] if model_info else 0
            ),
            "output_price_per_million": (
                model_info["output_price_per_million"] if model_info else 0
            ),
        }

        # Add static analysis metrics if available
        if static_results:
            static_summary = StaticAnalyzer.get_summary(static_results)
            metrics["static_analysis_run"] = True
            metrics["static_tools_passed"] = static_summary["tools_passed"]
            metrics["static_tools_failed"] = static_summary["tools_failed"]
            metrics["static_issues_found"] = static_summary["total_issues"]
        else:
            metrics["static_analysis_run"] = False

        # Aggregate system design insights from all batches
        aggregated_insights = (
            "\n\n".join(all_design_insights)
            if all_design_insights
            else "No architectural concerns identified."
        )

        final_report = CodeReviewReport(
            summary=f"Analyzed {total_files} files and found {len(all_issues)} issues",
            metrics=metrics,
            issues=all_issues,
            system_design_insights=aggregated_insights,
            recommendations=_generate_recommendations(all_issues),
            improvement_suggestions=all_suggestions[:5],  # Keep top 5 suggestions
        )

        # Display token usage and cost
        console.print("\n[cyan]💰 Token Usage & Cost Estimate:[/cyan]")
        console.print(f"   Input tokens:  {analyzer.total_input_tokens:,}")
        console.print(f"   Output tokens: {analyzer.total_output_tokens:,}")
        console.print(
            f"   Total tokens:  {analyzer.total_input_tokens + analyzer.total_output_tokens:,}"
        )

        # Calculate cost using model's pricing
        if model_info:
            input_price = float(model_info["input_price_per_million"])
            output_price = float(model_info["output_price_per_million"])
            input_cost = (analyzer.total_input_tokens / 1_000_000) * input_price
            output_cost = (analyzer.total_output_tokens / 1_000_000) * output_price
            total_cost = input_cost + output_cost
            console.print(f"   [bold]Estimated cost: ${total_cost:.4f}[/bold]")
        console.print()

        # Warn if any files were skipped
        if analyzer.skipped_files:
            console.print(
                f"[yellow]⚠️  Warning: {len(analyzer.skipped_files)} file(s) could not be read:[/yellow]"
            )
            for file_path, error in analyzer.skipped_files[:5]:  # Show first 5
                console.print(f"   • {file_path}: {error}")
            if len(analyzer.skipped_files) > 5:
                console.print(f"   ... and {len(analyzer.skipped_files) - 5} more")
            console.print()

        # Step 5: Render results (with severity filtering)
        renderer = TerminalRenderer()
        renderer.render(final_report, min_severity=severity)

        # Step 6: Export to Markdown if requested
        if output:
            exporter = MarkdownExporter()
            exporter.export(final_report, output)
            console.print(f"\n[green]✓ Report exported to: {output}[/green]\n")

    except NoCredentialsError:
        console.print("\n[red]✗ AWS credentials not found[/red]\n")
        console.print("[yellow]Please configure AWS credentials:[/yellow]")
        console.print("  1. Run: aws configure")
        console.print(
            "  2. Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
        )
        console.print("  3. Or use --aws-profile flag\n")
        raise click.Abort()

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_msg = e.response.get("Error", {}).get("Message", "")

        console.print(f"\n[red]✗ AWS Error ({error_code}): {error_msg}[/red]\n")

        if error_code == "AccessDeniedException":
            console.print("[yellow]Troubleshooting:[/yellow]")
            console.print("  1. Ensure you have access to AWS Bedrock in your region")
            console.print("  2. Check that Claude Opus 4.5 model access is enabled")
            console.print(
                "  3. Verify your IAM permissions include 'bedrock:InvokeModel'\n"
            )
        elif error_code == "ResourceNotFoundException":
            console.print(
                "[yellow]The Claude Opus 4.5 model may not be available in your region.[/yellow]"
            )
            console.print(
                "Try using --aws-region with a supported region (e.g., us-west-2)\n"
            )

        if verbose:
            console.print(traceback.format_exc())
        raise click.Abort()

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]\n")
        if verbose:
            console.print(traceback.format_exc())
        raise click.Abort()


def _render_static_analysis_results(results: dict) -> None:
    """Render static analysis results to terminal."""
    summary = StaticAnalyzer.get_summary(results)

    # Create summary table
    table = Table(
        title="Static Analysis Results", show_header=True, header_style="bold cyan"
    )
    table.add_column("Tool", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Issues", justify="right")

    for tool_name, result in results.items():
        tool_config = StaticAnalyzer.TOOLS.get(tool_name, {})
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
        console.print(
            f"[yellow]⚠️  {summary['tools_failed']} tool(s) found issues ({summary['total_issues']} total)[/yellow]\n"
        )

    # Show details for failed tools
    for tool_name, result in results.items():
        if not result.passed and not result.errors and result.output:
            tool_config = StaticAnalyzer.TOOLS.get(tool_name, {})
            tool_display = tool_config.get("name", tool_name)

            # Limit output to first 20 lines
            output_lines = result.output.split("\n")[:20]
            output_preview = "\n".join(output_lines)

            if len(result.output.split("\n")) > 20:
                output_preview += (
                    f"\n... ({len(result.output.split('\n')) - 20} more lines)"
                )

            console.print(
                Panel(
                    output_preview,
                    title=f"[yellow]{tool_display} Output[/yellow]",
                    border_style="yellow",
                )
            )
            console.print()


def _generate_recommendations(issues: list[ReviewIssue]) -> list[str]:
    """Generate top recommendations from issues."""
    critical = [i for i in issues if i.severity == "Critical"]
    high = [i for i in issues if i.severity == "High"]

    recommendations = []

    if critical:
        recommendations.append(
            f"🚨 Address {len(critical)} critical issues immediately"
        )

    if high:
        recommendations.append(f"⚠️  Fix {len(high)} high-priority issues")

    if len(issues) > 10:
        recommendations.append("📊 Consider refactoring to reduce technical debt")

    return recommendations[:5]


def _render_dry_run(files, batches, model_info, model_name, console) -> None:
    """Render dry-run output showing files and estimated costs."""
    from rich.table import Table

    console.print("[bold cyan]📋 Dry Run Mode[/bold cyan]\n")

    # Estimate tokens for each file (4 chars ≈ 1 token)
    def estimate_tokens(file_path: Path) -> int:
        """Estimate token count for a file (~4 chars per token heuristic)."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return len(content) // 4
        except (OSError, UnicodeDecodeError):
            return 0

    # Build file table
    table = Table(title="Files to Analyze", show_header=True, header_style="bold cyan")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Est. Tokens", justify="right")

    total_tokens = 0
    for file_path in files:
        size_kb = file_path.stat().st_size / 1024
        tokens = estimate_tokens(file_path)
        total_tokens += tokens
        table.add_row(
            str(file_path.name),
            f"{size_kb:.1f} KB",
            f"{tokens:,}",
        )

    console.print(table)
    console.print()

    # Add system prompt tokens estimate (~500 tokens per batch)
    system_prompt_tokens = len(batches) * 500
    total_input_tokens = total_tokens + system_prompt_tokens

    # Estimate output tokens (roughly 20% of input for code review)
    estimated_output_tokens = int(total_input_tokens * 0.2)

    # Calculate cost
    if model_info:
        input_price = float(model_info["input_price_per_million"])
        output_price = float(model_info["output_price_per_million"])
        input_cost = (total_input_tokens / 1_000_000) * input_price
        output_cost = (estimated_output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost
    else:
        input_price = output_price = total_cost = 0

    # Summary
    console.print("[bold]💰 Estimated Cost Summary[/bold]")
    console.print(f"   Model: {model_name}")
    console.print(f"   Files: {len(files)}")
    console.print(f"   Batches: {len(batches)}")
    console.print(f"   Est. input tokens: ~{total_input_tokens:,}")
    console.print(f"   Est. output tokens: ~{estimated_output_tokens:,}")
    console.print(f"   [bold]Est. cost: ${total_cost:.4f}[/bold]")
    console.print(f"      (Input: ${input_cost:.4f} @ ${input_price}/M)")
    console.print(f"      (Output: ${output_cost:.4f} @ ${output_price}/M)")
    console.print()
    console.print("[dim]Run without --dry-run to perform the actual analysis.[/dim]")


if __name__ == "__main__":
    main()
