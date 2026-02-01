"""CLI entry point for code review tool."""

import os
import time
import traceback
from pathlib import Path

import click
from botocore.exceptions import (  # type: ignore[import-untyped]
    ClientError,
    NoCredentialsError,
)
from rich.console import Console
from rich.markup import escape
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
from codereview.batcher import FileBatch, FileBatcher
from codereview.callbacks import (
    BaseCallbackHandler,
    ProgressCallbackHandler,
    StreamingCallbackHandler,
)
from codereview.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    MAX_FILE_SIZE_KB,
    MODEL_ALIASES,
)
from codereview.models import CodeReviewReport, ReviewIssue, ReviewMetrics
from codereview.providers.base import ModelProvider
from codereview.providers.factory import ProviderFactory
from codereview.readme_finder import (
    find_readme,
    prompt_readme_confirmation,
    read_readme_content,
)
from codereview.renderer import (
    MarkdownExporter,
    StaticAnalysisRenderer,
    TerminalRenderer,
    ValidationRenderer,
)
from codereview.scanner import FileScanner
from codereview.static_analysis import StaticAnalyzer

# Get all valid model names (both primary IDs and aliases)
ALL_MODEL_NAMES = sorted(MODEL_ALIASES.keys())
# Get only primary model IDs for display
PRIMARY_MODEL_IDS = sorted(set(MODEL_ALIASES.values()))


class ModelChoice(click.ParamType):
    """Custom Click type that accepts model IDs and aliases but shows only primary IDs."""

    name = "model"

    def get_metavar(
        self, param: click.Parameter, ctx: click.Context | None = None
    ) -> str:
        """Show MODEL placeholder in help - use --list-models for full list."""
        return "MODEL"

    def convert(
        self, value: str, param: click.Parameter | None, ctx: click.Context | None
    ) -> str:
        """Validate that value is a valid model ID or alias."""
        # Build case-insensitive lookup map
        lower_map = {name.lower(): name for name in ALL_MODEL_NAMES}
        normalized = lower_map.get(value.lower())
        if normalized:
            return normalized
        self.fail(
            f"'{value}' is not a valid model. "
            f"Choose from: {', '.join(PRIMARY_MODEL_IDS)} (or use --list-models)",
            param,
            ctx,
        )


console = Console()


def display_available_models() -> None:
    """Display all available models in a formatted table."""
    factory = ProviderFactory()
    models_by_provider = factory.list_available_models()

    # Create models table
    table = Table(
        title="Available Models", show_header=True, header_style="bold magenta"
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Provider", style="yellow")
    table.add_column("Aliases", style="blue")

    # Add rows grouped by provider
    for provider_name, models in sorted(models_by_provider.items()):
        for model in models:
            table.add_row(
                model["id"],
                model["name"],
                provider_name,
                model["aliases"],
            )

    console.print(table)
    console.print()

    # Create provider setup table
    setup_table = Table(
        title="Provider Setup", show_header=True, header_style="bold magenta"
    )
    setup_table.add_column("Provider", style="yellow", no_wrap=True)
    setup_table.add_column("Required Environment Variables", style="cyan")
    setup_table.add_column("Setup", style="dim")

    setup_table.add_row(
        "bedrock",
        "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\nor AWS_PROFILE",
        "aws configure",
    )
    setup_table.add_row(
        "azure_openai",
        "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY",
        "Azure Portal → OpenAI resource",
    )
    setup_table.add_row(
        "nvidia",
        "NVIDIA_API_KEY",
        "https://build.nvidia.com",
    )

    console.print(setup_table)
    console.print()
    console.print("[bold]Usage:[/bold] codereview <directory> --model <id>")
    console.print("[dim]Example: codereview ./src --model opus[/dim]")


def validate_provider_credentials(model_name: str, aws_profile: str | None) -> None:
    """Validate provider credentials without running analysis.

    Args:
        model_name: Model ID or alias to validate
        aws_profile: Optional AWS profile to use
    """
    # Set AWS profile if provided
    if aws_profile:
        os.environ["AWS_PROFILE"] = aws_profile

    try:
        factory = ProviderFactory()
        provider = factory.create_provider(model_name)

        console.print(
            f"\n[bold]Validating credentials for:[/bold] {provider.get_model_display_name()}"
        )
        console.print()

        # Run validation
        result = provider.validate_credentials()

        # Render results
        renderer = ValidationRenderer(console)
        renderer.render(result)

        # Exit with appropriate code
        if not result.valid:
            raise SystemExit(1)

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    except (NoCredentialsError, ClientError) as e:
        console.print(f"[red]AWS Error:[/red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise SystemExit(1)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output Markdown report to file (e.g., report.md)",
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
@click.option(
    "--max-files",
    type=click.IntRange(min=1, max=10000),
    help="Maximum number of files to analyze (1-10000)",
)
@click.option(
    "--max-file-size",
    type=click.IntRange(min=1, max=10240),
    default=MAX_FILE_SIZE_KB,
    help=f"Maximum file size in KB (default: {MAX_FILE_SIZE_KB}, max: 10240)",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1, max=50),
    default=10,
    help="Files per batch (default: 10). Smaller batches may help with timeout issues.",
)
@click.option("--aws-profile", type=str, help="AWS CLI profile to use")
@click.option(
    "--model",
    "-m",
    "model_name",
    type=ModelChoice(),
    default="opus",
    help="Model to use (default: opus). Use --list-models to see all options.",
)
@click.option(
    "--static-analysis",
    is_flag=True,
    help="Run static analysis tools for all supported languages",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Temperature for model inference (uses model-specific default if not specified)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show files and estimated cost without making API calls",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@click.option(
    "--stream",
    is_flag=True,
    help="Enable streaming output with real-time token display",
)
@click.option(
    "--list-models",
    is_flag=True,
    help="List all available models and exit",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate provider credentials without running analysis",
)
@click.option(
    "--readme",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Specify README file for project context (skips auto-discovery)",
)
@click.option(
    "--no-readme",
    is_flag=True,
    help="Skip README context entirely (no prompt)",
)
@click.pass_context
def main(
    ctx: click.Context,
    directory: Path | None,
    output: Path | None,
    severity: str,
    exclude: tuple[str, ...],
    max_files: int | None,
    max_file_size: int,
    batch_size: int,
    aws_profile: str | None,
    model_name: str,
    static_analysis: bool,
    temperature: float | None,
    dry_run: bool,
    verbose: bool,
    stream: bool,
    list_models: bool,
    validate: bool,
    readme: Path | None,
    no_readme: bool,
) -> None:
    """
    Analyze code in DIRECTORY and generate a comprehensive review report.

    Reviews Python, Go, Shell, C++, Java, JavaScript, and TypeScript files using
    LLM models via AWS Bedrock or Azure OpenAI.
    """
    # Handle --list-models flag first
    if list_models:
        display_available_models()
        return

    # Handle --validate flag
    if validate:
        validate_provider_credentials(model_name, aws_profile)
        return

    # Show help if no directory provided
    if directory is None:
        click.echo(ctx.get_help())
        ctx.exit(0)

    # Initialize callback handlers for cleanup in finally block
    streaming_handler: StreamingCallbackHandler | None = None
    progress_handler: ProgressCallbackHandler | None = None

    try:
        # Set AWS profile if provided
        if aws_profile:
            os.environ["AWS_PROFILE"] = aws_profile

        # Get model display name from provider
        factory = ProviderFactory()
        provider = factory.create_provider(model_name, temperature)
        model_display_name = str(provider.get_model_display_name())

        console.print("\n[bold cyan]🔍 Code Review Tool[/bold cyan]\n")
        console.print(f"📂 Scanning directory: {directory}")
        console.print(f"🤖 Model: {model_display_name}\n")

        # Handle README context
        readme_content: str | None = None
        if not no_readme:
            if readme:
                # User specified a README file via --readme
                result = read_readme_content(readme)
                if result:
                    content, size = result
                    readme_content = content
                    console.print(
                        f"📄 Using README: [cyan]{readme}[/cyan] ({size/1024:.1f} KB)\n"
                    )
                else:
                    console.print(f"[yellow]⚠️  Could not read {readme}[/yellow]\n")
            else:
                # Auto-discover README
                found_readme = find_readme(directory)
                confirmed_readme = prompt_readme_confirmation(found_readme, console)
                if confirmed_readme:
                    result = read_readme_content(confirmed_readme)
                    if result:
                        readme_content, _ = result
                console.print()  # Blank line after README section

        # Start timing
        start_time = time.time()

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

            if max_files and len(files) > max_files:
                console.print(
                    f"[yellow]⚠️  Limiting analysis to {max_files} of {len(files)} files[/yellow]"
                )
                files = files[:max_files]

            progress.update(task, completed=True)

        if not files:
            console.print("[yellow]⚠️  No files found to review[/yellow]")
            return

        console.print(f"✓ Found {len(files)} files to review")

        # Count total lines of code
        total_lines = 0
        for file_path in files:
            try:
                with file_path.open("r", encoding="utf-8", errors="replace") as f:
                    total_lines += sum(1 for _ in f)
            except OSError:
                # Skip files that can't be read
                pass

        console.print(f"✓ Total lines of code: {total_lines:,}")

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

                StaticAnalysisRenderer(console).render(static_results)

        # Step 2: Create batches
        batcher = FileBatcher(max_files_per_batch=batch_size)
        batches = batcher.create_batches(files)

        try:
            console.print(f"📦 Created {len(batches)} batches\n")
        except OSError:
            # Handle terminal I/O errors
            print(f"Created {len(batches)} batches")

        # Handle dry-run mode
        if dry_run:
            _render_dry_run(
                files, batches, model_name, model_display_name, provider, console
            )
            return

        # Step 3: Analyze batches
        # Set up callbacks for streaming/progress if requested
        callbacks: list[BaseCallbackHandler] | None = None
        if stream:
            streaming_handler = StreamingCallbackHandler(console=console, verbose=True)
            callbacks = [streaming_handler]
        elif verbose:
            progress_handler = ProgressCallbackHandler(console=console)
            callbacks = [progress_handler]

        analyzer = CodeAnalyzer(
            model_name=model_name,
            temperature=temperature,
            callbacks=callbacks,
            project_context=readme_content,
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
                        console.print(
                            f"\n[red]✗ AWS Access Denied: {escape(error_msg)}[/red]"
                        )
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
                            f"\n[red]✗ AWS Error on batch {i} ({escape(error_code)}): {escape(error_msg)}[/red]"
                        )

                    if verbose:
                        console.print(traceback.format_exc())

                except (ValueError, KeyError) as e:
                    # Parse/validation errors from structured output
                    console.print(
                        f"\n[red]✗ Parse error in batch {i}: {type(e).__name__}: {escape(str(e))}[/red]"
                    )
                    if verbose:
                        console.print(traceback.format_exc())

                except (RuntimeError, OSError) as e:
                    # Recoverable errors - log and continue with other batches
                    console.print(
                        f"\n[red]✗ Error analyzing batch {i}: "
                        f"{type(e).__name__}: {escape(str(e))}[/red]"
                    )
                    if verbose:
                        console.print(traceback.format_exc())

                progress.update(task, advance=1)

        # Step 4: Create final report
        # Get pricing from provider
        pricing = analyzer.provider.get_pricing()

        # Build metrics object
        metrics = ReviewMetrics(
            files_analyzed=total_files,
            total_lines=total_lines,
            total_issues=len(all_issues),
            critical=sum(1 for i in all_issues if i.severity == "Critical"),
            high=sum(1 for i in all_issues if i.severity == "High"),
            medium=sum(1 for i in all_issues if i.severity == "Medium"),
            low=sum(1 for i in all_issues if i.severity == "Low"),
            info=sum(1 for i in all_issues if i.severity == "Info"),
            input_tokens=analyzer.provider.total_input_tokens,
            output_tokens=analyzer.provider.total_output_tokens,
            total_tokens=analyzer.provider.total_input_tokens
            + analyzer.provider.total_output_tokens,
            model_name=model_display_name,
            input_price_per_million=pricing["input_price_per_million"],
            output_price_per_million=pricing["output_price_per_million"],
            static_analysis_run=False,
        )

        # Add static analysis metrics if available
        if static_results:
            static_summary = StaticAnalyzer.get_summary(static_results)
            # Create a new metrics object with static analysis fields
            # Exclude static_analysis fields to avoid duplicate keyword arguments
            base_metrics = metrics.model_dump(
                exclude_none=True,
                exclude={
                    "static_analysis_run",
                    "static_tools_passed",
                    "static_tools_failed",
                    "static_issues_found",
                },
            )
            metrics = ReviewMetrics(
                **base_metrics,
                static_analysis_run=True,
                static_tools_passed=static_summary["tools_passed"],
                static_tools_failed=static_summary["tools_failed"],
                static_issues_found=static_summary["total_issues"],
            )

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
        console.print(f"   Input tokens:  {analyzer.provider.total_input_tokens:,}")
        console.print(f"   Output tokens: {analyzer.provider.total_output_tokens:,}")
        console.print(
            f"   Total tokens:  {analyzer.provider.total_input_tokens + analyzer.provider.total_output_tokens:,}"
        )

        # Calculate cost using model's pricing
        input_price = float(pricing["input_price_per_million"])
        output_price = float(pricing["output_price_per_million"])
        input_cost = (analyzer.provider.total_input_tokens / 1_000_000) * input_price
        output_cost = (analyzer.provider.total_output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost
        console.print(f"   [bold]Estimated cost: ${total_cost:.4f}[/bold]")
        console.print()

        # Warn if any files were skipped
        if analyzer.skipped_files:
            console.print(
                f"[yellow]⚠️  Warning: {len(analyzer.skipped_files)} file(s) could not be read:[/yellow]"
            )
            for skipped_file, error in analyzer.skipped_files[:5]:  # Show first 5
                console.print(f"   • {skipped_file}: {error}")
            if len(analyzer.skipped_files) > 5:
                console.print(f"   ... and {len(analyzer.skipped_files) - 5} more")
            console.print()

        # Step 5: Render results (with severity filtering)
        renderer = TerminalRenderer()
        renderer.render(final_report, min_severity=severity)

        # Display total elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            console.print(f"[dim]⏱️  Completed in {minutes}m {seconds}s[/dim]\n")
        else:
            console.print(f"[dim]⏱️  Completed in {elapsed_time:.1f}s[/dim]\n")

        # Step 6: Export to Markdown if requested
        if output:
            # Collect all skipped files for the report
            all_skipped_files: list[tuple[str, str]] = []

            # Add scanner skipped files (convert Path to str)
            for skipped_path, reason in scanner.skipped_files:
                all_skipped_files.append((str(skipped_path), reason))

            # Add analyzer skipped files (already strings)
            all_skipped_files.extend(analyzer.skipped_files)

            exporter = MarkdownExporter()
            exporter.export(
                final_report,
                output,
                skipped_files=all_skipped_files if all_skipped_files else None,
            )
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

        console.print(
            f"\n[red]✗ AWS Error ({escape(error_code)}): {escape(error_msg)}[/red]\n"
        )

        if error_code == "AccessDeniedException":
            console.print("[yellow]Troubleshooting:[/yellow]")
            console.print("  1. Ensure you have access to AWS Bedrock in your region")
            console.print("  2. Check that Claude Opus 4.5 model access is enabled")
            console.print(
                "  3. Verify your IAM permissions include 'bedrock:InvokeModel'\n"
            )
        elif error_code == "ResourceNotFoundException":
            console.print(
                "[yellow]The requested model may not be available in your AWS region.[/yellow]"
            )
            console.print(
                "Check that the model is enabled in your AWS Bedrock settings.\n"
            )

        if verbose:
            console.print(traceback.format_exc())
        raise click.Abort()

    except Exception as e:
        console.print(f"\n[red]✗ Error: {escape(str(e))}[/red]\n")
        if verbose:
            console.print(traceback.format_exc())
        raise click.Abort()

    finally:
        # Ensure callback handlers are cleaned up
        if streaming_handler:
            streaming_handler.cleanup()
        if progress_handler:
            progress_handler.cleanup()


def _generate_recommendations(issues: list[ReviewIssue]) -> list[str]:
    """Generate top recommendations based on issue severity distribution.

    Args:
        issues: List of ReviewIssue objects from analysis

    Returns:
        Up to 5 prioritized recommendations based on issue counts
    """
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


def _render_dry_run(
    files: list[Path],
    batches: list[FileBatch],
    _model_name: str,
    model_display_name: str,
    provider: ModelProvider,
    console: Console,
) -> None:
    """Render dry-run output showing files, validation, and estimated costs.

    Args:
        files: List of file paths to analyze
        batches: List of FileBatch objects
        _model_name: Model name (unused, kept for call-site compatibility)
        model_display_name: Human-readable model name for display
        provider: Model provider instance
        console: Rich console for output
    """
    # Get pricing from provider
    pricing = provider.get_pricing()

    console.print("[bold cyan]📋 Dry Run Mode[/bold cyan]\n")

    # Pre-flight validation
    console.print("[cyan]🔍 Pre-flight Validation[/cyan]\n")
    validation = provider.validate_credentials()
    ValidationRenderer(console).render(validation)
    console.print()

    # Estimate tokens from file size (avoids reading file content for speed)
    def estimate_tokens_from_size(file_path: Path) -> int:
        """Estimate token count from file size (~4 bytes per token heuristic)."""
        try:
            return file_path.stat().st_size // 4
        except OSError:
            return 0

    # Build file table
    table = Table(title="Files to Analyze", show_header=True, header_style="bold cyan")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Est. Tokens", justify="right")

    total_tokens = 0
    for file_path in files:
        try:
            size_kb = file_path.stat().st_size / 1024
        except OSError:
            size_kb = 0.0  # File may have been deleted since scan
        tokens = estimate_tokens_from_size(file_path)
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
    input_price = float(pricing["input_price_per_million"])
    output_price = float(pricing["output_price_per_million"])
    input_cost = (total_input_tokens / 1_000_000) * input_price
    output_cost = (estimated_output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    # Summary
    console.print("[bold]💰 Estimated Cost Summary[/bold]")
    console.print(f"   Model: {model_display_name}")
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
