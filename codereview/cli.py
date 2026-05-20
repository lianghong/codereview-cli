"""CLI entry point for code review tool."""

import os
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from codereview.batcher import BYTES_PER_TOKEN, FileBatch, FileBatcher, count_tokens
from codereview.callbacks import (
    BaseCallbackHandler,
    ProgressCallbackHandler,
    StreamingCallbackHandler,
)
from codereview.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    MAX_FILE_SIZE_KB,
    MODEL_ALIASES,
    SYSTEM_PROMPT,
    detect_languages_from_paths,
    get_config_loader,
)
from codereview.models import (
    CodeReviewReport,
    ReviewIssue,
    ReviewMetrics,
    get_drift_counters,
    reset_drift_counters,
)
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


def validate_exclude_pattern(pattern: str) -> bool:
    """Validate exclude pattern for safety (prevent ReDoS attacks).

    Args:
        pattern: Glob pattern to validate

    Returns:
        True if pattern is safe to use, False otherwise
    """
    # Limit pattern length to prevent ReDoS
    if len(pattern) > 200:
        return False
    # Limit ** recursion depth to prevent catastrophic backtracking
    if pattern.count("**") > 3:
        return False
    # Disallow null bytes
    if "\x00" in pattern:
        return False
    return True


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
        return value  # unreachable: self.fail() raises


def _create_console(quiet: bool = False, no_color: bool = False) -> Console:
    """Create a Rich console, optionally in quiet or no-color mode.

    Args:
        quiet: If True, suppress all Rich output via Console(quiet=True)
        no_color: If True, strip ANSI color/style codes from output

    Returns:
        Console instance
    """
    if quiet:
        return Console(quiet=True)
    if no_color:
        return Console(no_color=True, highlight=False)
    return Console()


def _format_batch_files_desc(file_names: list[str]) -> str:
    """Format batch file list for progress display."""
    if len(file_names) <= 3:
        return ", ".join(file_names)
    return f"{', '.join(file_names[:3])} +{len(file_names) - 3} more"


def _render_batch_error(
    con: Console,
    batch_num: int,
    error: Exception,
    model_display_name: str,
    verbose: bool,
) -> None:
    """Render a per-batch error message to the console.

    Centralizes the formatting that was previously inline in the batch loop.
    Called from the main thread after futures complete, so console writes
    don't interleave.
    """
    if isinstance(error, ClientError):
        error_code = error.response.get("Error", {}).get("Code", "")
        error_msg = error.response.get("Error", {}).get("Message", "")
        if error_code == "AccessDeniedException":
            con.print(f"\n[red]✗ AWS Access Denied: {escape(error_msg)}[/red]")
            con.print(
                f"[yellow]Check that you have access to "
                f"{escape(model_display_name)} on AWS Bedrock[/yellow]"
            )
        elif error_code in ("ThrottlingException", "TooManyRequestsException"):
            con.print(f"\n[red]✗ Rate limit exceeded on batch {batch_num}[/red]")
            con.print(
                "[yellow]Consider reducing batch size or "
                "waiting before retrying[/yellow]"
            )
        else:
            con.print(
                f"\n[red]✗ AWS Error on batch {batch_num} "
                f"({escape(error_code)}): {escape(error_msg)}[/red]"
            )
    elif isinstance(error, (ValueError, KeyError)):
        con.print(
            f"\n[red]✗ Parse error in batch {batch_num}: "
            f"{type(error).__name__}: {escape(str(error))}[/red]"
        )
    elif isinstance(error, (RuntimeError, OSError)):
        con.print(
            f"\n[red]✗ Error analyzing batch {batch_num}: "
            f"{type(error).__name__}: {escape(str(error))}[/red]"
        )
    else:
        # Provider-specific errors after retries exhausted
        # (e.g., openai.RateLimitError, httpx.HTTPStatusError,
        #  google.api_core.exceptions.ResourceExhausted)
        error_str = str(error)
        is_rate_limit = "429" in error_str or "rate" in error_str.lower()
        if is_rate_limit:
            con.print(
                f"\n[red]✗ Rate limit exceeded on batch {batch_num} after retries[/red]"
            )
            con.print(
                "[yellow]Consider reducing batch size with "
                "--batch-size or waiting before retrying[/yellow]"
            )
        else:
            con.print(
                f"\n[red]✗ Error on batch {batch_num}: "
                f"{type(error).__name__}: {escape(error_str)}[/red]"
            )

    if verbose:
        # Worker threads (ThreadPoolExecutor) re-raise the original exception
        # at future.result(); traceback.format_exc() would only show that
        # re-raise frame. Format from the exception's own __traceback__ so we
        # surface where the failure actually happened inside the worker.
        tb = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        con.print(tb)


def _is_pricing_tbd(input_price: float, output_price: float) -> bool:
    """Check if model pricing is a placeholder (provider hasn't published rates).

    Many free-tier or preview models in models.yaml use 0.00 placeholders.
    Treating these as $0 in cost output misleads users into thinking the
    model is permanently free; surface them as TBD instead.
    """
    return input_price == 0.0 and output_price == 0.0


def display_available_models(console: Console) -> None:
    """Display all available models in a formatted table.

    Args:
        console: Rich console instance (respects --no-color flag)
    """
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

    # Add rows grouped by provider, sorted by model ID within each provider
    for provider_name, models in sorted(models_by_provider.items()):
        for model in sorted(models, key=lambda m: m["id"]):
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
        "google_genai",
        "GOOGLE_API_KEY",
        "https://aistudio.google.com/apikey",
    )
    setup_table.add_row(
        "nvidia",
        "NVIDIA_API_KEY",
        "https://build.nvidia.com",
    )
    setup_table.add_row(
        "zai",
        "ZAI_API_KEY",
        "https://z.ai (Zhipu international)",
    )
    setup_table.add_row(
        "deepseek",
        "DEEPSEEK_API_KEY",
        "https://platform.deepseek.com/api_keys",
    )
    setup_table.add_row(
        "moonshot",
        "KIMI_API_KEY",
        "platform.moonshot.cn (Kimi); .ai for international",
    )

    console.print(setup_table)
    console.print()
    console.print("[bold]Usage:[/bold] codereview <directory> --model <id>")
    console.print("[dim]Example: codereview ./src --model opus[/dim]")


def validate_provider_credentials(
    model_name: str, aws_profile: str | None, console: Console
) -> None:
    """Validate provider credentials without running analysis.

    Args:
        model_name: Model ID or alias to validate
        aws_profile: Optional AWS profile to use
        console: Rich console instance (respects --no-color flag)
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
    help="Output report to file (e.g., report.md or report.json)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json"], case_sensitive=False),
    default="markdown",
    help="Output format when using --output (default: markdown)",
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
    "--include-hidden",
    is_flag=True,
    help="Scan inside hidden directories (e.g. .github/scripts). "
    "By default, directories starting with '.' are skipped.",
)
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
    default="opus4.7",
    help="Model to use (default: opus4.7). Use --list-models to see all options.",
)
@click.option(
    "--static-analysis",
    is_flag=True,
    help="Run static analysis tools for all supported languages",
)
@click.option(
    "--tool-timeout",
    type=click.IntRange(min=1, max=3600),
    default=None,
    help=(
        "Per-tool subprocess timeout in seconds (default: 120). "
        "Raise this for slow runs like cppcheck --enable=all on large "
        "C++ repos or mypy strict on big Python codebases."
    ),
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
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress all terminal output (for CI/CD). Only exit code and --output file.",
)
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
@click.option(
    "--no-color",
    is_flag=True,
    help="Disable ANSI colors/styles for copy-paste friendly output",
)
@click.pass_context
def main(
    ctx: click.Context,
    directory: Path | None,
    output: Path | None,
    output_format: str,
    severity: str,
    exclude: tuple[str, ...],
    include_hidden: bool,
    max_files: int | None,
    max_file_size: int,
    batch_size: int,
    aws_profile: str | None,
    model_name: str,
    static_analysis: bool,
    tool_timeout: int | None,
    temperature: float | None,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
    stream: bool,
    list_models: bool,
    validate: bool,
    readme: Path | None,
    no_readme: bool,
    no_color: bool,
) -> None:
    """
    Analyze code in DIRECTORY and generate a comprehensive review report.

    Reviews Python, Go, Shell, C++, Java, JavaScript, and TypeScript files using
    LLM models via AWS Bedrock, Azure OpenAI, NVIDIA NIM, or Google Generative AI.
    """
    # Create console early to support --list-models and --validate with --no-color
    con = _create_console(quiet=quiet, no_color=no_color)

    # Handle --list-models flag first
    if list_models:
        display_available_models(con)
        return

    # Handle --validate flag
    if validate:
        validate_provider_credentials(model_name, aws_profile, con)
        return

    # Show help if no directory provided
    if directory is None:
        click.echo(ctx.get_help())
        ctx.exit(0)
        return

    # Quiet mode overrides verbose and stream
    if quiet:
        verbose = False
        stream = False

    # Initialize callback handlers for cleanup in finally block
    streaming_handler: StreamingCallbackHandler | None = None
    progress_handler: ProgressCallbackHandler | None = None

    try:
        # Set AWS profile if provided
        if aws_profile:
            os.environ["AWS_PROFILE"] = aws_profile

        # Get model display name from config (avoids creating provider just for the name)
        config_loader = get_config_loader()
        _, model_config = config_loader.resolve_model(model_name)
        model_display_name = model_config.name

        con.print("\n[bold cyan]🔍 Code Review Tool[/bold cyan]\n")
        con.print(f"📂 Scanning directory: {directory}")
        con.print(f"🤖 Model: {model_display_name}\n")

        # Handle README context
        readme_content: str | None = None
        if not no_readme:
            if readme:
                # User specified a README file via --readme
                result = read_readme_content(readme)
                if result:
                    content, size = result
                    readme_content = content
                    con.print(
                        f"📄 Using README: [cyan]{readme}[/cyan] ({size / 1024:.1f} KB)\n"
                    )
                else:
                    con.print(f"[yellow]⚠️  Could not read {readme}[/yellow]\n")
            else:
                # Auto-discover README
                found_readme = find_readme(directory)
                confirmed_readme = prompt_readme_confirmation(found_readme, con)
                if confirmed_readme:
                    result = read_readme_content(confirmed_readme)
                    if result:
                        readme_content, _ = result
                con.print()  # Blank line after README section

        # Start timing
        start_time = time.time()

        # Step 1: Scan files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=con,
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)

            # Validate and filter user-provided exclude patterns
            validated_exclude = []
            rejected_exclude = []
            for p in exclude:
                if validate_exclude_pattern(p):
                    validated_exclude.append(p)
                else:
                    rejected_exclude.append(p)
            for p in rejected_exclude:
                con.print(
                    f"[yellow]⚠️  Ignoring invalid --exclude pattern: {escape(repr(p))} "
                    f"(too long, too complex, or contains invalid characters)[/yellow]"
                )

            # Combine default exclusions with validated user-provided patterns
            exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS) + validated_exclude
            scanner = FileScanner(
                directory,
                exclude_patterns=exclude_patterns,
                max_file_size_kb=max_file_size,
                exclude_hidden=not include_hidden,
            )
            files = scanner.scan()

            if max_files and len(files) > max_files:
                con.print(
                    f"[yellow]⚠️  Limiting analysis to {max_files} of {len(files)} files[/yellow]"
                )
                files = files[:max_files]

            progress.update(task, completed=True)

        if not files:
            con.print("[yellow]⚠️  No files found to review[/yellow]")
            return

        con.print(f"✓ Found {len(files)} files to review")

        # Count total lines of code. Use binary chunked newline counting rather
        # than decoding each line (~2-3x faster and avoids decode overhead).
        # The LLM read happens again later in analyze_batch; bytes-only scan
        # here keeps the second read cheaper without changing its encoding.
        total_lines = 0
        for file_path in files:
            try:
                with file_path.open("rb") as fb:
                    while chunk := fb.read(65536):
                        total_lines += chunk.count(b"\n")
            except OSError:
                # Skip files that can't be read
                pass

        con.print(f"✓ Total lines of code: {total_lines:,}")

        # Report files skipped during scanning (e.g., too large)
        if scanner.skipped_files:
            con.print(
                f"[yellow]⚠️  {len(scanner.skipped_files)} file(s) skipped during scan:[/yellow]"
            )
            for skipped_path, reason in scanner.skipped_files[:5]:
                con.print(f"   • {skipped_path.name}: {reason}")
            if len(scanner.skipped_files) > 5:
                con.print(f"   ... and {len(scanner.skipped_files) - 5} more")
        con.print()

        # Step 1.5: Run static analysis if requested
        static_results = None
        if static_analysis:
            con.print("[cyan]Running static analysis tools...[/cyan]\n")
            analyzer_static = StaticAnalyzer(directory, tool_timeout=tool_timeout)

            if not analyzer_static.available_tools:
                con.print(
                    "[yellow]⚠️  No static analysis tools found. Install: pip install ruff mypy black isort[/yellow]\n"
                )
            else:
                con.print(
                    f"[cyan]Available tools: {', '.join(analyzer_static.available_tools)}[/cyan]\n"
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=con,
                ) as progress:
                    task = progress.add_task("Running static analysis...", total=None)
                    static_results = analyzer_static.run_all()
                    progress.update(task, completed=True)

                StaticAnalysisRenderer(con).render(static_results)

        # Step 2: Create batches (token-budget-aware when possible)
        token_budget: int | None = None
        if model_config.context_window:
            max_output = (
                model_config.inference_params.max_output_tokens
                if model_config.inference_params
                and model_config.inference_params.max_output_tokens
                else 16000
            )
            # Use the real tokenizer (tiktoken cl100k_base) for fixed prompt
            # blocks too — they're constant strings, encoded once per run.
            system_prompt_tokens = count_tokens(SYSTEM_PROMPT)
            readme_tokens = count_tokens(readme_content) if readme_content else 0
            # Reserve an upper bound for the linter findings block when static
            # analysis ran. condense_for_prompt enforces the same 4 KB cap.
            linter_tokens = (4000 // BYTES_PER_TOKEN) if static_results else 0
            safety_margin = max(1000, min(model_config.context_window // 10, 20000))
            computed_budget = (
                model_config.context_window
                - max_output
                - system_prompt_tokens
                - readme_tokens
                - linter_tokens
                - safety_margin
            )
            if computed_budget > 0:
                token_budget = computed_budget
                if verbose:
                    con.print("[cyan]Token budget breakdown:[/cyan]")
                    con.print(
                        f"   Context window:    {model_config.context_window:>10,}"
                    )
                    con.print(f"   Max output:        {max_output:>10,}")
                    con.print(f"   System prompt:     {system_prompt_tokens:>10,}")
                    con.print(f"   README context:    {readme_tokens:>10,}")
                    if linter_tokens:
                        con.print(f"   Linter findings:   {linter_tokens:>10,}")
                    con.print(f"   Safety margin:     {safety_margin:>10,}")
                    con.print(f"   [bold]Token budget:      {token_budget:>10,}[/bold]")
                    con.print()
            elif verbose:
                con.print(
                    "[yellow]Token budget negative; falling back to count-only batching[/yellow]\n"
                )

        batcher = FileBatcher(max_files_per_batch=batch_size, token_budget=token_budget)
        batches = batcher.create_batches(files)

        if batcher.skipped_oversized:
            try:
                con.print(
                    f"[yellow]⚠ Skipped {len(batcher.skipped_oversized)} file(s) "
                    "too large to review with this model:[/yellow]"
                )
                for skipped_path, est_tokens in batcher.skipped_oversized:
                    con.print(f"   {skipped_path.name} (~{est_tokens:,} tokens)")
                con.print()
            except OSError:
                for skipped_path, est_tokens in batcher.skipped_oversized:
                    print(
                        f"  Skipped (too large): {skipped_path.name} (~{est_tokens:,} tokens)"
                    )

        try:
            con.print(f"📦 Created {len(batches)} batches\n")
        except OSError:
            # Handle terminal I/O errors
            print(f"Created {len(batches)} batches")

        # Handle dry-run mode
        if dry_run:
            factory = ProviderFactory(config_loader=config_loader)
            dry_run_provider = factory.create_provider(model_name, temperature)
            _render_dry_run(files, batches, model_display_name, dry_run_provider, con)
            return

        # Step 3: Analyze batches
        # Set up callbacks for streaming/progress if requested
        callbacks: list[BaseCallbackHandler] | None = None
        if stream:
            streaming_handler = StreamingCallbackHandler(console=con, verbose=True)
            callbacks = [streaming_handler]
        elif verbose:
            progress_handler = ProgressCallbackHandler(console=con)
            callbacks = [progress_handler]

        # Drift counters are process-wide; zero them at the top of each run so
        # the post-run report only reflects this invocation.
        reset_drift_counters()

        analyzer = CodeAnalyzer(
            model_name=model_name,
            temperature=temperature,
            callbacks=callbacks,
            project_context=readme_content,
        )

        # Hand the LLM the raw linter results so each batch can be sliced
        # down to just its own files at prompt-prep time. Pre-condensing
        # would force every batch to see every other batch's diagnostics.
        if static_results:
            analyzer.set_linter_findings(static_results)
            if verbose:
                tools_with_issues = sum(
                    1
                    for r in static_results.values()
                    if not r.passed or r.issues_count > 0
                )
                con.print(
                    f"[dim]Linter findings from {tools_with_issues} tool(s) "
                    "will be sliced per-batch into review prompts.[/dim]"
                )
        all_issues = []
        all_suggestions = []
        all_design_insights = []
        failed_batches = 0
        total_files = len(files)

        # Concurrency: parallel batches give a 3-5x speedup on multi-batch runs.
        # Streaming forces sequential execution because token-by-token output
        # from concurrent batches would interleave incomprehensibly.
        # Single-batch runs also use sequential to avoid executor overhead.
        max_workers = 1 if (stream or len(batches) <= 1) else min(len(batches), 4)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=con,
            transient=False,
        ) as progress:
            task = progress.add_task(
                f"Analyzing ({max_workers} worker{'s' if max_workers > 1 else ''})...",
                total=len(batches),
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(analyzer.analyze_batch, batch): (i, batch)
                    for i, batch in enumerate(batches, 1)
                }

                for future in as_completed(future_to_batch):
                    i, batch = future_to_batch[future]
                    file_names = [f.name for f in batch.files]
                    files_desc = _format_batch_files_desc(file_names)

                    progress.update(
                        task,
                        description=(
                            f"[cyan]Batch {i}/{len(batches)}[/cyan] {files_desc}"
                        ),
                    )

                    if verbose:
                        con.print(f"  Files: {', '.join(file_names)}")

                    try:
                        report = future.result()
                        all_issues.extend(report.issues)
                        all_suggestions.extend(report.improvement_suggestions)
                        if report.system_design_insights:
                            all_design_insights.append(report.system_design_insights)
                    except Exception as e:
                        failed_batches += 1
                        _render_batch_error(con, i, e, model_display_name, verbose)

                    progress.update(task, advance=1)

        # Abort if every batch failed — no results to report
        if failed_batches == len(batches):
            con.print()
            con.print(
                f"[bold red]✗ All {len(batches)} batch(es) failed. "
                "No code review results to report.[/bold red]"
            )
            con.print("[yellow]Possible causes:[/yellow]")
            con.print("  - API rate limits exceeded (wait and retry)")
            con.print("  - Invalid or expired credentials")
            con.print("  - Model service temporarily unavailable")
            elapsed = time.time() - start_time
            if elapsed >= 60:
                m, s = int(elapsed // 60), int(elapsed % 60)
                con.print(f"\n[dim]⏱️  Completed in {m}m {s}s[/dim]")
            else:
                con.print(f"\n[dim]⏱️  Completed in {elapsed:.1f}s[/dim]")
            return

        # Warn about partial results when some batches failed
        if failed_batches > 0:
            succeeded = len(batches) - failed_batches
            con.print(
                f"\n[yellow]⚠ {failed_batches}/{len(batches)} batch(es) failed. "
                f"Results below are from {succeeded} successful batch(es) only.[/yellow]"
            )

        # Step 4: Create final report
        # Get pricing from provider
        pricing = analyzer.provider.get_pricing()

        # Collapse near-duplicates that concurrent batches reported independently.
        before_dedupe = len(all_issues)
        all_issues = _dedupe_issues(all_issues)
        if verbose and before_dedupe > len(all_issues):
            con.print(
                f"[dim]Deduplicated {before_dedupe - len(all_issues)} "
                f"cross-batch duplicate issue(s).[/dim]"
            )

        # Surface schema-coercion drift so prompt drift doesn't stay hidden in
        # logs. A high coercion rate means the model is emitting categories,
        # severities, or reference URLs that don't match our schema — usually
        # a signal that the prompt or the model has drifted.
        drift = get_drift_counters()
        total_drift = sum(drift.values())
        if total_drift > 0 and (verbose or total_drift >= 5):
            con.print(
                f"[yellow]⚠ Schema drift: {drift['severity_coerced']} severity, "
                f"{drift['category_coerced']} category, "
                f"{drift['reference_dropped']} reference URL(s) coerced/dropped. "
                "May indicate prompt drift; rerun with --verbose for log details.[/yellow]"
            )

        # Build metrics object
        severity_counts = Counter(i.severity for i in all_issues)
        metrics = ReviewMetrics(
            files_analyzed=total_files,
            total_lines=total_lines,
            total_issues=len(all_issues),
            critical_issues=severity_counts.get("Critical", 0),
            high_issues=severity_counts.get("High", 0),
            medium_issues=severity_counts.get("Medium", 0),
            low_issues=severity_counts.get("Low", 0),
            info_issues=severity_counts.get("Info", 0),
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
            base_metrics = metrics.model_dump(  # type: ignore[attr-defined]
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

        # Aggregate system design insights from all batches:
        # 1. drop the default "no concerns" placeholder
        # 2. collapse near-duplicate paraphrases that concurrent batches emit
        default_insight = "No architectural concerns identified"
        meaningful_insights = [
            insight for insight in all_design_insights if insight != default_insight
        ]
        before_insight_dedupe = len(meaningful_insights)
        meaningful_insights = _dedupe_design_insights(meaningful_insights)
        insights_collapsed = before_insight_dedupe - len(meaningful_insights)
        aggregated_insights = (
            "\n\n".join(meaningful_insights)
            if meaningful_insights
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
        con.print("\n[cyan]💰 Token Usage & Cost Estimate:[/cyan]")
        con.print(f"   Input tokens:  {analyzer.provider.total_input_tokens:,}")
        con.print(f"   Output tokens: {analyzer.provider.total_output_tokens:,}")
        con.print(
            f"   Total tokens:  {analyzer.provider.total_input_tokens + analyzer.provider.total_output_tokens:,}"
        )

        # Calculate cost using model's pricing
        input_price = float(pricing["input_price_per_million"])
        output_price = float(pricing["output_price_per_million"])
        if _is_pricing_tbd(input_price, output_price):
            con.print(
                "   [bold]Estimated cost: TBD[/bold] "
                "[dim](provider has not published pricing yet)[/dim]"
            )
        else:
            input_cost = (
                analyzer.provider.total_input_tokens / 1_000_000
            ) * input_price
            output_cost = (
                analyzer.provider.total_output_tokens / 1_000_000
            ) * output_price
            total_cost = input_cost + output_cost
            con.print(f"   [bold]Estimated cost: ${total_cost:.4f}[/bold]")
        con.print()

        # Warn if any files were skipped
        if analyzer.skipped_files:
            con.print(
                f"[yellow]⚠️  Warning: {len(analyzer.skipped_files)} file(s) could not be read:[/yellow]"
            )
            for skipped_file, error in analyzer.skipped_files[:5]:  # Show first 5
                con.print(f"   • {skipped_file}: {error}")
            if len(analyzer.skipped_files) > 5:
                con.print(f"   ... and {len(analyzer.skipped_files) - 5} more")
            con.print()

        # Step 5: Render results (with severity filtering)
        if not quiet:
            renderer = TerminalRenderer(console=con)
            renderer.render(final_report, min_severity=severity)

        # Display total elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time >= 60:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            con.print(f"[dim]⏱️  Completed in {minutes}m {seconds}s[/dim]\n")
        else:
            con.print(f"[dim]⏱️  Completed in {elapsed_time:.1f}s[/dim]\n")

        # Step 6: Export report if requested
        if output:
            try:
                if output_format.lower() == "json":
                    # Export as JSON for programmatic consumption
                    output.write_text(final_report.model_dump_json(indent=2))  # type: ignore[attr-defined]
                    con.print(f"\n[green]✓ JSON report exported to: {output}[/green]\n")
                else:
                    # Export as Markdown (default)
                    # Collect all skipped files for the report.
                    # Three sources of skips, all surfaced so reports are
                    # honest about what was *not* analyzed:
                    #   - scanner: extension/pattern/size filters
                    #   - batcher: token budget exceeded by single file
                    #   - analyzer: read failure mid-batch
                    all_skipped_files: list[tuple[str, str]] = []

                    for skipped_path, reason in scanner.skipped_files:
                        all_skipped_files.append((str(skipped_path), reason))

                    for skipped_path, est_tokens in batcher.skipped_oversized:
                        all_skipped_files.append(
                            (
                                str(skipped_path),
                                f"Exceeds per-batch token budget (~{est_tokens:,} tokens)",
                            )
                        )

                    all_skipped_files.extend(analyzer.skipped_files)

                    # Bundle the run's post-processing signals so the markdown
                    # report is self-documenting beyond the terminal session.
                    linter_tools_with_issues = (
                        sum(
                            1
                            for r in static_results.values()
                            if not r.passed or r.issues_count > 0
                        )
                        if static_results
                        else 0
                    )
                    audit = {
                        "linter_tools_injected": linter_tools_with_issues,
                        "issues_deduplicated": before_dedupe - len(all_issues),
                        "design_insights_deduplicated": insights_collapsed,
                        "drift": drift,
                        "languages_in_batches": sorted(
                            detect_languages_from_paths(str(p) for p in files)
                        ),
                    }

                    exporter = MarkdownExporter()
                    exporter.export(
                        final_report,
                        output,
                        skipped_files=all_skipped_files if all_skipped_files else None,
                        audit=audit,
                    )
                    con.print(f"\n[green]✓ Report exported to: {output}[/green]\n")
            except OSError as e:
                con.print(f"\n[red]✗ Failed to write report to {output}: {e}[/red]\n")
                raise click.Abort()

    except NoCredentialsError:
        con.print("\n[red]✗ AWS credentials not found[/red]\n")
        con.print("[yellow]Please configure AWS credentials:[/yellow]")
        con.print("  1. Run: aws configure")
        con.print(
            "  2. Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
        )
        con.print("  3. Or use --aws-profile flag\n")
        raise click.Abort()

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_msg = e.response.get("Error", {}).get("Message", "")

        con.print(
            f"\n[red]✗ AWS Error ({escape(error_code)}): {escape(error_msg)}[/red]\n"
        )

        if error_code == "AccessDeniedException":
            # model_display_name may be unbound if resolve_model itself raised;
            # fall back to the raw CLI argument.
            model_label = locals().get("model_display_name", model_name)
            con.print("[yellow]Troubleshooting:[/yellow]")
            con.print("  1. Ensure you have access to AWS Bedrock in your region")
            con.print(f"  2. Check that {escape(model_label)} model access is enabled")
            con.print(
                "  3. Verify your IAM permissions include 'bedrock:InvokeModel'\n"
            )
        elif error_code == "ResourceNotFoundException":
            con.print(
                "[yellow]The requested model may not be available in your AWS region.[/yellow]"
            )
            con.print("Check that the model is enabled in your AWS Bedrock settings.\n")

        if verbose:
            con.print(traceback.format_exc())
        raise click.Abort()

    except Exception as e:
        con.print(f"\n[red]✗ Error: {escape(str(e))}[/red]\n")
        if verbose:
            con.print(traceback.format_exc())
        raise click.Abort()

    finally:
        # Ensure callback handlers are cleaned up
        if streaming_handler:
            streaming_handler.cleanup()
        if progress_handler:
            progress_handler.cleanup()


def _dedupe_issues(issues: list[ReviewIssue]) -> list[ReviewIssue]:
    """Drop near-duplicate issues across batches.

    Concurrent batches can independently surface the same architectural concern
    (same bare except, same missing timeout) for the same file. The prompt's
    "report once" rule only applies within a batch, so we collapse duplicates
    at aggregation time, keyed on (file_path, line_start, normalized title).

    Title normalization is deliberately coarse — lowercased, alphanumerics only —
    so "Bare except clause" and "Bare `except:` clause" collapse together.
    Highest-severity issue wins on tie.
    """
    severity_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}

    def fingerprint(issue: ReviewIssue) -> tuple[str, int, str]:
        normalized = "".join(c for c in issue.title.lower() if c.isalnum())
        return (issue.file_path, issue.line_start, normalized)

    best: dict[tuple[str, int, str], ReviewIssue] = {}
    for issue in issues:
        key = fingerprint(issue)
        existing = best.get(key)
        if existing is None or severity_rank.get(
            issue.severity, 99
        ) < severity_rank.get(existing.severity, 99):
            best[key] = issue
    return list(best.values())


def _dedupe_design_insights(insights: list[str]) -> list[str]:
    """Drop near-duplicate architectural insights from concurrent batches.

    Each batch independently emits `system_design_insights`, so on a 5-batch
    run you can see the same observation ("no centralized error handling",
    "providers share a base class but ...") repeated five times. Collapse on
    the first 120 alphanumeric chars — coarse enough to fuse paraphrases,
    fine enough to keep genuinely different observations apart.
    """
    seen: set[str] = set()
    out: list[str] = []
    for text in insights:
        normalized = "".join(c.lower() for c in text if c.isalnum())[:120]
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(text)
    return out


def _generate_recommendations(issues: list[ReviewIssue]) -> list[str]:
    """Generate top recommendations based on issue severity and category distribution.

    Args:
        issues: List of ReviewIssue objects from analysis

    Returns:
        Up to 5 prioritized recommendations based on severity and category counts
    """
    critical = [i for i in issues if i.severity == "Critical"]
    high = [i for i in issues if i.severity == "High"]

    recommendations = []

    if critical:
        recommendations.append(
            f"🚨 Address {len(critical)} critical issue(s) immediately"
        )

    if high:
        recommendations.append(f"⚠️  Fix {len(high)} high-priority issue(s)")

    # Category-based recommendations (ordered by impact)
    category_configs = [
        ("Security", "🔒 Resolve {n} security issue(s)"),
        ("Performance", "⚡ Investigate {n} performance issue(s)"),
        ("Testing", "🧪 Address {n} testing issue(s)"),
        ("Code Quality", "🔧 Review {n} code quality issue(s)"),
        ("System Design", "🏗️ Review {n} system design issue(s)"),
    ]
    for category, template in category_configs:
        if len(recommendations) >= 5:
            break
        count = sum(1 for i in issues if i.category == category)
        if count:
            recommendations.append(template.format(n=count))

    return recommendations[:5]


def _render_dry_run(
    files: list[Path],
    batches: list[FileBatch],
    model_display_name: str,
    provider: ModelProvider,
    console: Console,
) -> None:
    """Render dry-run output showing files, validation, and estimated costs.

    Args:
        files: List of file paths to analyze
        batches: List of FileBatch objects
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
        tokens = FileBatcher.estimate_file_tokens(file_path)
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
    pricing_tbd = _is_pricing_tbd(input_price, output_price)

    # Summary
    console.print("[bold]💰 Estimated Cost Summary[/bold]")
    console.print(f"   Model: {model_display_name}")
    console.print(f"   Files: {len(files)}")
    console.print(f"   Batches: {len(batches)}")
    console.print(f"   Est. input tokens: ~{total_input_tokens:,}")
    console.print(f"   Est. output tokens: ~{estimated_output_tokens:,}")
    if pricing_tbd:
        console.print(
            "   [bold]Est. cost: TBD[/bold] "
            "[dim](provider has not published pricing yet)[/dim]"
        )
    else:
        input_cost = (total_input_tokens / 1_000_000) * input_price
        output_cost = (estimated_output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost
        console.print(f"   [bold]Est. cost: ${total_cost:.4f}[/bold]")
        console.print(f"      (Input: ${input_cost:.4f} @ ${input_price}/M)")
        console.print(f"      (Output: ${output_cost:.4f} @ ${output_price}/M)")
    console.print()
    console.print("[dim]Run without --dry-run to perform the actual analysis.[/dim]")


if __name__ == "__main__":
    main()
