"""README finder for project context discovery."""

from pathlib import Path

import click
from rich.console import Console

# Size limits for README files
README_WARN_SIZE = 50 * 1024  # 50KB - warn user
README_MAX_SIZE = 100 * 1024  # 100KB - truncate


def find_readme(target_dir: Path) -> Path | None:
    """Find README.md starting from target directory and searching upward.

    Searches target directory first, then parent directories until:
    - README.md is found
    - Git repository root is reached (.git directory found)
    - Filesystem root is reached

    Args:
        target_dir: Directory to start searching from

    Returns:
        Path to README.md if found, None otherwise
    """
    current = target_dir.resolve()

    while True:
        # Check for README.md in current directory
        readme = current / "README.md"
        if readme.is_file():
            return readme

        # Check if we've reached git root (stop here)
        git_dir = current / ".git"
        if git_dir.is_dir():
            return None

        # Check if we've reached filesystem root
        parent = current.parent
        if parent == current:
            return None

        current = parent


def read_readme_content(
    readme_path: Path, max_size: int = README_MAX_SIZE
) -> tuple[str, int] | None:
    """Read README content with size limits.

    Args:
        readme_path: Path to README file
        max_size: Maximum content size in bytes (truncates if larger)

    Returns:
        Tuple of (content, original_size) if readable, None otherwise
    """
    if not readme_path.is_file():
        return None

    try:
        # Get file size first
        file_size = readme_path.stat().st_size

        # Read content
        content = readme_path.read_text(encoding="utf-8")

        # Check for binary content (null bytes indicate binary)
        if "\x00" in content:
            return None

        # Truncate if too large
        if len(content) > max_size:
            truncate_at = max_size - 100  # Leave room for truncation message
            content = (
                content[:truncate_at] + "\n\n[TRUNCATED - README exceeded size limit]"
            )

        return content, file_size

    except OSError, UnicodeDecodeError:
        return None


def prompt_readme_confirmation(
    readme_path: Path | None,
    console: Console | None = None,
) -> Path | None:
    """Prompt user to confirm or specify README file for project context.

    If a README path is provided, asks user to confirm its use or specify an alternative.
    If no README path is provided, asks user to optionally specify a file.

    Args:
        readme_path: Path to found README.md, or None if not found
        console: Rich Console for output (creates one if not provided)

    Returns:
        Path to README/context file if confirmed, None if user declines
    """
    if console is None:
        console = Console()

    if readme_path is not None:
        result = read_readme_content(readme_path)
        if result is None:
            console.print(f"[yellow]Warning: Could not read {readme_path}[/yellow]")
            readme_path = None
        else:
            content, size = result
            size_kb = size / 1024
            console.print(
                f"[dim]Found README:[/dim] [cyan]{readme_path}[/cyan] ({size_kb:.1f} KB)"
            )

            if size > README_WARN_SIZE:
                console.print(
                    "[yellow]   Warning: Large file - may use significant tokens[/yellow]"
                )

            response = click.prompt(
                "   Use this file for project context? [Y/n/path]",
                default="",
                show_default=False,
            )

            response_stripped = response.strip()
            response_lower = response_stripped.lower()
            if response_lower == "" or response_lower == "y":
                return readme_path
            elif response_lower == "n":
                return None
            else:
                # User specified a custom path - preserve original case
                custom_path = Path(response_stripped).expanduser().resolve()
                if custom_path.is_file():
                    return custom_path
                else:
                    console.print(f"[red]File not found: {response_stripped}[/red]")
                    return None

    # No README found (either initially or after read failure)
    if readme_path is None:
        console.print("[dim]No README.md found in target or parent directories[/dim]")
        response = click.prompt(
            "   Specify a file for project context? [path/N]",
            default="",
            show_default=False,
        )

        response = response.strip()
        if response == "" or response.lower() == "n":
            return None

        custom_path = Path(response).expanduser().resolve()
        if custom_path.is_file():
            return custom_path
        else:
            console.print(f"[red]File not found: {response}[/red]")
            return None

    return None
