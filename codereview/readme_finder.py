"""README finder for project context discovery."""

from pathlib import Path

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

    except (OSError, UnicodeDecodeError):
        return None
