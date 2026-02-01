"""README finder for project context discovery."""

from pathlib import Path


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
