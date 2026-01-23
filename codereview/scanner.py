# codereview/scanner.py
"""File scanner for discovering code files to review."""

from pathlib import Path, PurePath
from typing import List, Tuple

from codereview.config import (
    DEFAULT_EXCLUDE_EXTENSIONS,
    DEFAULT_EXCLUDE_PATTERNS,
    MAX_FILE_SIZE_KB,
)


class FileScanner:
    """Scans directory for Python, Go, Shell, C++, Java, JavaScript, and TypeScript files to review."""

    def __init__(
        self,
        root_dir: Path | str,
        exclude_patterns: List[str] | None = None,
        max_file_size_kb: int = MAX_FILE_SIZE_KB,
    ) -> None:
        """
        Initialize file scanner.

        Args:
            root_dir: Root directory to scan for code files
            exclude_patterns: Glob patterns to exclude (defaults to DEFAULT_EXCLUDE_PATTERNS)
            max_file_size_kb: Maximum file size in KB to include (larger files are skipped)
        """
        self.root_dir = Path(root_dir)
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
        self.max_file_size_kb = max_file_size_kb
        self.skipped_files: List[Tuple[Path, str]] = (
            []
        )  # Track skipped files with reasons

    def scan(self) -> List[Path]:
        """Scan directory and return list of files to review."""
        target_extensions = {
            # Python
            ".py",
            # Go
            ".go",
            # Shell
            ".sh",
            ".bash",
            # C++
            ".cpp",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
            # Java
            ".java",
            # JavaScript
            ".js",
            ".jsx",
            ".mjs",
            # TypeScript
            ".ts",
            ".tsx",
        }
        files = []
        self.skipped_files = []  # Reset on each scan

        # Resolve root directory once for path traversal protection
        resolved_root = self.root_dir.resolve()

        for file_path in self.root_dir.rglob("*"):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Path traversal protection: ensure file is within root directory
            # This prevents symlink attacks that could escape the scan directory
            try:
                resolved_path = file_path.resolve()
                if not resolved_path.is_relative_to(resolved_root):
                    continue  # Skip files outside root directory
            except (OSError, ValueError):
                continue  # Skip files that can't be resolved

            # Skip excluded extensions
            if file_path.suffix in DEFAULT_EXCLUDE_EXTENSIONS:
                continue

            # Skip if not target language
            if file_path.suffix not in target_extensions:
                continue

            # Skip excluded patterns
            relative_path = file_path.relative_to(self.root_dir)
            if self._is_excluded(str(relative_path)):
                continue

            # Skip if file too large (but track it)
            file_size_kb = file_path.stat().st_size / 1024
            if file_size_kb > self.max_file_size_kb:
                self.skipped_files.append(
                    (
                        file_path,
                        f"File too large: {file_size_kb:.1f}KB > {self.max_file_size_kb}KB",
                    )
                )
                continue

            files.append(file_path)

        return sorted(files)

    def _is_excluded(self, path: str) -> bool:
        """Check if path matches any exclusion pattern.

        Uses PurePath.match() which properly handles ** glob patterns.
        """
        pure_path = PurePath(path)
        for pattern in self.exclude_patterns:
            if pure_path.match(pattern):
                return True
        return False
