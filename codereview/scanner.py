"""File scanner for discovering code files to review."""

import logging
import os
from pathlib import Path, PurePath

from codereview.config import (
    DEFAULT_EXCLUDE_EXTENSIONS,
    DEFAULT_EXCLUDE_PATTERNS,
    MAX_FILE_SIZE_KB,
)


class FileScanner:
    """Scans directory for Python, Go, Shell, C++, Java, JavaScript, and TypeScript files to review."""

    # Supported file extensions for code review
    TARGET_EXTENSIONS: frozenset[str] = frozenset(
        {
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
    )

    def __init__(
        self,
        root_dir: Path | str,
        exclude_patterns: list[str] | None = None,
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
        self.exclude_patterns = (
            DEFAULT_EXCLUDE_PATTERNS if exclude_patterns is None else exclude_patterns
        )
        self.max_file_size_kb = max_file_size_kb
        self.skipped_files: list[
            tuple[Path, str]
        ] = []  # Track skipped files with reasons

    def scan(self) -> list[Path]:
        """Scan directory and return list of files to review.

        Uses os.walk with in-place directory pruning to avoid traversing
        into excluded directories (e.g., node_modules, .venv, __pycache__).
        """
        files = []
        self.skipped_files = []  # Reset on each scan

        # Resolve root directory once for path traversal protection
        resolved_root = self.root_dir.resolve()

        # Precompute directory-level exclusion names from glob patterns
        # Patterns like "**/node_modules/**" or "**/.venv/**" indicate
        # entire directories to skip
        excluded_dir_names = self._get_excluded_dir_names()

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Prune excluded directories in-place to prevent traversal
            dirnames[:] = [
                d
                for d in dirnames
                if d not in excluded_dir_names and not d.startswith(".")
            ]

            for filename in filenames:
                file_path = Path(dirpath) / filename

                # Path traversal protection: ensure file is within root directory
                try:
                    resolved_path = file_path.resolve()
                    if not resolved_path.is_relative_to(resolved_root):
                        logging.debug("Skipping file outside root: %s", file_path)
                        continue
                except OSError as e:
                    logging.debug("Skipping file due to OSError: %s - %s", file_path, e)
                    continue
                except ValueError as e:
                    logging.debug(
                        "Skipping file due to ValueError: %s - %s", file_path, e
                    )
                    continue

                # Skip excluded extensions
                if file_path.suffix in DEFAULT_EXCLUDE_EXTENSIONS:
                    continue

                # Skip if not target language
                if file_path.suffix not in self.TARGET_EXTENSIONS:
                    continue

                # Skip excluded patterns (for fine-grained glob matching)
                relative_path = file_path.relative_to(self.root_dir)
                if self._is_excluded(str(relative_path)):
                    continue

                # Skip if file too large (but track it)
                try:
                    file_size_kb = file_path.stat().st_size / 1024
                except OSError:
                    continue
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

    def _get_excluded_dir_names(self) -> set[str]:
        """Extract directory names that should be pruned from exclusion patterns.

        Parses glob patterns like '**/node_modules/**' to extract 'node_modules'
        as a directory name to skip during os.walk traversal.

        Returns:
            Set of directory base names to prune
        """
        dir_names: set[str] = set()
        for pattern in self.exclude_patterns:
            # Match patterns like "**/dirname/**" or "**/dirname/*"
            parts = PurePath(pattern).parts
            for part in parts:
                if part != "**" and part != "*" and "*" not in part:
                    dir_names.add(part)
        return dir_names

    def _is_excluded(self, path: str) -> bool:
        """Check if path matches any exclusion pattern.

        Uses PurePath.match() which properly handles ** glob patterns.
        """
        pure_path = PurePath(path)
        for pattern in self.exclude_patterns:
            if pure_path.match(pattern):
                return True
        return False
