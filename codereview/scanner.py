# codereview/scanner.py
"""File scanner for discovering code files to review."""
from pathlib import Path
from fnmatch import fnmatch
from typing import List
from codereview.config import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_EXTENSIONS,
    MAX_FILE_SIZE_KB
)


class FileScanner:
    """Scans directory for Python and Go files to review."""

    def __init__(
        self,
        root_dir: Path | str,
        exclude_patterns: List[str] | None = None,
        max_file_size_kb: int = MAX_FILE_SIZE_KB
    ):
        self.root_dir = Path(root_dir)
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
        self.max_file_size_kb = max_file_size_kb

    def scan(self) -> List[Path]:
        """Scan directory and return list of files to review."""
        target_extensions = {".py", ".go"}
        files = []

        for file_path in self.root_dir.rglob("*"):
            # Skip if not a file
            if not file_path.is_file():
                continue

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

            # Skip if file too large
            file_size_kb = file_path.stat().st_size / 1024
            if file_size_kb > self.max_file_size_kb:
                continue

            files.append(file_path)

        return sorted(files)

    def _is_excluded(self, path: str) -> bool:
        """Check if path matches any exclusion pattern."""
        for pattern in self.exclude_patterns:
            if fnmatch(path, pattern):
                return True
        return False
