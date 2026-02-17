"""Tests for README finder functionality."""

from pathlib import Path


class TestFindReadme:
    """Tests for find_readme function."""

    def test_finds_readme_in_target_directory(self, tmp_path: Path) -> None:
        """Test that find_readme finds README.md in the target directory."""
        from codereview.readme_finder import find_readme

        # Create README.md in target directory
        readme = tmp_path / "README.md"
        readme.write_text("# Test Project")

        result = find_readme(tmp_path)

        assert result is not None
        assert result == readme
        assert result.is_file()

    def test_finds_readme_in_parent_directory(self, tmp_path: Path) -> None:
        """Test that find_readme finds README.md in parent when not in target."""
        from codereview.readme_finder import find_readme

        # Create README.md in parent directory
        readme = tmp_path / "README.md"
        readme.write_text("# Parent Project")

        # Create a subdirectory without README
        subdir = tmp_path / "src" / "module"
        subdir.mkdir(parents=True)

        result = find_readme(subdir)

        assert result is not None
        assert result == readme

    def test_returns_none_when_no_readme(self, tmp_path: Path) -> None:
        """Test that find_readme returns None when no README exists."""
        from codereview.readme_finder import find_readme

        # Create a directory structure without any README
        subdir = tmp_path / "project" / "src"
        subdir.mkdir(parents=True)

        # Add a .git directory at root to stop the search
        git_dir = tmp_path / "project" / ".git"
        git_dir.mkdir()

        result = find_readme(subdir)

        assert result is None

    def test_stops_at_git_root(self, tmp_path: Path) -> None:
        """Test that find_readme stops at .git directory and doesn't search beyond."""
        from codereview.readme_finder import find_readme

        # Create README.md ABOVE the git root (should not be found)
        readme_above = tmp_path / "README.md"
        readme_above.write_text("# Above Git Root")

        # Create a git repository subdirectory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        # Create a subdirectory inside the repo
        src_dir = repo_dir / "src"
        src_dir.mkdir()

        result = find_readme(src_dir)

        # Should return None because it stops at git root, not find the README above
        assert result is None

    def test_finds_first_readme_going_up(self, tmp_path: Path) -> None:
        """Test that find_readme returns the closest README when multiple exist."""
        from codereview.readme_finder import find_readme

        # Create README.md in root
        root_readme = tmp_path / "README.md"
        root_readme.write_text("# Root Project")

        # Create subdirectory with its own README
        subdir = tmp_path / "packages" / "core"
        subdir.mkdir(parents=True)
        sub_readme = subdir / "README.md"
        sub_readme.write_text("# Core Package")

        # Create a deeper directory without README
        deep_dir = subdir / "src" / "utils"
        deep_dir.mkdir(parents=True)

        # Search from deep_dir should find sub_readme first
        result = find_readme(deep_dir)

        assert result is not None
        assert result == sub_readme
        assert "Core Package" in result.read_text()


class TestReadReadmeContent:
    """Tests for read_readme_content function."""

    def test_reads_readme_content(self, tmp_path: Path) -> None:
        """Test that read_readme_content reads content and returns (content, size)."""
        from codereview.readme_finder import read_readme_content

        # Create a README file
        readme = tmp_path / "README.md"
        content = "# My Project\n\nThis is a test README."
        readme.write_text(content)

        result = read_readme_content(readme)

        assert result is not None
        returned_content, file_size = result
        assert returned_content == content
        assert file_size == len(content.encode("utf-8"))

    def test_truncates_large_readme(self, tmp_path: Path) -> None:
        """Test that read_readme_content truncates when > max_size."""
        from codereview.readme_finder import read_readme_content

        # Create a large README file
        readme = tmp_path / "README.md"
        large_content = "# Large README\n" + "x" * 1000
        readme.write_text(large_content)

        # Use a small max_size to trigger truncation
        result = read_readme_content(readme, max_size=100)

        assert result is not None
        returned_content, file_size = result
        assert len(returned_content) < len(large_content)
        assert "[TRUNCATED" in returned_content
        assert file_size == len(large_content.encode("utf-8"))

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Test that read_readme_content returns None for non-existent file."""
        from codereview.readme_finder import read_readme_content

        # Non-existent file
        missing = tmp_path / "NONEXISTENT.md"

        result = read_readme_content(missing)

        assert result is None

    def test_returns_none_for_binary_file(self, tmp_path: Path) -> None:
        """Test that read_readme_content returns None for binary content (null bytes)."""
        from codereview.readme_finder import read_readme_content

        # Create a file with binary content (null bytes)
        readme = tmp_path / "README.md"
        binary_content = b"# README\x00\x00Binary data"
        readme.write_bytes(binary_content)

        result = read_readme_content(readme)

        assert result is None


class TestPromptReadmeConfirmation:
    """Tests for prompt_readme_confirmation function."""

    def test_returns_path_on_yes(self, tmp_path: Path) -> None:
        """Test that prompt_readme_confirmation returns path when user confirms."""
        from unittest.mock import patch

        from codereview.readme_finder import prompt_readme_confirmation

        # Create a README file
        readme = tmp_path / "README.md"
        readme.write_text("# Test Project")

        # Mock isatty to return True so _timed_input is called
        with patch("codereview.readme_finder.sys.stdin.isatty", return_value=True):
            # Test with empty string (default yes)
            with patch("codereview.readme_finder._timed_input", return_value=""):
                result = prompt_readme_confirmation(readme)
                assert result == readme

            # Test with explicit "y"
            with patch("codereview.readme_finder._timed_input", return_value="y"):
                result = prompt_readme_confirmation(readme)
                assert result == readme

            # Test with "Y" (uppercase)
            with patch("codereview.readme_finder._timed_input", return_value="Y"):
                result = prompt_readme_confirmation(readme)
                assert result == readme

    def test_returns_none_on_no(self, tmp_path: Path) -> None:
        """Test that prompt_readme_confirmation returns None when user declines."""
        from unittest.mock import patch

        from codereview.readme_finder import prompt_readme_confirmation

        # Create a README file
        readme = tmp_path / "README.md"
        readme.write_text("# Test Project")

        # Mock isatty to return True so _timed_input is called
        with patch("codereview.readme_finder.sys.stdin.isatty", return_value=True):
            # Test with "n"
            with patch("codereview.readme_finder._timed_input", return_value="n"):
                result = prompt_readme_confirmation(readme)
                assert result is None

            # Test with "N" (uppercase)
            with patch("codereview.readme_finder._timed_input", return_value="N"):
                result = prompt_readme_confirmation(readme)
                assert result is None

    def test_returns_custom_path(self, tmp_path: Path) -> None:
        """Test that prompt_readme_confirmation returns custom path when specified."""
        from unittest.mock import patch

        from codereview.readme_finder import prompt_readme_confirmation

        # Create original README
        readme = tmp_path / "README.md"
        readme.write_text("# Original")

        # Create an alternative file
        custom = tmp_path / "CONTEXT.md"
        custom.write_text("# Custom Context")

        # Mock isatty to return True so _timed_input is called
        with patch("codereview.readme_finder.sys.stdin.isatty", return_value=True):
            # User specifies custom path
            with patch(
                "codereview.readme_finder._timed_input", return_value=str(custom)
            ):
                result = prompt_readme_confirmation(readme)
                assert result == custom

    def test_returns_none_for_invalid_custom_path(self, tmp_path: Path) -> None:
        """Test that prompt_readme_confirmation returns None for invalid custom path."""
        from unittest.mock import patch

        from codereview.readme_finder import prompt_readme_confirmation

        # Create original README
        readme = tmp_path / "README.md"
        readme.write_text("# Original")

        # Mock isatty to return True so _timed_input is called
        with patch("codereview.readme_finder.sys.stdin.isatty", return_value=True):
            # User specifies non-existent path
            with patch(
                "codereview.readme_finder._timed_input",
                return_value="/nonexistent/file.md",
            ):
                result = prompt_readme_confirmation(readme)
                assert result is None

    def test_prompts_for_path_when_none_found(self, tmp_path: Path) -> None:
        """Test that prompt_readme_confirmation asks for path when no README found."""
        from unittest.mock import patch

        from codereview.readme_finder import prompt_readme_confirmation

        # Create a context file that user can specify
        context_file = tmp_path / "CONTEXT.md"
        context_file.write_text("# Context")

        # User specifies a file when prompted (simulate interactive terminal)
        with (
            patch("codereview.readme_finder.sys.stdin") as mock_stdin,
            patch(
                "codereview.readme_finder.click.prompt",
                return_value=str(context_file),
            ),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt_readme_confirmation(None)
            assert result == context_file

    def test_returns_none_when_no_readme_and_user_skips(self) -> None:
        """Test that prompt_readme_confirmation returns None when no README and user skips."""
        from unittest.mock import patch

        from codereview.readme_finder import prompt_readme_confirmation

        # Test with empty string (default skip, interactive terminal)
        with (
            patch("codereview.readme_finder.sys.stdin") as mock_stdin,
            patch("codereview.readme_finder.click.prompt", return_value=""),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt_readme_confirmation(None)
            assert result is None

        # Test with "n"
        with (
            patch("codereview.readme_finder.sys.stdin") as mock_stdin,
            patch("codereview.readme_finder.click.prompt", return_value="n"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt_readme_confirmation(None)
            assert result is None

        # Test with "N" (uppercase)
        with (
            patch("codereview.readme_finder.sys.stdin") as mock_stdin,
            patch("codereview.readme_finder.click.prompt", return_value="N"),
        ):
            mock_stdin.isatty.return_value = True
            result = prompt_readme_confirmation(None)
            assert result is None

    def test_auto_confirms_in_non_interactive_mode(self, tmp_path: Path) -> None:
        """Test that prompt_readme_confirmation auto-confirms when stdin is not a tty."""
        from unittest.mock import patch

        from codereview.readme_finder import prompt_readme_confirmation

        # Create a README file
        readme = tmp_path / "README.md"
        readme.write_text("# Test Project")

        # When stdin.isatty() returns False, should auto-confirm with "Y"
        with patch("codereview.readme_finder.sys.stdin.isatty", return_value=False):
            result = prompt_readme_confirmation(readme)
            assert result == readme
