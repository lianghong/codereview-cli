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
