# README Context Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add project README.md as context for AI code reviews with auto-discovery and user confirmation.

**Architecture:** New `readme_finder.py` module handles discovery and prompts. README content flows through `CodeAnalyzer` to providers via `project_context` parameter. Base provider prepends README to batch context.

**Tech Stack:** Python 3.14, Click (CLI), Rich (console output), pathlib

---

### Task 1: Create readme_finder module with find_readme function

**Files:**
- Create: `codereview/readme_finder.py`
- Create: `tests/test_readme_finder.py`

**Step 1: Write the failing test for find_readme**

```python
"""Tests for readme_finder module."""

import tempfile
from pathlib import Path

import pytest

from codereview.readme_finder import find_readme


class TestFindReadme:
    """Tests for find_readme function."""

    def test_finds_readme_in_target_directory(self, tmp_path: Path) -> None:
        """Should find README.md in the target directory."""
        readme = tmp_path / "README.md"
        readme.write_text("# Project")

        result = find_readme(tmp_path)

        assert result == readme

    def test_finds_readme_in_parent_directory(self, tmp_path: Path) -> None:
        """Should find README.md in parent when not in target."""
        subdir = tmp_path / "src"
        subdir.mkdir()
        readme = tmp_path / "README.md"
        readme.write_text("# Project")

        result = find_readme(subdir)

        assert result == readme

    def test_returns_none_when_no_readme(self, tmp_path: Path) -> None:
        """Should return None when no README.md found."""
        result = find_readme(tmp_path)

        assert result is None

    def test_stops_at_git_root(self, tmp_path: Path) -> None:
        """Should stop searching at git repository root."""
        # Create git repo structure
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        subdir = tmp_path / "src" / "components"
        subdir.mkdir(parents=True)

        # README outside git root should not be found
        # (tmp_path is the git root, so search stops there)
        result = find_readme(subdir)

        assert result is None

    def test_finds_first_readme_going_up(self, tmp_path: Path) -> None:
        """Should return first README found when multiple exist."""
        # Create nested structure with multiple READMEs
        subdir = tmp_path / "src"
        subdir.mkdir()
        readme_root = tmp_path / "README.md"
        readme_root.write_text("# Root")
        readme_src = subdir / "README.md"
        readme_src.write_text("# Src")

        result = find_readme(subdir)

        assert result == readme_src  # Finds closest one first
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_readme_finder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'codereview.readme_finder'`

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_readme_finder.py::TestFindReadme -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add codereview/readme_finder.py tests/test_readme_finder.py
git commit -m "feat(readme): add find_readme function for README discovery"
```

---

### Task 2: Add read_readme_content helper function

**Files:**
- Modify: `codereview/readme_finder.py`
- Modify: `tests/test_readme_finder.py`

**Step 1: Write the failing test**

```python
from codereview.readme_finder import find_readme, read_readme_content


class TestReadReadmeContent:
    """Tests for read_readme_content function."""

    def test_reads_readme_content(self, tmp_path: Path) -> None:
        """Should read and return README content."""
        readme = tmp_path / "README.md"
        readme.write_text("# My Project\n\nDescription here.")

        content, size = read_readme_content(readme)

        assert content == "# My Project\n\nDescription here."
        assert size == len(content)

    def test_truncates_large_readme(self, tmp_path: Path) -> None:
        """Should truncate README larger than max_size."""
        readme = tmp_path / "README.md"
        large_content = "x" * 200_000  # 200KB
        readme.write_text(large_content)

        content, size = read_readme_content(readme, max_size=100_000)

        assert len(content) < 100_000
        assert "[TRUNCATED" in content
        assert size == 200_000  # Original size reported

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Should return None for non-existent file."""
        readme = tmp_path / "README.md"

        result = read_readme_content(readme)

        assert result is None

    def test_returns_none_for_binary_file(self, tmp_path: Path) -> None:
        """Should return None for binary files."""
        readme = tmp_path / "README.md"
        readme.write_bytes(b"\x00\x01\x02\x03")

        result = read_readme_content(readme)

        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_readme_finder.py::TestReadReadmeContent -v`
Expected: FAIL with `ImportError: cannot import name 'read_readme_content'`

**Step 3: Write implementation**

Add to `codereview/readme_finder.py`:

```python
# Size limits for README files
README_WARN_SIZE = 50 * 1024  # 50KB - warn user
README_MAX_SIZE = 100 * 1024  # 100KB - truncate


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
            content = content[:truncate_at] + "\n\n[TRUNCATED - README exceeded size limit]"

        return content, file_size

    except (OSError, UnicodeDecodeError):
        return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_readme_finder.py::TestReadReadmeContent -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add codereview/readme_finder.py tests/test_readme_finder.py
git commit -m "feat(readme): add read_readme_content with size limits"
```

---

### Task 3: Add prompt_readme_confirmation function

**Files:**
- Modify: `codereview/readme_finder.py`
- Modify: `tests/test_readme_finder.py`

**Step 1: Write the failing test**

```python
from unittest.mock import patch

from codereview.readme_finder import (
    find_readme,
    read_readme_content,
    prompt_readme_confirmation,
)


class TestPromptReadmeConfirmation:
    """Tests for prompt_readme_confirmation function."""

    def test_returns_path_on_yes(self, tmp_path: Path) -> None:
        """Should return path when user confirms with Enter/Y."""
        readme = tmp_path / "README.md"
        readme.write_text("# Project")

        with patch("click.prompt", return_value=""):
            result = prompt_readme_confirmation(readme)

        assert result == readme

    def test_returns_none_on_no(self, tmp_path: Path) -> None:
        """Should return None when user declines."""
        readme = tmp_path / "README.md"
        readme.write_text("# Project")

        with patch("click.prompt", return_value="n"):
            result = prompt_readme_confirmation(readme)

        assert result is None

    def test_returns_custom_path(self, tmp_path: Path) -> None:
        """Should return custom path when user specifies one."""
        readme = tmp_path / "README.md"
        readme.write_text("# Project")
        custom = tmp_path / "CUSTOM.md"
        custom.write_text("# Custom")

        with patch("click.prompt", return_value=str(custom)):
            result = prompt_readme_confirmation(readme)

        assert result == custom

    def test_prompts_for_path_when_none_found(self, tmp_path: Path) -> None:
        """Should ask for path when no README found."""
        custom = tmp_path / "CUSTOM.md"
        custom.write_text("# Custom")

        with patch("click.prompt", return_value=str(custom)):
            result = prompt_readme_confirmation(None)

        assert result == custom

    def test_returns_none_when_no_readme_and_user_skips(self) -> None:
        """Should return None when no README and user skips."""
        with patch("click.prompt", return_value=""):
            result = prompt_readme_confirmation(None)

        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_readme_finder.py::TestPromptReadmeConfirmation -v`
Expected: FAIL with `ImportError: cannot import name 'prompt_readme_confirmation'`

**Step 3: Write implementation**

Add to `codereview/readme_finder.py`:

```python
import click
from rich.console import Console


def prompt_readme_confirmation(
    readme_path: Path | None,
    console: Console | None = None,
) -> Path | None:
    """Prompt user to confirm README usage or specify alternative.

    Args:
        readme_path: Path to found README, or None if not found
        console: Rich console for output (creates default if None)

    Returns:
        Confirmed README path, or None if user skips
    """
    if console is None:
        console = Console()

    if readme_path is not None:
        # README found - ask for confirmation
        result = read_readme_content(readme_path)
        if result is None:
            console.print(f"[yellow]⚠️  Could not read {readme_path}[/yellow]")
            readme_path = None
        else:
            content, size = result
            size_kb = size / 1024
            console.print(f"📄 Found README: [cyan]{readme_path}[/cyan] ({size_kb:.1f} KB)")

            if size > README_WARN_SIZE:
                console.print(
                    f"[yellow]   ⚠️  Large file - may use significant tokens[/yellow]"
                )

            response = click.prompt(
                "   Use this file for project context? [Y/n/path]",
                default="",
                show_default=False,
            )

            response = response.strip().lower()
            if response == "" or response == "y":
                return readme_path
            elif response == "n":
                return None
            else:
                # User provided a path
                custom_path = Path(response).expanduser().resolve()
                if custom_path.is_file():
                    return custom_path
                else:
                    console.print(f"[red]✗ File not found: {response}[/red]")
                    return None

    # No README found - ask if user wants to specify one
    console.print("📄 No README.md found in target or parent directories")
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
        console.print(f"[red]✗ File not found: {response}[/red]")
        return None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_readme_finder.py::TestPromptReadmeConfirmation -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add codereview/readme_finder.py tests/test_readme_finder.py
git commit -m "feat(readme): add prompt_readme_confirmation for user interaction"
```

---

### Task 4: Modify base provider to accept project_context

**Files:**
- Modify: `codereview/providers/base.py`
- Modify: `tests/test_provider_base.py`

**Step 1: Write the failing test**

Add to `tests/test_provider_base.py`:

```python
class TestPrepareContextWithReadme:
    """Tests for _prepare_batch_context with project context."""

    def test_prepends_readme_to_batch_context(self) -> None:
        """Should prepend README content before file contents."""

        class ConcreteProvider(ModelProvider):
            def analyze_batch(self, *args, **kwargs):
                pass

            def get_model_display_name(self):
                return "Test"

            def get_pricing(self):
                return {}

        provider = ConcreteProvider()
        files = {"test.py": "print('hello')"}
        readme = "# My Project\n\nThis is a test project."

        context = provider._prepare_batch_context(
            batch_number=1,
            total_batches=1,
            files_content=files,
            project_context=readme,
        )

        # README should appear before code
        assert "== PROJECT CONTEXT ==" in context
        assert "# My Project" in context
        assert context.index("PROJECT CONTEXT") < context.index("test.py")

    def test_no_readme_section_when_none(self) -> None:
        """Should not include README section when project_context is None."""

        class ConcreteProvider(ModelProvider):
            def analyze_batch(self, *args, **kwargs):
                pass

            def get_model_display_name(self):
                return "Test"

            def get_pricing(self):
                return {}

        provider = ConcreteProvider()
        files = {"test.py": "print('hello')"}

        context = provider._prepare_batch_context(
            batch_number=1,
            total_batches=1,
            files_content=files,
            project_context=None,
        )

        assert "PROJECT CONTEXT" not in context
        assert "test.py" in context
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_base.py::TestPrepareContextWithReadme -v`
Expected: FAIL with `TypeError: _prepare_batch_context() got an unexpected keyword argument 'project_context'`

**Step 3: Write implementation**

Modify `_prepare_batch_context` in `codereview/providers/base.py`:

```python
    def _prepare_batch_context(
        self,
        batch_number: int,
        total_batches: int,
        files_content: dict[str, str],
        project_context: str | None = None,
    ) -> str:
        """Prepare context string for LLM.

        Args:
            batch_number: Current batch number
            total_batches: Total number of batches
            files_content: Dictionary mapping file paths to file contents
            project_context: Optional project README content for background context

        Returns:
            Formatted context string with file contents and line numbers
        """
        lines = []

        # Add project context (README) if provided
        if project_context:
            lines.extend([
                "== PROJECT CONTEXT ==",
                "The following is the project README for background context:",
                "",
                "--- README.md ---",
                project_context,
                "--- END README ---",
                "",
                "== CODE REVIEW ==",
            ])

        lines.extend([
            f"Analyzing Batch {batch_number}/{total_batches}",
            f"Files in this batch: {len(files_content)}",
            "",
            "=" * 80,
            "",
        ])

        for file_path, content in files_content.items():
            lines.append(f"File: {file_path}")
            lines.append("-" * 80)

            # Add line numbers (use extend with generator for efficiency)
            lines.extend(
                f"{i:4d} | {line}"
                for i, line in enumerate(content.splitlines(), start=1)
            )

            lines.append("")
            lines.append("=" * 80)
            lines.append("")

        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provider_base.py::TestPrepareContextWithReadme -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add codereview/providers/base.py tests/test_provider_base.py
git commit -m "feat(provider): add project_context parameter to batch context"
```

---

### Task 5: Update providers to pass project_context through

**Files:**
- Modify: `codereview/providers/bedrock.py`
- Modify: `codereview/providers/azure_openai.py`
- Modify: `codereview/providers/nvidia.py`

**Step 1: Update provider signatures**

Each provider's `__init__` needs to accept and store `project_context`, and `analyze_batch` needs to pass it to `_prepare_batch_context`.

For `codereview/providers/bedrock.py`, modify `__init__`:

```python
    def __init__(
        self,
        model_config: ModelConfig,
        provider_config: BedrockConfig,
        temperature: float | None = None,
        requests_per_second: float = 1.0,
        callbacks: list[BaseCallbackHandler] | None = None,
        enable_output_fixing: bool = True,
        project_context: str | None = None,
    ):
        # ... existing code ...
        self.project_context = project_context
```

And modify `analyze_batch` call to `_prepare_batch_context`:

```python
        batch_context = self._prepare_batch_context(
            batch_number, total_batches, files_content, self.project_context
        )
```

**Step 2: Apply same changes to azure_openai.py and nvidia.py**

Same pattern: add `project_context: str | None = None` parameter to `__init__`, store it, and pass to `_prepare_batch_context`.

**Step 3: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_bedrock_provider.py tests/test_azure_provider.py tests/test_nvidia_provider.py -v`
Expected: All tests PASS (existing tests don't pass project_context, defaults to None)

**Step 4: Commit**

```bash
git add codereview/providers/bedrock.py codereview/providers/azure_openai.py codereview/providers/nvidia.py
git commit -m "feat(providers): add project_context parameter to all providers"
```

---

### Task 6: Update ProviderFactory to accept project_context

**Files:**
- Modify: `codereview/providers/factory.py`
- Modify: `tests/test_provider_factory.py`

**Step 1: Write the failing test**

Add to `tests/test_provider_factory.py`:

```python
def test_passes_project_context_to_provider(mock_bedrock_client: Mock) -> None:
    """Should pass project_context to provider."""
    factory = ProviderFactory()
    readme_content = "# Test Project"

    provider = factory.create_provider("opus", project_context=readme_content)

    assert provider.project_context == readme_content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provider_factory.py::test_passes_project_context_to_provider -v`
Expected: FAIL with `TypeError: create_provider() got an unexpected keyword argument 'project_context'`

**Step 3: Modify factory.py**

Update `create_provider` method signature and pass through:

```python
    def create_provider(
        self,
        model_name: str,
        temperature: float | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        project_context: str | None = None,
    ) -> ModelProvider:
```

And pass to each provider constructor:

```python
            return BedrockProvider(
                model_config=model_config,
                provider_config=bedrock_config,
                temperature=temperature,
                callbacks=callbacks,
                project_context=project_context,
            )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provider_factory.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add codereview/providers/factory.py tests/test_provider_factory.py
git commit -m "feat(factory): pass project_context through to providers"
```

---

### Task 7: Update CodeAnalyzer to accept project_context

**Files:**
- Modify: `codereview/analyzer.py`
- Modify: `tests/test_analyzer.py`

**Step 1: Write the failing test**

Add to `tests/test_analyzer.py`:

```python
def test_passes_project_context_to_provider(mock_provider_factory: Mock) -> None:
    """Should pass project_context to provider factory."""
    readme_content = "# Test Project"

    analyzer = CodeAnalyzer(model_name="opus", project_context=readme_content)

    mock_provider_factory.return_value.create_provider.assert_called_with(
        "opus", None, callbacks=None, project_context=readme_content
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analyzer.py::test_passes_project_context_to_provider -v`
Expected: FAIL with `TypeError: CodeAnalyzer.__init__() got an unexpected keyword argument 'project_context'`

**Step 3: Modify analyzer.py**

Add `project_context` parameter to `__init__`:

```python
    def __init__(
        self,
        model_name: str = "opus",
        temperature: float | None = None,
        provider_factory: ProviderFactory | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        project_context: str | None = None,
        # Legacy parameters
        region: str | None = None,
        model_id: str | None = None,
    ):
```

And pass to factory:

```python
        self.provider = self.factory.create_provider(
            model_name, temperature, callbacks=callbacks, project_context=project_context
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add codereview/analyzer.py tests/test_analyzer.py
git commit -m "feat(analyzer): add project_context parameter"
```

---

### Task 8: Add CLI options --readme and --no-readme

**Files:**
- Modify: `codereview/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Add CLI options**

Add to `cli.py` after the `--list-models` option:

```python
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
```

Update the `main` function signature to include these parameters:

```python
def main(
    ctx: click.Context,
    directory: Path | None,
    # ... existing params ...
    list_models: bool,
    readme: Path | None,
    no_readme: bool,
) -> None:
```

**Step 2: Run existing CLI tests to verify no regressions**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add codereview/cli.py
git commit -m "feat(cli): add --readme and --no-readme options"
```

---

### Task 9: Integrate readme finder into CLI main flow

**Files:**
- Modify: `codereview/cli.py`

**Step 1: Add imports and integration logic**

Add import at top of `cli.py`:

```python
from codereview.readme_finder import (
    find_readme,
    prompt_readme_confirmation,
    read_readme_content,
)
```

Add README handling after the model display (around line 266), before file scanning:

```python
        # Handle README context
        readme_content: str | None = None
        if not no_readme:
            if readme:
                # User specified a README file
                result = read_readme_content(readme)
                if result:
                    content, size = result
                    readme_content = content
                    console.print(f"📄 Using README: [cyan]{readme}[/cyan] ({size/1024:.1f} KB)\n")
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
```

Update CodeAnalyzer instantiation to pass project_context:

```python
        analyzer = CodeAnalyzer(
            model_name=model_name,
            temperature=temperature,
            callbacks=callbacks,
            project_context=readme_content,
        )
```

**Step 2: Test manually**

Run: `uv run codereview ./codereview --dry-run`
Expected: Should prompt about README.md, then show dry-run output

**Step 3: Commit**

```bash
git add codereview/cli.py
git commit -m "feat(cli): integrate README discovery and confirmation"
```

---

### Task 10: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add documentation for new feature**

Add to the CLI Options table:

```markdown
| `--readme <path>` | Specify README file for project context | None |
| `--no-readme` | Skip README context entirely | False |
```

Add new section under "Running the Tool":

```markdown
### README Context

The tool automatically discovers your project's README.md to provide context for code reviews:

```bash
# Auto-discover README (prompts for confirmation)
uv run codereview ./src

# Specify README explicitly
uv run codereview ./src --readme ./docs/PROJECT.md

# Skip README context
uv run codereview ./src --no-readme
```

The README content is included in each batch sent to the LLM, helping it understand project conventions and requirements.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add README context feature documentation"
```

---

### Task 11: Run full test suite and verify

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run static analysis**

Run: `uv run ruff check codereview/ tests/ && uv run mypy codereview/ --ignore-missing-imports`
Expected: No errors

**Step 3: Manual integration test**

Run: `uv run codereview ./codereview -m haiku --dry-run`
Expected: Prompts for README, shows dry-run with README info

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | find_readme function | readme_finder.py, test_readme_finder.py |
| 2 | read_readme_content function | readme_finder.py, test_readme_finder.py |
| 3 | prompt_readme_confirmation | readme_finder.py, test_readme_finder.py |
| 4 | Base provider project_context | providers/base.py, test_provider_base.py |
| 5 | Update all providers | bedrock.py, azure_openai.py, nvidia.py |
| 6 | ProviderFactory project_context | providers/factory.py, test_provider_factory.py |
| 7 | CodeAnalyzer project_context | analyzer.py, test_analyzer.py |
| 8 | CLI options | cli.py |
| 9 | CLI integration | cli.py |
| 10 | Documentation | CLAUDE.md |
| 11 | Final verification | - |
