import re
from contextlib import ExitStack
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from codereview.cli import main


@pytest.fixture
def cli_runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_code_dir(tmp_path):
    """Create sample code directory."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    return tmp_path


def test_cli_no_args(cli_runner):
    """Test CLI with no arguments shows help."""
    result = cli_runner.invoke(main, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "DIRECTORY" in result.output


def test_cli_help(cli_runner):
    """Test CLI help command."""
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_with_directory(cli_runner, sample_code_dir):
    """Test CLI with directory argument."""
    with (
        patch("codereview.cli.CodeAnalyzer") as mock_analyzer_cls,
        patch("codereview.cli.FileScanner") as mock_scanner_cls,
        patch("codereview.cli.ProviderFactory") as mock_factory_cls,
    ):
        # Setup factory mock
        mock_factory = Mock()
        mock_factory.get_model_display_name.return_value = "Claude Opus 5"
        mock_factory_cls.return_value = mock_factory

        # Setup analyzer mock
        mock_analyzer = Mock()
        mock_provider = Mock()
        mock_provider.total_input_tokens = 100
        mock_provider.total_output_tokens = 50
        mock_provider.get_pricing.return_value = {
            "input_price_per_million": 5.0,
            "output_price_per_million": 25.0,
        }
        mock_analyzer.provider = mock_provider
        mock_analyzer.analyze_batch.return_value = Mock(
            summary="Test",
            files_analyzed=1,
            issues_found=0,
            critical_issues=0,
            issues=[],
            improvement_suggestions=[],
            recommendations=[],
            system_design_insights="No issues",
        )
        mock_analyzer.skipped_files = []
        mock_analyzer_cls.return_value = mock_analyzer

        # Setup scanner mock
        mock_scanner = Mock()
        mock_scanner.scan.return_value = [sample_code_dir / "test.py"]
        mock_scanner.skipped_files = []
        mock_scanner_cls.return_value = mock_scanner

        result = cli_runner.invoke(main, [str(sample_code_dir), "--no-readme"])

        # Should succeed
        assert result.exit_code == 0, f"CLI failed with: {result.output}"


def test_cli_output_option(cli_runner, sample_code_dir, tmp_path):
    """Test CLI with output file option."""
    with (
        patch("codereview.cli.CodeAnalyzer") as mock_analyzer_cls,
        patch("codereview.cli.FileScanner") as mock_scanner_cls,
        patch("codereview.cli.ProviderFactory") as mock_factory_cls,
        patch("codereview.cli.MarkdownExporter") as mock_exporter_cls,
    ):
        # Setup factory mock
        mock_factory = Mock()
        mock_factory.get_model_display_name.return_value = "Claude Opus 5"
        mock_factory_cls.return_value = mock_factory

        # Setup analyzer mock
        mock_analyzer = Mock()
        mock_provider = Mock()
        mock_provider.total_input_tokens = 100
        mock_provider.total_output_tokens = 50
        mock_provider.get_pricing.return_value = {
            "input_price_per_million": 5.0,
            "output_price_per_million": 25.0,
        }
        mock_analyzer.provider = mock_provider
        mock_analyzer.analyze_batch.return_value = Mock(
            summary="Test",
            files_analyzed=1,
            issues_found=0,
            critical_issues=0,
            issues=[],
            improvement_suggestions=[],
            recommendations=[],
            system_design_insights="No issues",
        )
        mock_analyzer.skipped_files = []
        mock_analyzer_cls.return_value = mock_analyzer

        # Setup scanner mock
        mock_scanner = Mock()
        mock_scanner.scan.return_value = [sample_code_dir / "test.py"]
        mock_scanner.skipped_files = []
        mock_scanner_cls.return_value = mock_scanner

        # Setup exporter mock
        mock_exporter = Mock()
        mock_exporter_cls.return_value = mock_exporter

        output_file = tmp_path / "report.md"
        result = cli_runner.invoke(
            main, [str(sample_code_dir), "--output", str(output_file), "--no-readme"]
        )

        # Command should succeed
        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        # Verify exporter was called
        mock_exporter.export.assert_called_once()


def test_list_models_flag(cli_runner, monkeypatch):
    """Test --list-models displays available models."""
    from unittest.mock import Mock, patch

    # Set up environment variables for Azure OpenAI
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    with patch("codereview.cli.ProviderFactory") as mock_factory_cls:
        # Mock factory to return test models
        mock_factory = Mock()
        mock_factory.list_available_models.return_value = {
            "bedrock": [
                {"id": "test-opus", "name": "Test Opus", "aliases": "opus-test"},
                {"id": "test-sonnet", "name": "Test Sonnet", "aliases": "sonnet-test"},
            ],
            "azure_openai": [
                {"id": "test-gpt", "name": "Test GPT", "aliases": "gpt-test"},
            ],
        }
        mock_factory_cls.return_value = mock_factory

        result = cli_runner.invoke(main, ["--list-models"])

        assert result.exit_code == 0
        assert "Available Models" in result.output
        assert "test-opus" in result.output
        assert "Test Opus" in result.output
        assert "bedrock" in result.output
        assert "azure_openai" in result.output
        assert "Usage:" in result.output


def _list_models_output(cli_runner, models, extra_args=()):
    """Invoke --list-models against a mocked registry and return its output.

    Renders at a wide terminal so alias assertions aren't defeated by Rich
    wrapping a name across lines.
    """
    from unittest.mock import Mock, patch

    with patch("codereview.cli.ProviderFactory") as mock_factory_cls:
        mock_factory = Mock()
        mock_factory.list_available_models.return_value = {"bedrock": models}
        mock_factory_cls.return_value = mock_factory
        result = cli_runner.invoke(
            main,
            ["--list-models", "--no-color", *extra_args],
            terminal_width=200,
        )
    assert result.exit_code == 0, result.output
    return result.output


_DEPRECATED_MODEL = [
    {
        "id": "opus5",
        "name": "Claude Opus 5",
        "aliases": "opus, claude-opus",
        "deprecated_aliases": "legacy-name-a, legacy-name-b",
    }
]


def test_list_models_hides_deprecated_aliases_by_default(cli_runner):
    """Deprecated aliases are counted, not spelled out.

    Advertising them is actively misleading: they resolve, but to a *successor*
    model. A reader scanning the Aliases column can't tell which spelling is
    current, so the default view shows only current names plus a count.
    """
    output = _list_models_output(cli_runner, _DEPRECATED_MODEL)

    assert "opus" in output
    assert "legacy-name-a" not in output
    assert "+2 deprecated" in output
    assert "--list-models --verbose" in output


def test_list_models_verbose_shows_deprecated_aliases(cli_runner):
    """--verbose discloses them in full, so nothing becomes undiscoverable."""
    output = _list_models_output(cli_runner, _DEPRECATED_MODEL, ["--verbose"])

    assert "legacy-name-a" in output
    assert "legacy-name-b" in output
    # The summary line is for the hidden case only.
    assert "+2 deprecated" not in output


def test_list_models_never_truncates_an_alias(cli_runner):
    """Every alias must print verbatim — a Rich ellipsis prints an invalid name.

    The Aliases column used to truncate ("claude-opus-4.…"), which is worse than
    useless in a table whose whole purpose is telling users what they can type.
    Regression guard for the ``overflow="fold"`` + ``min_width`` pairing: fold
    alone still splits a long name across lines at narrow widths.
    """
    long_alias = "claude-opus-4.6-extra-long-name"
    output = _list_models_output(
        cli_runner,
        [
            {
                "id": "opus5",
                "name": "Claude Opus 5",
                "aliases": long_alias,
                "deprecated_aliases": "",
            }
        ],
    )

    # Scope to the models table — the Provider Setup table below it legitimately
    # elides long setup URLs, which is not what this guards.
    models_table = output.split("Provider Setup")[0]
    assert "…" not in models_table
    assert long_alias in models_table


def test_list_models_tolerates_missing_deprecated_key(cli_runner):
    """A model dict without the key must render, not KeyError.

    ``list_available_models`` always supplies it, but this display path is
    reached with hand-built dicts in tests and by any external caller.
    """
    output = _list_models_output(cli_runner, [{"id": "x", "name": "X", "aliases": "y"}])

    assert "Available Models" in output
    assert "deprecated" not in output


def test_list_models_exits_without_directory(cli_runner, monkeypatch):
    """Test --list-models doesn't require directory argument."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

    with patch("codereview.cli.ProviderFactory") as mock_factory_cls:
        mock_factory = Mock()
        mock_factory.list_available_models.return_value = {"bedrock": []}
        mock_factory_cls.return_value = mock_factory

        result = cli_runner.invoke(main, ["--list-models"])

        assert result.exit_code == 0
        # Should not attempt directory validation
        assert "Scanning" not in result.output


def test_cli_with_model_option(cli_runner, sample_code_dir):
    """Test CLI with --model option uses model_name parameter."""
    with (
        patch("codereview.cli.CodeAnalyzer") as mock_analyzer_cls,
        patch("codereview.cli.FileScanner") as mock_scanner_cls,
        patch("codereview.cli.ProviderFactory") as mock_factory_cls,
    ):
        # Setup factory mock
        mock_factory = Mock()
        mock_factory.get_model_display_name.return_value = "Claude Sonnet 4.6"
        mock_factory_cls.return_value = mock_factory

        # Setup analyzer mock
        mock_analyzer = Mock()
        mock_provider = Mock()
        mock_provider.total_input_tokens = 100
        mock_provider.total_output_tokens = 50
        mock_provider.get_pricing.return_value = {
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
        }
        mock_analyzer.provider = mock_provider
        mock_analyzer.analyze_batch.return_value = Mock(
            summary="Test",
            files_analyzed=1,
            issues_found=0,
            critical_issues=0,
            issues=[],
            improvement_suggestions=[],
            recommendations=[],
            system_design_insights="No issues",
        )
        mock_analyzer.skipped_files = []
        mock_analyzer_cls.return_value = mock_analyzer

        # Setup scanner mock
        mock_scanner = Mock()
        mock_scanner.scan.return_value = [sample_code_dir / "test.py"]
        mock_scanner.skipped_files = []
        mock_scanner_cls.return_value = mock_scanner

        result = cli_runner.invoke(
            main, [str(sample_code_dir), "--model", "sonnet", "--no-readme"]
        )

        # Should succeed
        assert result.exit_code == 0, f"CLI failed with: {result.output}"

        # Verify CodeAnalyzer was called with model_name
        mock_analyzer_cls.assert_called_once()
        call_kwargs = mock_analyzer_cls.call_args[1]
        assert "model_name" in call_kwargs
        assert call_kwargs["model_name"] == "sonnet"
        # Should not have old parameters
        assert "model_id" not in call_kwargs
        assert "region" not in call_kwargs


def test_cli_default_model(cli_runner, sample_code_dir):
    """Test CLI uses default model (opus)."""
    with (
        patch("codereview.cli.CodeAnalyzer") as mock_analyzer_cls,
        patch("codereview.cli.FileScanner") as mock_scanner_cls,
        patch("codereview.cli.ProviderFactory") as mock_factory_cls,
    ):
        # Setup factory mock
        mock_factory = Mock()
        mock_factory.get_model_display_name.return_value = "Claude Opus 5"
        mock_factory_cls.return_value = mock_factory

        # Setup analyzer mock
        mock_analyzer = Mock()
        mock_provider = Mock()
        mock_provider.total_input_tokens = 100
        mock_provider.total_output_tokens = 50
        mock_provider.get_pricing.return_value = {
            "input_price_per_million": 5.0,
            "output_price_per_million": 25.0,
        }
        mock_analyzer.provider = mock_provider
        mock_analyzer.analyze_batch.return_value = Mock(
            summary="Test",
            files_analyzed=1,
            issues_found=0,
            critical_issues=0,
            issues=[],
            improvement_suggestions=[],
            recommendations=[],
            system_design_insights="No issues",
        )
        mock_analyzer.skipped_files = []
        mock_analyzer_cls.return_value = mock_analyzer

        # Setup scanner mock
        mock_scanner = Mock()
        mock_scanner.scan.return_value = [sample_code_dir / "test.py"]
        mock_scanner.skipped_files = []
        mock_scanner_cls.return_value = mock_scanner

        cli_runner.invoke(main, [str(sample_code_dir), "--no-readme"])

        # Verify default model is "opus5"
        mock_analyzer_cls.assert_called_once()
        call_kwargs = mock_analyzer_cls.call_args[1]
        assert call_kwargs["model_name"] == "opus5"


def test_cli_model_short_name(cli_runner, sample_code_dir):
    """Test CLI accepts short model names like 'haiku'."""
    with (
        patch("codereview.cli.CodeAnalyzer") as mock_analyzer_cls,
        patch("codereview.cli.FileScanner") as mock_scanner_cls,
        patch("codereview.cli.ProviderFactory") as mock_factory_cls,
    ):
        # Setup factory mock
        mock_factory = Mock()
        mock_factory.get_model_display_name.return_value = "Claude Haiku 4.5"
        mock_factory_cls.return_value = mock_factory

        # Setup analyzer mock
        mock_analyzer = Mock()
        mock_provider = Mock()
        mock_provider.total_input_tokens = 100
        mock_provider.total_output_tokens = 50
        mock_provider.get_pricing.return_value = {
            "input_price_per_million": 1.0,
            "output_price_per_million": 5.0,
        }
        mock_analyzer.provider = mock_provider
        mock_analyzer.analyze_batch.return_value = Mock(
            summary="Test",
            files_analyzed=1,
            issues_found=0,
            critical_issues=0,
            issues=[],
            improvement_suggestions=[],
            recommendations=[],
            system_design_insights="No issues",
        )
        mock_analyzer.skipped_files = []
        mock_analyzer_cls.return_value = mock_analyzer

        # Setup scanner mock
        mock_scanner = Mock()
        mock_scanner.scan.return_value = [sample_code_dir / "test.py"]
        mock_scanner.skipped_files = []
        mock_scanner_cls.return_value = mock_scanner

        result = cli_runner.invoke(
            main, [str(sample_code_dir), "-m", "haiku", "--no-readme"]
        )

        # Should succeed with short name
        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        mock_analyzer_cls.assert_called_once()
        call_kwargs = mock_analyzer_cls.call_args[1]
        assert call_kwargs["model_name"] == "haiku"


def test_validate_flag(cli_runner):
    """Test --validate runs credential validation without directory."""
    from codereview.providers.base import ValidationResult

    with patch("codereview.cli.ProviderFactory") as mock_factory_cls:
        # Setup factory mock
        mock_factory = Mock()
        mock_provider = Mock()
        mock_provider.get_model_display_name.return_value = "Claude Opus 5"

        # Mock validation result
        mock_result = ValidationResult(valid=True, provider="AWS Bedrock")
        mock_result.add_check("API Key", True, "Configured")
        mock_provider.validate_credentials.return_value = mock_result

        mock_factory.create_provider.return_value = mock_provider
        mock_factory_cls.return_value = mock_factory

        result = cli_runner.invoke(main, ["--validate", "-m", "opus"])

        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        assert "Validating credentials" in result.output
        assert "Claude Opus 5" in result.output
        mock_provider.validate_credentials.assert_called_once()


def test_validate_flag_failure(cli_runner):
    """Test --validate exits with code 1 on validation failure."""
    from codereview.providers.base import ValidationResult

    with patch("codereview.cli.ProviderFactory") as mock_factory_cls:
        # Setup factory mock
        mock_factory = Mock()
        mock_provider = Mock()
        mock_provider.get_model_display_name.return_value = "Test Model"

        # Mock failed validation result
        mock_result = ValidationResult(valid=False, provider="Test Provider")
        mock_result.add_check("API Key", False, "Not configured")
        mock_provider.validate_credentials.return_value = mock_result

        mock_factory.create_provider.return_value = mock_provider
        mock_factory_cls.return_value = mock_factory

        result = cli_runner.invoke(main, ["--validate", "-m", "opus"])

        assert result.exit_code == 1
        mock_provider.validate_credentials.assert_called_once()


# ---------------------------------------------------------------------------
# Aggregation helpers (dedupe across batches)
# ---------------------------------------------------------------------------


def test_dedupe_design_insights_collapses_paraphrases():
    """Near-identical insights from concurrent batches collapse to one."""
    from codereview.cli import _dedupe_design_insights

    insights = [
        "The providers share a common base class with template hooks.",
        "The providers share a common base class with template hooks!",  # punctuation
        "The Providers share A common base class with template hooks.",  # case
        "Static analysis tools run in parallel via ThreadPoolExecutor.",
    ]
    out = _dedupe_design_insights(insights)
    assert len(out) == 2


def test_dedupe_design_insights_preserves_distinct_observations():
    """Genuinely different observations stay separate."""
    from codereview.cli import _dedupe_design_insights

    insights = [
        "Token tracking is lock-guarded for concurrent batches.",
        "Pricing falls back to TBD for unannounced rates.",
        "README content is treated as untrusted data, not instructions.",
    ]
    out = _dedupe_design_insights(insights)
    assert len(out) == 3


def test_dedupe_design_insights_empty_safe():
    from codereview.cli import _dedupe_design_insights

    assert _dedupe_design_insights([]) == []
    assert _dedupe_design_insights(["", "   "]) == []


def _issue(title, *, file_path="app/x.py", line_start=42, severity="Medium"):
    """Build a minimal valid ReviewIssue for dedup fingerprint tests."""
    from codereview.models import ReviewIssue

    return ReviewIssue(
        file_path=file_path,
        line_start=line_start,
        title=title,
        description="d",
        rationale="r",
        severity=severity,
    )


def test_dedupe_issues_collapses_punctuation_and_casing():
    """#4: fingerprint is lowercased + alphanumeric-only, so titles differing
    only in punctuation/casing/whitespace collapse to one issue."""
    from codereview.cli import _dedupe_issues

    issues = [
        _issue("Bare except clause"),
        _issue("Bare `except:` clause"),  # punctuation
        _issue("BARE EXCEPT CLAUSE"),  # casing
        _issue("bare   except   clause"),  # whitespace
    ]
    out = _dedupe_issues(issues)
    assert len(out) == 1


def test_dedupe_issues_keyed_on_file_and_line():
    """Same title at a different file or line is a distinct finding."""
    from codereview.cli import _dedupe_issues

    issues = [
        _issue("Missing timeout", file_path="a.py", line_start=10),
        _issue("Missing timeout", file_path="b.py", line_start=10),  # diff file
        _issue("Missing timeout", file_path="a.py", line_start=20),  # diff line
    ]
    out = _dedupe_issues(issues)
    assert len(out) == 3


def test_dedupe_issues_highest_severity_wins_on_tie():
    """When fingerprints match, the highest-severity issue is kept."""
    from codereview.cli import _dedupe_issues

    issues = [
        _issue("SQL injection", severity="Medium"),
        _issue("SQL injection", severity="Critical"),
        _issue("SQL injection", severity="High"),
    ]
    out = _dedupe_issues(issues)
    assert len(out) == 1
    assert out[0].severity == "Critical"


def test_dedupe_issues_empty_safe():
    from codereview.cli import _dedupe_issues

    assert _dedupe_issues([]) == []


# ---------------------------------------------------------------------------
# Smoke test: full import graph + model registry resolution
# ---------------------------------------------------------------------------


def test_smoke_list_models_exercises_full_import_graph(cli_runner):
    """`--list-models` must exit 0 with a populated table — no mocks.

    Catches regressions in the CLI / config / factory layer that the
    mocked --list-models tests above cannot, because those mock
    ``ProviderFactory`` itself.

    Note this does NOT catch a SyntaxError in an individual provider
    module — ``factory.list_available_models`` only walks YAML configs;
    individual provider modules are imported lazily inside
    ``factory.create_provider``. The companion test
    ``test_smoke_every_provider_module_imports`` covers that gap.

    Runs in <1s, no network.
    """
    result = cli_runner.invoke(main, ["--list-models"])

    assert result.exit_code == 0, (
        f"--list-models failed (likely import-graph regression): "
        f"{result.output}\n"
        f"Exception: {result.exception!r}"
    )
    # Output must include the table header and at least one provider
    # section heading — empty output would mean the loader silently
    # skipped every provider.
    assert "Available Models" in result.output
    assert "Provider Setup" in result.output


def test_smoke_every_provider_module_imports():
    """Every provider module must be importable on its own.

    Catches the case where, e.g., providers/zai.py has a SyntaxError
    that codereview.providers.__init__ would normally hide via lazy
    __getattr__. If --list-models is broken in CI this test gives a
    much shorter, file-level fingerprint of which provider broke.
    """
    import importlib

    provider_modules = [
        "codereview.providers.bedrock",
        "codereview.providers.azure_openai",
        "codereview.providers.nvidia",
        "codereview.providers.google_genai",
        "codereview.providers.zai",
        "codereview.providers.deepseek",
        "codereview.providers.moonshot",
    ]
    for name in provider_modules:
        importlib.import_module(name)


def _dry_run_input_tokens(console_text: str) -> int:
    """Pull the 'Est. input tokens: ~N,NNN' integer out of dry-run output."""
    import re

    m = re.search(r"input tokens:\s*~([\d,]+)", console_text)
    assert m, f"no input-token line in:\n{console_text}"
    return int(m.group(1).replace(",", ""))


def test_dry_run_estimate_includes_readme_tokens(tmp_path):
    """Dry-run cost must account for README context sent per batch.

    Regression: _render_dry_run previously counted only file + system-prompt
    tokens, understating the estimate whenever --readme supplied a large
    README — exactly the metric --dry-run exists to provide.
    """
    from rich.console import Console

    from codereview.batcher import FileBatch
    from codereview.cli import _render_dry_run

    code_file = tmp_path / "mod.py"
    code_file.write_text("def f():\n    return 1\n")
    batch = FileBatch(files=[code_file], batch_number=1, total_batches=1)

    provider = Mock()
    provider.get_pricing.return_value = {
        "input_price_per_million": 5.0,
        "output_price_per_million": 25.0,
    }
    provider.validate_credentials.return_value = Mock(
        valid=True, provider="Test", checks=[], errors=[], warnings=[], suggestions=[]
    )

    readme = "# Project\n" + ("context line\n" * 500)

    def render(readme_content):
        console = Console(record=True, width=100)
        _render_dry_run(
            [code_file],
            [batch],
            "Test Model",
            provider,
            console,
            readme_content=readme_content,
        )
        return _dry_run_input_tokens(console.export_text())

    with_readme = render(readme)
    without_readme = render(None)
    assert with_readme > without_readme, (
        "README tokens must increase the dry-run estimate"
    )


def test_per_batch_overhead_shared_by_budget_and_dry_run():
    """#1: the budget path and the dry-run estimator use one overhead formula.

    Locks in that both callers count the same three components (system prompt,
    README, linter block) so they cannot drift apart.
    """
    from codereview.cli import SYSTEM_PROMPT, _per_batch_overhead_tokens, count_tokens

    # No README, no linters: only the system prompt.
    base = _per_batch_overhead_tokens(None, has_linters=False)
    assert base.readme == 0
    assert base.linter == 0
    assert base.system_prompt == count_tokens(SYSTEM_PROMPT)
    assert base.total == base.system_prompt

    # README and linters each add a positive, separately-tracked component.
    full = _per_batch_overhead_tokens("# Readme\n" * 200, has_linters=True)
    assert full.readme > 0
    assert full.linter > 0
    assert full.total == full.system_prompt + full.readme + full.linter
    assert full.total > base.total


def test_dry_run_estimate_is_upper_bound_on_actual_input(tmp_path):
    """#2: dry-run input estimate must conservatively bound a real multi-batch run.

    The real run sends, per batch, a language-SLICED system prompt (<= the
    worst-case all-language SYSTEM_PROMPT the dry-run uses) plus the README and
    a condensed linter block (<= the 4000-char cap the dry-run reserves), plus
    the batch's file tokens. So dry-run-estimated input >= the sum of tokens
    actually sent. Asserting >= (not ==) is deliberate: equality would be flaky
    because the dry-run intentionally over-reserves.
    """
    from rich.console import Console

    from codereview.batcher import FileBatch, count_tokens
    from codereview.cli import _render_dry_run
    from codereview.config import build_system_prompt, detect_languages_from_paths

    # Two batches, each one Python file — a representative multi-batch run.
    f1 = tmp_path / "a.py"
    f1.write_text("def a():\n    return 1\n" * 20)
    f2 = tmp_path / "b.py"
    f2.write_text("def b():\n    return 2\n" * 20)
    batches = [
        FileBatch(files=[f1], batch_number=1, total_batches=2),
        FileBatch(files=[f2], batch_number=2, total_batches=2),
    ]

    readme = "# Project\n" + ("context line\n" * 100)

    provider = Mock()
    provider.get_pricing.return_value = {
        "input_price_per_million": 5.0,
        "output_price_per_million": 25.0,
    }
    provider.validate_credentials.return_value = Mock(
        valid=True, provider="Test", checks=[], errors=[], warnings=[], suggestions=[]
    )

    console = Console(record=True, width=100)
    _render_dry_run(
        [f1, f2],
        batches,
        "Test Model",
        provider,
        console,
        readme_content=readme,
        static_results={"ruff": object()},  # truthy → linter block reserved
    )
    estimated = _dry_run_input_tokens(console.export_text())

    # Approximate the actual per-batch payload: sliced system prompt + README +
    # file content. (Omit the linter block on the actual side — the dry-run
    # reserves it, so including it here only widens the headroom.)
    actual = 0
    for f in (f1, f2):
        langs = detect_languages_from_paths([str(f)])
        sliced_prompt = build_system_prompt(langs)
        actual += count_tokens(sliced_prompt)
        actual += count_tokens(readme)
        actual += count_tokens(f.read_text())

    assert estimated >= actual, (
        f"dry-run estimate {estimated} must be an upper bound on actual {actual}"
    )


def test_all_batches_failed_exits_nonzero(cli_runner, sample_code_dir):
    """A run where every batch fails must exit non-zero.

    Regression guard: the all-batches-failed path printed an error then
    returned bare, which Click converts to exit code 0 — CI pipelines saw
    success on runs that produced no review at all.
    """
    with (
        patch("codereview.cli.CodeAnalyzer") as mock_analyzer_cls,
        patch("codereview.cli.FileScanner") as mock_scanner_cls,
        patch("codereview.cli.ProviderFactory") as mock_factory_cls,
    ):
        mock_factory = Mock()
        mock_factory.get_model_display_name.return_value = "Claude Opus 5"
        mock_factory_cls.return_value = mock_factory

        mock_analyzer = Mock()
        mock_analyzer.provider = Mock()
        mock_analyzer.analyze_batch.side_effect = RuntimeError("rate limited")
        mock_analyzer.skipped_files = []
        mock_analyzer_cls.return_value = mock_analyzer

        mock_scanner = Mock()
        mock_scanner.scan.return_value = [sample_code_dir / "test.py"]
        mock_scanner.skipped_files = []
        mock_scanner_cls.return_value = mock_scanner

        result = cli_runner.invoke(main, [str(sample_code_dir), "--no-readme"])

        assert "failed" in result.output.lower()
        assert result.exit_code != 0, (
            "all-batches-failed run must not exit 0 — CI would treat a run "
            "with zero review results as success"
        )


# ---------------------------------------------------------------------------
# --fail-on quality gate
# ---------------------------------------------------------------------------


def _gate_issue(severity, **overrides):
    """Build a ReviewIssue at a given severity for gate tests."""
    from codereview.models import ReviewIssue

    return ReviewIssue(
        **{
            "severity": severity,
            "category": "Correctness",
            "file_path": "src/app.py",
            "line_start": 12,
            "title": f"{severity} finding for gate test",
            "description": "Concrete description for the gate test.",
            "rationale": "Concrete rationale for the gate test.",
            **overrides,
        }
    )


def test_evaluate_fail_on_threshold_is_inclusive_and_upward():
    """--fail-on high must trip on High AND Critical, not High alone."""
    from codereview.cli import _evaluate_fail_on

    issues = [_gate_issue("Critical"), _gate_issue("High"), _gate_issue("Low")]

    assert _evaluate_fail_on(issues, "high") == (2, "High")
    assert _evaluate_fail_on(issues, "critical") == (1, "Critical")
    assert _evaluate_fail_on(issues, "info") == (3, "Info")


def test_evaluate_fail_on_passes_when_nothing_meets_threshold():
    """Below-threshold findings must not trip the gate."""
    from codereview.cli import _evaluate_fail_on

    assert _evaluate_fail_on([_gate_issue("Low"), _gate_issue("Info")], "high") is None
    assert _evaluate_fail_on([], "info") is None


def test_evaluate_fail_on_accepts_any_case():
    """Click passes the raw user string; casing must not matter."""
    from codereview.cli import _evaluate_fail_on

    for spelling in ("HIGH", "High", "high"):
        assert _evaluate_fail_on([_gate_issue("Critical")], spelling) == (1, "High")


def _run_with_issues(cli_runner, sample_code_dir, issues, extra_args=()):
    """Invoke the CLI with a mocked analyzer returning `issues`."""
    from codereview.models import CodeReviewReport, ReviewMetrics

    with (
        patch("codereview.cli.CodeAnalyzer") as mock_analyzer_cls,
        patch("codereview.cli.FileScanner") as mock_scanner_cls,
        patch("codereview.cli.ProviderFactory") as mock_factory_cls,
    ):
        mock_factory = Mock()
        mock_factory.get_model_display_name.return_value = "Claude Opus 5"
        mock_factory_cls.return_value = mock_factory

        mock_provider = Mock()
        mock_provider.total_input_tokens = 100
        mock_provider.total_output_tokens = 50
        mock_provider.get_pricing.return_value = {
            "input_price_per_million": 5.0,
            "output_price_per_million": 25.0,
        }

        mock_analyzer = Mock()
        mock_analyzer.provider = mock_provider
        mock_analyzer.analyze_batch.return_value = CodeReviewReport(
            summary="Test",
            metrics=ReviewMetrics(files_analyzed=1),
            issues=issues,
            system_design_insights="No issues",
            recommendations=[],
            improvement_suggestions=[],
        )
        mock_analyzer.skipped_files = []
        mock_analyzer_cls.return_value = mock_analyzer

        mock_scanner = Mock()
        mock_scanner.scan.return_value = [sample_code_dir / "test.py"]
        mock_scanner.skipped_files = []
        mock_scanner_cls.return_value = mock_scanner

        return cli_runner.invoke(
            main, [str(sample_code_dir), "--no-readme", *extra_args]
        )


def test_fail_on_trips_with_exit_code_2(cli_runner, sample_code_dir):
    """A Critical finding with --fail-on high must exit 2.

    Exit 2 is distinct from 1 (the run itself failed) so CI can tell "the tool
    broke" from "the tool worked and your code has problems".
    """
    result = _run_with_issues(
        cli_runner,
        sample_code_dir,
        [_gate_issue("Critical")],
        extra_args=["--fail-on", "high"],
    )

    assert result.exit_code == 2, f"expected gate failure, got: {result.output}"
    assert "Quality gate failed" in result.output


def test_fail_on_passes_when_below_threshold(cli_runner, sample_code_dir):
    """Findings below the threshold must still exit 0."""
    result = _run_with_issues(
        cli_runner,
        sample_code_dir,
        [_gate_issue("Low"), _gate_issue("Info")],
        extra_args=["--fail-on", "high"],
    )

    assert result.exit_code == 0, f"gate should have passed: {result.output}"
    assert "Quality gate failed" not in result.output


def test_no_fail_on_never_gates(cli_runner, sample_code_dir):
    """Without --fail-on, findings must not change the exit code.

    Backwards compatibility: the gate is strictly opt-in.
    """
    result = _run_with_issues(cli_runner, sample_code_dir, [_gate_issue("Critical")])

    assert result.exit_code == 0, f"expected no gating: {result.output}"
    assert "Quality gate failed" not in result.output


def test_severity_filter_does_not_affect_fail_on(cli_runner, sample_code_dir):
    """--severity filters the DISPLAY only; it must not weaken the gate.

    Regression guard for the most dangerous way to wire these two flags
    together: gating on the rendered subset would let `--severity critical`
    silently suppress a High finding from the exit code, turning a display
    preference into a security hole in the CI gate.
    """
    result = _run_with_issues(
        cli_runner,
        sample_code_dir,
        [_gate_issue("High")],
        extra_args=["--severity", "critical", "--fail-on", "high"],
    )

    assert result.exit_code == 2, (
        "a High finding hidden by --severity critical must still trip "
        f"--fail-on high; got: {result.output}"
    )


def test_fail_on_still_writes_report(cli_runner, sample_code_dir, tmp_path):
    """The gate must trip AFTER export so CI keeps its artifact."""
    report_path = tmp_path / "report.json"

    result = _run_with_issues(
        cli_runner,
        sample_code_dir,
        [_gate_issue("Critical")],
        extra_args=[
            "--fail-on",
            "critical",
            "--output",
            str(report_path),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 2
    assert report_path.exists(), (
        "gate must run after export — a failing build still needs its report"
    )


# ---------------------------------------------------------------------------
# Token-budget fallback must warn on every run, not only under --verbose
#
# Losing the token budget means batches are packed by file count alone and can
# overflow the context window mid-run — that changes what the review actually
# does, so hiding it behind --verbose lets a degraded run look like a clean one.
# ---------------------------------------------------------------------------


def _run_with_tiny_context_window(cli_runner, code_dir, extra_args=()):
    """Invoke the CLI against a model whose context window can't fit overhead."""
    from codereview.config.models import ModelConfig, PricingConfig

    # context_window is truthy but far smaller than max_output + system prompt +
    # safety margin, so computed_budget goes negative and the fallback triggers.
    tiny_model = ModelConfig(
        id="tiny",
        full_id="vendor/tiny",
        name="Tiny Context Model",
        pricing=PricingConfig(input_per_million=1.0, output_per_million=2.0),
        context_window=1000,
    )

    with (
        patch("codereview.cli.CodeAnalyzer") as mock_analyzer_cls,
        patch("codereview.cli.FileScanner") as mock_scanner_cls,
        patch("codereview.cli.ProviderFactory") as mock_factory_cls,
        patch("codereview.cli.get_config_loader") as mock_loader,
    ):
        mock_loader.return_value.resolve_model.return_value = ("bedrock", tiny_model)

        mock_factory_cls.return_value.get_model_display_name.return_value = "Tiny"

        mock_provider = Mock()
        mock_provider.total_input_tokens = 10
        mock_provider.total_output_tokens = 5
        mock_provider.get_pricing.return_value = {
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
        }
        mock_analyzer = Mock()
        mock_analyzer.provider = mock_provider
        mock_analyzer.analyze_batch.return_value = Mock(
            summary="Test",
            files_analyzed=1,
            issues_found=0,
            critical_issues=0,
            issues=[],
            improvement_suggestions=[],
            recommendations=[],
            system_design_insights="No issues",
        )
        mock_analyzer.skipped_files = []
        mock_analyzer_cls.return_value = mock_analyzer

        mock_scanner = Mock()
        mock_scanner.scan.return_value = [code_dir / "test.py"]
        mock_scanner.skipped_files = []
        mock_scanner_cls.return_value = mock_scanner

        return cli_runner.invoke(
            main, [str(code_dir), "--no-readme", "--no-color", *extra_args]
        )


def test_token_budget_fallback_warns_without_verbose(cli_runner, sample_code_dir):
    """The count-only-batching fallback is visible on a plain (non-verbose) run."""
    result = _run_with_tiny_context_window(cli_runner, sample_code_dir)

    assert result.exit_code == 0, f"CLI failed with: {result.output}"
    assert "count-only batching" in result.output, (
        "degraded batching must be announced without --verbose; otherwise a run "
        "that may overflow the context window looks identical to a clean one"
    )


def test_token_budget_fallback_also_warns_with_verbose(cli_runner, sample_code_dir):
    """--verbose keeps the warning (it is not an either/or with the breakdown)."""
    result = _run_with_tiny_context_window(
        cli_runner, sample_code_dir, extra_args=["--verbose"]
    )

    assert result.exit_code == 0, f"CLI failed with: {result.output}"
    assert "count-only batching" in result.output


# ---------------------------------------------------------------------------
# --list-models' Provider Setup table must match models.yaml's real env vars
#
# The table is hand-maintained prose in cli.py while the authoritative list is
# the set of ${VAR} references in models.yaml. Nothing connected the two, so
# adding a provider (or renaming its env var) could leave the table telling
# users to export a variable that nothing reads, or omit one they need.
# ---------------------------------------------------------------------------


def _yaml_provider_env_vars() -> dict[str, set[str]]:
    """provider name -> {env vars its models.yaml block expands}.

    Read from the RAW yaml, before ConfigLoader's ${VAR} expansion, so the
    result doesn't depend on which credentials happen to be set in the
    environment running the tests.
    """
    import re
    from pathlib import Path

    import yaml

    raw = (
        Path(__file__).resolve().parent.parent / "codereview" / "config" / "models.yaml"
    )
    providers = yaml.safe_load(raw.read_text())["providers"]

    found: dict[str, set[str]] = {}
    for name, block in providers.items():
        scalars = (v for k, v in block.items() if k != "models" and isinstance(v, str))
        found[name] = {
            var for value in scalars for var in re.findall(r"\$\{(\w+)\}", value)
        }
    return found


def _setup_table_rows() -> dict[str, str]:
    """provider name -> the 'Required Environment Variables' cell, from cli.py."""
    from rich.console import Console
    from rich.table import Table

    from codereview.cli import display_available_models

    captured: dict[str, str] = {}
    real_add_row = Table.add_row

    def spy(self, *cells, **kwargs):
        if self.title == "Provider Setup" and len(cells) >= 2:
            captured[str(cells[0])] = str(cells[1])
        return real_add_row(self, *cells, **kwargs)

    with patch.object(Table, "add_row", spy):
        display_available_models(Console(file=StringIO(), width=200))
    return captured


def test_provider_setup_table_covers_every_configured_provider():
    """Every provider in models.yaml gets a Provider Setup row, and vice versa."""
    yaml_providers = set(_yaml_provider_env_vars())
    table_providers = set(_setup_table_rows())

    assert yaml_providers == table_providers, (
        "Provider Setup table (cli.py) and models.yaml disagree on the provider "
        f"list. Only in yaml: {sorted(yaml_providers - table_providers)}; "
        f"only in table: {sorted(table_providers - yaml_providers)}"
    )


def test_provider_setup_table_names_the_env_vars_models_yaml_actually_reads():
    """Each row must name every ${VAR} its provider block expands.

    Catches both directions of drift: a row that omits a required variable
    (user hits a confusing failure later) and a row advertising a variable
    nothing reads (user exports it for nothing).
    """
    yaml_env = _yaml_provider_env_vars()
    rows = _setup_table_rows()

    problems: list[str] = []
    for provider, expected in yaml_env.items():
        cell = rows.get(provider, "")
        missing = sorted(var for var in expected if var not in cell)
        if missing:
            problems.append(f"{provider}: row omits {missing}")

        # Any *_API_KEY / *_BASE_URL-shaped name advertised in the row should be
        # one the YAML really expands. Bedrock's row documents AWS credential-
        # chain variables, which are read by boto3 rather than models.yaml.
        if provider != "bedrock":
            advertised = set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", cell))
            stale = sorted(advertised - expected)
            if stale:
                problems.append(f"{provider}: row advertises unread {stale}")

    assert not problems, "Provider Setup table drifted from models.yaml: " + "; ".join(
        problems
    )


# ---------------------------------------------------------------------------
# run_review(): the pipeline, callable without CliRunner
#
# main() is now Click parsing plus the flags that exit before any review work
# (--list-models, --validate, no-directory help); run_review owns everything
# from scanning to the --fail-on gate. These tests drive run_review directly,
# which lets them assert on *ordering* and on real exception types instead of
# inferring both from an exit code.
# ---------------------------------------------------------------------------


def _pipeline_mocks(code_dir, issues=(), *, analyzer_error=None, recommendations=()):
    """Context manager stack + the mock analyzer, shared by run_review tests."""
    from codereview.models import CodeReviewReport, ReviewMetrics

    stack = ExitStack()
    mock_analyzer_cls = stack.enter_context(patch("codereview.cli.CodeAnalyzer"))
    mock_scanner_cls = stack.enter_context(patch("codereview.cli.FileScanner"))
    stack.enter_context(patch("codereview.cli.ProviderFactory"))

    mock_provider = Mock()
    mock_provider.total_input_tokens = 100
    mock_provider.total_output_tokens = 50
    mock_provider.get_pricing.return_value = {
        "input_price_per_million": 5.0,
        "output_price_per_million": 25.0,
    }

    mock_analyzer = Mock()
    mock_analyzer.provider = mock_provider
    if analyzer_error is not None:
        mock_analyzer.analyze_batch.side_effect = analyzer_error
    else:
        mock_analyzer.analyze_batch.return_value = CodeReviewReport(
            summary="Test",
            metrics=ReviewMetrics(files_analyzed=1),
            issues=list(issues),
            system_design_insights="No issues",
            recommendations=list(recommendations),
            improvement_suggestions=[],
        )
    mock_analyzer.skipped_files = []
    mock_analyzer_cls.return_value = mock_analyzer

    mock_scanner = Mock()
    mock_scanner.scan.return_value = [code_dir / "test.py"]
    mock_scanner.skipped_files = []
    mock_scanner_cls.return_value = mock_scanner

    return stack, mock_analyzer


def test_run_review_is_callable_without_the_click_runner(sample_code_dir):
    """The pipeline runs against a plain Console, no CliRunner.invoke needed."""
    from codereview.cli import run_review

    buffer = StringIO()
    stack, mock_analyzer = _pipeline_mocks(sample_code_dir)
    with stack:
        run_review(sample_code_dir, console=Console(file=buffer), no_readme=True)

    assert mock_analyzer.analyze_batch.called
    assert "Code Review Tool" in buffer.getvalue()


def test_run_review_defaults_match_the_click_option_defaults(
    cli_runner, sample_code_dir
):
    """main() must be a pass-through: no default may drift between the two.

    Compares what a bare ``codereview <dir>`` invocation actually hands
    run_review against run_review's own signature defaults. Behavioural rather
    than structural on purpose — it doesn't care how Click spells "no default"
    internally, only that calling run_review directly reviews under the same
    settings the CLI does.
    """
    import inspect

    from codereview.cli import run_review

    with patch("codereview.cli.run_review") as mock_run:
        result = cli_runner.invoke(main, [str(sample_code_dir), "--no-readme"])
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()

    passed = mock_run.call_args.kwargs
    signature = inspect.signature(run_review)

    mismatches = []
    for name, value in passed.items():
        if name == "console":  # built by main, has no signature default
            continue
        default = signature.parameters[name].default
        # --no-readme is the one flag this invocation overrides.
        if name == "no_readme":
            continue
        if value != default:
            mismatches.append(f"{name}: cli={value!r} run_review default={default!r}")

    assert not mismatches, (
        "run_review's defaults drifted from the Click options it mirrors: "
        + "; ".join(mismatches)
    )
    # Every reviewable option must be forwarded — a new Click option that
    # main() forgets to pass through would otherwise go unnoticed.
    forwarded = set(passed) | {"directory"}
    expected = {
        name for name in signature.parameters if name not in {"console", "directory"}
    }
    missing = sorted(expected - forwarded)
    assert not missing, f"main() does not forward: {missing}"


def test_run_review_applies_the_gate_after_writing_the_report(
    sample_code_dir, tmp_path
):
    """Export happens before the gate — asserted by observed ordering.

    The CliRunner version of this test can only check that the file exists
    afterwards. Calling run_review directly lets us watch the export and the
    SystemExit in sequence, which is what the invariant actually says.
    """
    from codereview.cli import EXIT_QUALITY_GATE_FAILED, run_review

    report_path = tmp_path / "report.json"
    events = []

    real_write_text = Path.write_text

    def recording_write_text(self, *args, **kwargs):
        if self == report_path:
            events.append("export")
        return real_write_text(self, *args, **kwargs)

    stack, _ = _pipeline_mocks(sample_code_dir, [_gate_issue("Critical")])
    with stack, patch.object(Path, "write_text", recording_write_text):
        with pytest.raises(SystemExit) as excinfo:
            run_review(
                sample_code_dir,
                console=Console(file=StringIO()),
                no_readme=True,
                fail_on="critical",
                output=report_path,
                output_format="json",
            )
        events.append("gate")

    assert excinfo.value.code == EXIT_QUALITY_GATE_FAILED
    assert events == ["export", "gate"], (
        "the --fail-on gate must be the last statement, after export, so a "
        f"failing CI build still keeps its artifact; saw {events}"
    )
    assert report_path.exists()


def test_run_review_cleans_up_callbacks_even_when_the_gate_trips(sample_code_dir):
    """SystemExit is a BaseException, so `finally` still runs the cleanup.

    Locks the reason the gate can raise SystemExit from inside the try block
    without leaking a live Rich display.
    """
    from codereview.cli import run_review

    handler = Mock()
    stack, _ = _pipeline_mocks(sample_code_dir, [_gate_issue("Critical")])
    with stack, patch("codereview.cli.ProgressCallbackHandler") as handler_cls:
        handler_cls.return_value = handler
        with pytest.raises(SystemExit):
            run_review(
                sample_code_dir,
                console=Console(file=StringIO()),
                no_readme=True,
                verbose=True,
                fail_on="critical",
            )

    handler.cleanup.assert_called_once()


def test_run_review_raises_abort_when_every_batch_fails(sample_code_dir):
    """All batches failing exits 1, not 2: the run broke, findings didn't."""
    from codereview.cli import run_review

    stack, _ = _pipeline_mocks(
        sample_code_dir, analyzer_error=RuntimeError("provider exploded")
    )
    buffer = StringIO()
    with stack:
        with pytest.raises(SystemExit) as excinfo:
            run_review(sample_code_dir, console=Console(file=buffer), no_readme=True)

    assert excinfo.value.code == 1, (
        "a run with zero review results must not reuse the quality-gate code"
    )
    assert "No code review results" in buffer.getvalue()


def test_main_still_exits_before_the_pipeline_for_metadata_flags(cli_runner):
    """--list-models and --validate must not reach run_review at all."""
    with patch("codereview.cli.run_review") as mock_run:
        assert cli_runner.invoke(main, ["--list-models"]).exit_code == 0
        with patch("codereview.cli.ProviderFactory"):
            cli_runner.invoke(main, ["--validate"])
        assert cli_runner.invoke(main, []).exit_code == 0  # no directory -> help

    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# AWS Error.Message redaction (CWE-209)
#
# AWS puts the denying SCP statement, principal/role ARNs, account ids and
# resource ids in Error.Message. Console output lands in CI logs, which are
# retained and shared far more widely than the IAM config they describe. The
# error *code* is what a user acts on, so the code is always printed and the
# provider's prose only appears under --verbose. providers/bedrock.py's
# validate_credentials already did this; these lock the three paths that render
# a *run's* errors, which were the loud ones.
# ---------------------------------------------------------------------------

# A realistic explicit-deny message: everything in here is what must not leak.
_LEAKY_AWS_MESSAGE = (
    "User: arn:aws:sts::123456789012:assumed-role/ci-deploy/session is not "
    "authorized to perform: bedrock:InvokeModel with an explicit deny in a "
    "service control policy"
)


def _leaky_client_error(code="AccessDeniedException"):
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": code, "Message": _LEAKY_AWS_MESSAGE}}, "Converse"
    )


def _rendered(fn):
    """Run *fn* against a wide Console and return everything it printed."""
    buffer = StringIO()
    fn(Console(file=buffer, width=240, no_color=True))
    return buffer.getvalue()


# (error code, the label the branch prints). AccessDenied has its own
# human-readable branch; everything else falls through to the generic one that
# names the raw code.
_BATCH_ERROR_BRANCHES = [
    ("AccessDeniedException", "AWS Access Denied"),
    ("ValidationException", "ValidationException"),
]


@pytest.mark.parametrize("code,label", _BATCH_ERROR_BRANCHES)
def test_batch_error_withholds_the_aws_message_without_verbose(code, label):
    from codereview.cli import _render_batch_error

    output = _rendered(
        lambda con: _render_batch_error(
            con, 1, _leaky_client_error(code), "Model", False
        )
    )

    assert "explicit deny" not in output
    assert "arn:aws:sts::123456789012" not in output
    # The actionable part still has to be there, or the redaction has cost the
    # user the only thing they can act on.
    assert label in output
    assert "--verbose" in output, "must say how to see the detail"


@pytest.mark.parametrize("code,label", _BATCH_ERROR_BRANCHES)
def test_batch_error_shows_the_aws_message_with_verbose(code, label):
    """--verbose is the opt-in: the prose is withheld, not discarded."""
    from codereview.cli import _render_batch_error

    output = _rendered(
        lambda con: _render_batch_error(
            con, 1, _leaky_client_error(code), "Model", True
        )
    )

    assert "explicit deny" in output
    assert label in output


def test_validate_withholds_the_aws_message_without_verbose():
    from codereview.cli import validate_provider_credentials

    def run(con):
        with patch("codereview.cli.ProviderFactory") as factory_cls:
            factory_cls.return_value.create_provider.side_effect = _leaky_client_error()
            with pytest.raises(SystemExit):
                validate_provider_credentials("opus5", None, con)

    output = _rendered(run)
    assert "explicit deny" not in output
    assert "AccessDeniedException" in output


def test_validate_shows_the_aws_message_with_verbose():
    from codereview.cli import validate_provider_credentials

    def run(con):
        with patch("codereview.cli.ProviderFactory") as factory_cls:
            factory_cls.return_value.create_provider.side_effect = _leaky_client_error()
            with pytest.raises(SystemExit):
                validate_provider_credentials("opus5", None, con, verbose=True)

    assert "explicit deny" in _rendered(run)


@pytest.mark.parametrize("verbose", [False, True])
def test_run_review_aws_handler_gates_the_message_on_verbose(sample_code_dir, verbose):
    """The third render path: run_review's own `except ClientError`.

    Reached when the failure happens outside the batch loop (here: analyzer
    construction, i.e. provider setup), so it needs its own coverage — it was
    the one site still interpolating Error.Message after the other two were
    fixed. A ClientError raised *inside* the loop goes to _render_batch_error
    and ends as "no results", never touching this handler.
    """
    import click

    from codereview.cli import run_review

    def run(con):
        stack, _ = _pipeline_mocks(sample_code_dir)
        with stack, patch("codereview.cli.CodeAnalyzer") as analyzer_cls:
            analyzer_cls.side_effect = _leaky_client_error()
            with pytest.raises(click.Abort):
                run_review(
                    sample_code_dir, console=con, no_readme=True, verbose=verbose
                )

    output = _rendered(run)
    assert "AccessDeniedException" in output
    assert ("explicit deny" in output) is verbose
    # The troubleshooting hints are keyed off the code, so they must survive
    # redaction — they are what replaces the withheld prose.
    assert "bedrock:InvokeModel" in output


# ---------------------------------------------------------------------------
# Reported counts must describe what the run actually did
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, expected, why",
    [
        (b"a\nb\nc\n", 3, "trailing newline: newline count == line count"),
        (b"a\nb\nc", 3, "no trailing newline: the last line still counts"),
        (b"print(1)", 1, "single line, no newline at all — reported 0 before"),
        (b"", 0, "empty file contributes nothing"),
        (b"\n", 1, "a lone newline is one (empty) line"),
    ],
)
def test_total_lines_counts_the_final_unterminated_line(
    tmp_path, content, expected, why
):
    """Counting b"\\n" is `wc -l`, not a line count.

    A final line with no trailing newline has no separator to count, so every
    such file shaved 1 off the "Total lines of code" figure the report presents
    as the size of what was reviewed — and a one-line file without a trailing
    newline reported 0.
    """
    from codereview.cli import run_review

    code_file = tmp_path / "test.py"
    code_file.write_bytes(content)

    buffer = StringIO()
    stack, _ = _pipeline_mocks(tmp_path)
    with stack:
        run_review(tmp_path, console=Console(file=buffer, width=200), no_readme=True)

    assert f"Total lines of code: {expected:,}" in buffer.getvalue(), why


def _oversized_batch_mocks(code_dir, *, scanned, batched):
    """Pipeline mocks where the batcher drops some scanned files."""
    from codereview.batcher import FileBatch

    stack, mock_analyzer = _pipeline_mocks(code_dir)
    scanner_cls = stack.enter_context(patch("codereview.cli.FileScanner"))
    scanner = Mock()
    scanner.scan.return_value = list(scanned)
    scanner.skipped_files = []
    scanner_cls.return_value = scanner

    batcher_cls = stack.enter_context(patch("codereview.cli.FileBatcher"))
    batcher = Mock()
    batcher.create_batches.return_value = [
        FileBatch(files=list(batched), batch_number=1, total_batches=1)
    ]
    batcher.skipped_oversized = [
        (path, 999_999) for path in scanned if path not in batched
    ]
    batcher_cls.return_value = batcher
    # estimate_file_tokens is a staticmethod read off the class in _render_dry_run.
    batcher_cls.estimate_file_tokens = Mock(return_value=1000)
    return stack, mock_analyzer


def test_files_analyzed_excludes_files_the_batcher_dropped(tmp_path):
    """files_analyzed must count what was sent, not what was scanned.

    An oversized file is reported as skipped and then never sent to any
    provider, so counting the scan claimed coverage the review does not have:
    "Analyzed 2 files" for a run that reviewed 1.
    """
    from codereview.cli import run_review

    reviewed = tmp_path / "small.py"
    reviewed.write_text("x = 1\n")
    dropped = tmp_path / "huge.py"
    dropped.write_text("y = 2\n")

    buffer = StringIO()
    stack, _ = _oversized_batch_mocks(
        tmp_path, scanned=[reviewed, dropped], batched=[reviewed]
    )
    report_path = tmp_path / "out.json"
    with stack:
        run_review(
            tmp_path,
            console=Console(file=buffer, width=200),
            no_readme=True,
            output=report_path,
            output_format="json",
        )

    import json

    exported = json.loads(report_path.read_text())
    assert exported["metrics"]["files_analyzed"] == 1, (
        "the oversized file was never sent to the model but was counted as analyzed"
    )


def test_dry_run_does_not_bill_files_the_batcher_will_drop(tmp_path):
    """--dry-run answers "what will this cost?" — it must price the real request.

    An oversized file is excluded from every batch, so its tokens are never
    sent. Estimating from the scan let one skipped multi-MB file dominate the
    quote: a probe of a 50-line file plus one oversized file quoted ~1,000,350
    input tokens for a run that sends 300.
    """
    from codereview.cli import run_review

    reviewed = tmp_path / "small.py"
    reviewed.write_text("x = 1\n")
    dropped = tmp_path / "huge.py"
    dropped.write_text("y = 2\n")

    buffer = StringIO()
    stack, _ = _oversized_batch_mocks(
        tmp_path, scanned=[reviewed, dropped], batched=[reviewed]
    )
    with stack:
        run_review(
            tmp_path,
            console=Console(file=buffer, width=200, no_color=True),
            no_readme=True,
            dry_run=True,
        )

    output = buffer.getvalue()
    assert (
        "huge.py"
        not in output.split("Estimated Cost Summary")[0].split("Files to Analyze")[-1]
    ), "the dropped file is listed in the table of files to analyze"
    assert "Files: 1" in output
    assert "excludes 1 file(s) skipped as too large" in output


@pytest.mark.parametrize("bad", ["-0.5", "2.5", "99"])
def test_temperature_out_of_range_is_a_usage_error(cli_runner, sample_code_dir, bad):
    """An invalid --temperature must fail at parse time, not after the scan.

    _resolve_temperature enforces the same range, but only inside provider
    construction — after scanning, line counting, static analysis and batching.
    `--temperature 99` used to spend all of that (minutes, with
    --static-analysis) before rejecting an argument that was invalid before
    anything started. Click's usage error is also exit 2, the honest code for a
    bad argument.
    """
    result = cli_runner.invoke(
        main, [str(sample_code_dir), "--no-readme", "--temperature", bad]
    )

    assert result.exit_code == 2, result.output
    assert "--temperature" in result.output
    # Nothing from the pipeline may have run.
    assert "Scanning directory" not in result.output


@pytest.mark.parametrize("good", ["0", "0.7", "2.0"])
def test_in_range_temperature_is_still_accepted(cli_runner, sample_code_dir, good):
    """The bound must not reject the documented range's endpoints."""
    with patch("codereview.cli.run_review") as mock_run:
        result = cli_runner.invoke(
            main, [str(sample_code_dir), "--no-readme", "--temperature", good]
        )

    assert result.exit_code == 0, result.output
    assert mock_run.call_args.kwargs["temperature"] == float(good)


@pytest.mark.parametrize(
    "argv,expected",
    [([], False), (["--trust-repo-config"], True)],
)
def test_trust_repo_config_reaches_the_static_analyzer(
    cli_runner, sample_code_dir, argv, expected
):
    """The flag has to arrive at StaticAnalyzer, defaulting to *off*.

    The gate it controls refuses to run mypy/ESLint/Prettier against a
    repo-supplied config that would execute code from the tree. A flag that
    parsed but never reached the constructor would leave the default silently
    inverted — the failure being invisible is the whole problem.
    """
    with patch("codereview.cli.StaticAnalyzer") as analyzer_cls:
        analyzer_cls.return_value.available_tools = []
        analyzer_cls.return_value.run_all.return_value = {}
        cli_runner.invoke(
            main,
            [str(sample_code_dir), "--no-readme", "--static-analysis", "--dry-run"]
            + argv,
        )

    analyzer_cls.assert_called_once()
    assert analyzer_cls.call_args.kwargs["trust_repo_config"] is expected


def _exported_recommendations(tmp_path, *, issues=(), recommendations=()):
    """Run the pipeline and return the Recommendations from the JSON export."""
    import json

    from codereview.cli import run_review

    (tmp_path / "test.py").write_text("x = 1\n")
    report_path = tmp_path / "out.json"
    stack, _ = _pipeline_mocks(tmp_path, issues=issues, recommendations=recommendations)
    with stack:
        run_review(
            tmp_path,
            console=Console(file=StringIO(), width=200),
            no_readme=True,
            output=report_path,
            output_format="json",
        )
    return json.loads(report_path.read_text())["recommendations"]


def test_model_recommendations_survive_aggregation(tmp_path):
    """The model's traceable recommendations must reach the report.

    SYSTEM_PROMPT asks for recommendations "DERIVED FROM the issues you
    reported. Reference issue titles, not new ideas" — the point is that a
    reader can act on them. Aggregation collected every other list field and
    dropped this one, replacing "Fix the SQL injection in views.py:42" with
    "🔒 Resolve 1 security issue(s)": the counts the Metrics section already
    prints, with the file, line and title thrown away.
    """
    exported = _exported_recommendations(
        tmp_path,
        issues=[_gate_issue("Critical", category="Security")],
        recommendations=[
            "Fix the SQL injection in views.py:42",
            "Add a bounds check to parse_header (utils/http.py:88)",
        ],
    )

    assert exported[:2] == [
        "Fix the SQL injection in views.py:42",
        "Add a bounds check to parse_header (utils/http.py:88)",
    ], f"model recommendations were dropped: {exported}"


def test_recommendations_fall_back_to_counts_when_the_model_returns_none(tmp_path):
    """No model recommendations → the count summary, not an empty section.

    Happens for real: a partial-failure run where the surviving batches emitted
    none. A generic pointer beats a blank "what to do next".
    """
    exported = _exported_recommendations(
        tmp_path,
        issues=[_gate_issue("Critical", category="Security")],
        recommendations=[],
    )

    assert exported, "an issue list with no model recommendations rendered nothing"
    assert any("critical issue(s) immediately" in text for text in exported), exported


def test_duplicate_recommendations_from_concurrent_batches_collapse(tmp_path):
    """Batches reviewing the same shared helper recommend the same fix.

    Each batch sees the helper independently, so the identical recommendation
    arrives N times. Dedup is on the same coarse normalization
    _dedupe_design_insights uses, so punctuation and casing differences collapse
    too.
    """
    exported = _exported_recommendations(
        tmp_path,
        issues=[_gate_issue("High")],
        recommendations=[
            "Fix the SQL injection in views.py:42",
            "fix the SQL injection in views.py:42.",
            "Rotate the leaked credential in settings.py:7",
        ],
    )

    assert exported == [
        "Fix the SQL injection in views.py:42",
        "Rotate the leaked credential in settings.py:7",
    ], exported


def test_recommendations_are_capped_at_five(tmp_path):
    """The section is a shortlist; a 30-batch run must not dump 30 lines."""
    exported = _exported_recommendations(
        tmp_path,
        issues=[_gate_issue("Low")],
        recommendations=[f"Fix finding number {n} in mod{n}.py" for n in range(12)],
    )

    assert len(exported) == 5, exported
