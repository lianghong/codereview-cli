from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

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
        mock_factory.get_model_display_name.return_value = "Claude Opus 4.6"
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
        mock_factory.get_model_display_name.return_value = "Claude Opus 4.6"
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
        mock_factory.get_model_display_name.return_value = "Claude Opus 4.6"
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

        # Verify default model is "opus4.8"
        mock_analyzer_cls.assert_called_once()
        call_kwargs = mock_analyzer_cls.call_args[1]
        assert call_kwargs["model_name"] == "opus4.8"


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
        mock_provider.get_model_display_name.return_value = "Claude Opus 4.6"

        # Mock validation result
        mock_result = ValidationResult(valid=True, provider="AWS Bedrock")
        mock_result.add_check("API Key", True, "Configured")
        mock_provider.validate_credentials.return_value = mock_result

        mock_factory.create_provider.return_value = mock_provider
        mock_factory_cls.return_value = mock_factory

        result = cli_runner.invoke(main, ["--validate", "-m", "opus"])

        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        assert "Validating credentials" in result.output
        assert "Claude Opus 4.6" in result.output
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
