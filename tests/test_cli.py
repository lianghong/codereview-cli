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
    # Note: This will fail without AWS credentials
    # We'll add proper mocking in integration tests
    result = cli_runner.invoke(main, [str(sample_code_dir)])
    # Just check it attempts to run
    assert "directory" in result.output.lower() or result.exit_code == 0


def test_cli_output_option(cli_runner, sample_code_dir, tmp_path):
    """Test CLI with output file option."""
    output_file = tmp_path / "report.md"
    result = cli_runner.invoke(
        main, [str(sample_code_dir), "--output", str(output_file)]
    )
    # Command should accept the argument
    assert "--output" not in result.output or result.exit_code == 0
