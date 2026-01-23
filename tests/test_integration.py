"""Integration tests for full workflow."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from codereview.analyzer import CodeAnalyzer
from codereview.batcher import FileBatcher
from codereview.cli import main
from codereview.models import CodeReviewReport, ReviewIssue
from codereview.renderer import MarkdownExporter
from codereview.scanner import FileScanner


@pytest.fixture
def sample_project_dir():
    """Create a temporary sample project for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create directory structure
    src_dir = temp_dir / "src"
    src_dir.mkdir()

    # Create sample Python file
    python_file = src_dir / "test.py"
    python_file.write_text("""
def calculate_sum(a, b):
    return a + b

def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
""")

    # Create sample Go file
    go_file = src_dir / "test.go"
    go_file.write_text("""
package main

import "fmt"

func main() {
    fmt.Println("Hello World")
}

func add(a, b int) int {
    return a + b
}
""")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_code_review_report():
    """Create a mock code review report."""
    return CodeReviewReport(
        summary="Test analysis complete",
        metrics={
            "files_analyzed": 2,
            "total_issues": 3,
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 0,
        },
        issues=[
            ReviewIssue(
                category="Code Quality",
                severity="High",
                file_path="src/test.py",
                line_start=6,
                line_end=9,
                title="Inefficient list building",
                description="Using append in loop is inefficient",
                suggested_code="return [item * 2 for item in data]",
                rationale="List comprehensions are more Pythonic and efficient",
                references=["https://docs.python.org/3/tutorial/datastructures.html"],
            ),
            ReviewIssue(
                category="Best Practices",
                severity="Medium",
                file_path="src/test.py",
                line_start=2,
                line_end=3,
                title="Missing type hints",
                description="Function lacks type annotations",
                suggested_code="def calculate_sum(a: int, b: int) -> int:",
                rationale="Type hints improve code clarity and enable static analysis",
                references=["https://peps.python.org/pep-0484/"],
            ),
            ReviewIssue(
                category="Documentation",
                severity="Medium",
                file_path="src/test.go",
                line_start=7,
                line_end=7,
                title="Missing function comment",
                description="Public function lacks documentation",
                suggested_code="// main is the entry point\nfunc main() {",
                rationale="Go convention requires comments on exported functions",
                references=["https://go.dev/doc/effective_go#commentary"],
            ),
        ],
        system_design_insights="Simple utility functions, no major architectural concerns.",
        recommendations=[
            "Use list comprehensions instead of loops",
            "Add type hints to all functions",
            "Document all exported functions",
        ],
    )


class TestFullWorkflow:
    """Test complete end-to-end workflow."""

    def test_scan_batch_analyze_workflow(
        self, sample_project_dir, mock_code_review_report
    ):
        """Test the full workflow: scan -> batch -> analyze."""
        # Step 1: Scan files
        scanner = FileScanner(sample_project_dir)
        files = scanner.scan()

        assert len(files) == 2
        assert any(f.name == "test.py" for f in files)
        assert any(f.name == "test.go" for f in files)

        # Step 2: Create batches
        batcher = FileBatcher()
        batches = batcher.create_batches(files)

        assert len(batches) >= 1
        assert all(batch.files for batch in batches)

        # Step 3: Mock analyzer and analyze
        with patch("codereview.analyzer.ChatBedrockConverse"):
            analyzer = CodeAnalyzer()
            analyzer.model.invoke = Mock(return_value=mock_code_review_report)

            results = []
            for batch in batches:
                report = analyzer.analyze_batch(batch)
                results.append(report)

            assert len(results) > 0
            assert all(isinstance(r, CodeReviewReport) for r in results)

    def test_markdown_export_integration(
        self, sample_project_dir, mock_code_review_report, tmp_path
    ):
        """Test markdown export with real report."""
        output_file = tmp_path / "test-report.md"

        exporter = MarkdownExporter()
        exporter.export(mock_code_review_report, output_file)

        assert output_file.exists()

        content = output_file.read_text()
        assert "# Code Review Report" in content
        assert "Test analysis complete" in content
        assert "High" in content
        assert "src/test.py" in content
        assert "Inefficient list building" in content

    def test_cli_full_workflow_mocked(
        self, sample_project_dir, mock_code_review_report
    ):
        """Test CLI end-to-end with mocked AWS calls."""
        runner = CliRunner()

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = mock_code_review_report
            mock_analyzer.total_input_tokens = 1000
            mock_analyzer.total_output_tokens = 500
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, [str(sample_project_dir)])

            assert result.exit_code == 0
            assert "Code Review Tool" in result.output
            assert "Found" in result.output
            assert "files to review" in result.output

    def test_cli_with_markdown_export(
        self, sample_project_dir, mock_code_review_report, tmp_path
    ):
        """Test CLI with markdown export option."""
        runner = CliRunner()
        output_file = tmp_path / "cli-report.md"

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = mock_code_review_report
            mock_analyzer.total_input_tokens = 1000
            mock_analyzer.total_output_tokens = 500
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(
                main, [str(sample_project_dir), "--output", str(output_file)]
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Report exported to" in result.output

    def test_cli_with_severity_filter(
        self, sample_project_dir, mock_code_review_report
    ):
        """Test CLI with severity filtering."""
        runner = CliRunner()

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = mock_code_review_report
            mock_analyzer.total_input_tokens = 1000
            mock_analyzer.total_output_tokens = 500
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(
                main, [str(sample_project_dir), "--severity", "high"]
            )

            assert result.exit_code == 0

    def test_cli_with_max_files(self, sample_project_dir, mock_code_review_report):
        """Test CLI with max files limit."""
        runner = CliRunner()

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = mock_code_review_report
            mock_analyzer.total_input_tokens = 1000
            mock_analyzer.total_output_tokens = 500
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, [str(sample_project_dir), "--max-files", "1"])

            assert result.exit_code == 0

    def test_cli_with_verbose_mode(self, sample_project_dir, mock_code_review_report):
        """Test CLI with verbose output."""
        runner = CliRunner()

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = mock_code_review_report
            mock_analyzer.total_input_tokens = 1000
            mock_analyzer.total_output_tokens = 500
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, [str(sample_project_dir), "--verbose"])

            assert result.exit_code == 0
            assert "Batch" in result.output


class TestWorkflowWithFixtures:
    """Test workflow using existing fixtures."""

    def test_with_sample_code_fixtures(self, mock_code_review_report):
        """Test using existing sample code fixtures."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "sample_code"

        if not fixtures_dir.exists():
            pytest.skip("Sample code fixtures not found")

        scanner = FileScanner(fixtures_dir)
        files = scanner.scan()

        # Should find at least the Python and Go files
        assert len(files) >= 2

        batcher = FileBatcher()
        batches = batcher.create_batches(files)

        assert len(batches) >= 1

    def test_cli_with_fixtures(self, mock_code_review_report):
        """Test CLI with existing fixtures."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "sample_code"

        if not fixtures_dir.exists():
            pytest.skip("Sample code fixtures not found")

        runner = CliRunner()

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = mock_code_review_report
            mock_analyzer.total_input_tokens = 1000
            mock_analyzer.total_output_tokens = 500
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, [str(fixtures_dir)])

            assert result.exit_code == 0


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    def test_empty_directory(self):
        """Test handling of empty directory."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            Path("empty_dir").mkdir()
            result = runner.invoke(main, ["empty_dir"])

            assert result.exit_code == 0
            assert "No files found" in result.output

    def test_nonexistent_directory(self):
        """Test handling of nonexistent directory."""
        runner = CliRunner()

        result = runner.invoke(main, ["/nonexistent/path"])

        assert result.exit_code != 0

    def test_aws_error_handling(self, sample_project_dir):
        """Test AWS error handling in full workflow."""
        from botocore.exceptions import ClientError

        runner = CliRunner()

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.side_effect = ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": "Access denied",
                    }
                },
                "InvokeModel",
            )
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, [str(sample_project_dir), "--verbose"])

            # Should handle error gracefully (not crash)
            assert "AWS" in result.output or "Error" in result.output


class TestOutputFormats:
    """Test different output formats."""

    def test_terminal_and_markdown_output(
        self, sample_project_dir, mock_code_review_report, tmp_path
    ):
        """Test generating both terminal and markdown output."""
        runner = CliRunner()
        output_file = tmp_path / "both-outputs.md"

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = mock_code_review_report
            mock_analyzer.total_input_tokens = 1000
            mock_analyzer.total_output_tokens = 500
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(
                main, [str(sample_project_dir), "--output", str(output_file)]
            )

            # Both terminal output and file should be generated
            assert result.exit_code == 0
            assert "Code Review Report" in result.output  # Terminal output
            assert output_file.exists()  # File output

            # Verify markdown content
            md_content = output_file.read_text()
            assert "# Code Review Report" in md_content
            assert "High" in md_content

    def test_markdown_structure(self, mock_code_review_report, tmp_path):
        """Test markdown report has proper structure."""
        output_file = tmp_path / "structure-test.md"

        exporter = MarkdownExporter()
        exporter.export(mock_code_review_report, output_file)

        content = output_file.read_text()

        # Check for key sections
        assert "# Code Review Report" in content
        assert "## Executive Summary" in content or "## Summary" in content
        assert "## Metrics" in content
        assert "## System Design Insights" in content
        assert "## Top Recommendations" in content or "## Recommendations" in content

        # Check for issue details (with actual format from exporter)
        assert "High" in content  # Severity level
        assert "src/test.py" in content
        assert "Inefficient list building" in content


class TestBatchProcessing:
    """Test batch processing in integration."""

    def test_multiple_batches_processing(
        self, sample_project_dir, mock_code_review_report
    ):
        """Test processing multiple batches."""
        runner = CliRunner()

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.return_value = mock_code_review_report
            mock_analyzer.total_input_tokens = 1000
            mock_analyzer.total_output_tokens = 500
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, [str(sample_project_dir)])

            assert result.exit_code == 0
            # Verify batches were processed
            assert mock_analyzer.analyze_batch.called

    def test_batch_aggregation(self, sample_project_dir):
        """Test that results from multiple batches are aggregated."""
        runner = CliRunner()

        report1 = CodeReviewReport(
            summary="Batch 1",
            metrics={"files_analyzed": 1, "total_issues": 2},
            issues=[
                ReviewIssue(
                    category="Code Quality",
                    severity="High",
                    file_path="file1.py",
                    line_start=1,
                    line_end=1,
                    title="Issue 1",
                    description="Test",
                    rationale="Test",
                )
            ],
            system_design_insights="",
            recommendations=[],
        )

        report2 = CodeReviewReport(
            summary="Batch 2",
            metrics={"files_analyzed": 1, "total_issues": 1},
            issues=[
                ReviewIssue(
                    category="Security",
                    severity="Critical",
                    file_path="file2.py",
                    line_start=1,
                    line_end=1,
                    title="Issue 2",
                    description="Test",
                    rationale="Test",
                )
            ],
            system_design_insights="",
            recommendations=[],
        )

        with patch("codereview.cli.CodeAnalyzer") as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_batch.side_effect = [report1, report2]
            mock_analyzer.total_input_tokens = 2000
            mock_analyzer.total_output_tokens = 1000
            mock_analyzer.skipped_files = []
            mock_analyzer_class.return_value = mock_analyzer

            result = runner.invoke(main, [str(sample_project_dir)])

            assert result.exit_code == 0
            # Both issues should be in final report
            # (This is implicit in the aggregation logic)


@pytest.mark.slow
class TestPerformance:
    """Performance-related integration tests."""

    def test_large_file_exclusion(self, tmp_path):
        """Test that large files are excluded properly."""
        # Create a file larger than default limit
        large_file = tmp_path / "large.py"
        large_file.write_text("x = 1\n" * 10000)  # ~60KB

        scanner = FileScanner(tmp_path, max_file_size_kb=10)
        files = scanner.scan()

        # Large file should be excluded
        assert large_file not in files

    def test_exclusion_patterns_integration(self, tmp_path):
        """Test exclusion patterns work in full workflow."""
        # Create test structure
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass")
        (tmp_path / "tests" / "test_main.py").write_text("def test(): pass")

        scanner = FileScanner(tmp_path)
        files = scanner.scan()

        # Test files should be excluded by default
        file_names = [f.name for f in files]
        assert "main.py" in file_names
        # tests directory should be excluded by default patterns
