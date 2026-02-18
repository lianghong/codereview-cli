from unittest.mock import Mock, patch

import pytest

from codereview.analyzer import CodeAnalyzer
from codereview.batcher import FileBatch
from codereview.models import CodeReviewReport, ReviewMetrics


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = Mock()
    provider.analyze_batch.return_value = CodeReviewReport(
        summary="Test summary",
        metrics=ReviewMetrics(files_analyzed=1, total_issues=0),
        issues=[],
        system_design_insights="Test insights",
        recommendations=[],
    )
    provider.total_input_tokens = 100
    provider.total_output_tokens = 50
    provider.get_model_display_name.return_value = "Test Model"
    provider.estimate_cost.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "input_cost": 0.5,
        "output_cost": 1.25,
        "total_cost": 1.75,
    }
    provider.reset_state.return_value = None
    return provider


@pytest.fixture
def sample_batch(tmp_path):
    """Create sample batch with test file."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def foo():\n    pass\n")

    return FileBatch(files=[test_file], batch_number=1, total_batches=1)


def test_analyzer_initialization_with_model_name(mock_provider):
    """Test new model_name parameter."""
    with patch("codereview.analyzer.ProviderFactory") as mock_factory:
        mock_factory.return_value.create_provider.return_value = mock_provider

        analyzer = CodeAnalyzer(model_name="opus")

        assert analyzer.model_name == "opus"
        mock_factory.return_value.create_provider.assert_called_once_with(
            "opus", None, callbacks=None, project_context=None
        )


def test_analyzer_initialization_with_temperature(mock_provider):
    """Test temperature parameter is passed to provider."""
    with patch("codereview.analyzer.ProviderFactory") as mock_factory:
        mock_factory.return_value.create_provider.return_value = mock_provider

        analyzer = CodeAnalyzer(model_name="sonnet", temperature=0.5)

        assert analyzer.temperature == 0.5
        mock_factory.return_value.create_provider.assert_called_once_with(
            "sonnet", 0.5, callbacks=None, project_context=None
        )


def test_analyzer_legacy_parameters_deprecated():
    """Test legacy parameters show deprecation warning."""
    with (
        patch("codereview.analyzer.ProviderFactory") as mock_factory,
        pytest.warns(DeprecationWarning, match="model_id.*deprecated"),
    ):
        mock_factory.return_value.create_provider.return_value = Mock()
        CodeAnalyzer(model_id="global.anthropic.claude-opus-4-6-v1")


def test_analyzer_legacy_model_id_mapping(mock_provider):
    """Test legacy model_id is mapped to new short name."""
    with (
        patch("codereview.analyzer.ProviderFactory") as mock_factory,
        pytest.warns(DeprecationWarning),
    ):
        mock_factory.return_value.create_provider.return_value = mock_provider

        analyzer = CodeAnalyzer(model_id="global.anthropic.claude-opus-4-6-v1")

        # Should map to "opus" (Opus 4.6 legacy ID)
        assert analyzer.model_name == "opus"
        mock_factory.return_value.create_provider.assert_called_once_with(
            "opus", None, callbacks=None, project_context=None
        )


def test_analyzer_delegates_to_provider(mock_provider, sample_batch):
    """Test analyzer delegates analyze_batch to provider."""
    with patch("codereview.analyzer.ProviderFactory") as mock_factory:
        mock_factory.return_value.create_provider.return_value = mock_provider

        analyzer = CodeAnalyzer(model_name="opus")
        result = analyzer.analyze_batch(sample_batch)

        # Verify provider was called
        mock_provider.analyze_batch.assert_called_once()
        call_args = mock_provider.analyze_batch.call_args
        assert call_args[1]["batch_number"] == 1
        assert call_args[1]["total_batches"] == 1
        assert call_args[1]["max_retries"] == 3
        # Verify file content was passed
        files_content = call_args[1]["files_content"]
        assert len(files_content) == 1
        assert "test.py" in str(list(files_content.keys())[0])
        assert "def foo()" in list(files_content.values())[0]

        # Verify result
        assert isinstance(result, CodeReviewReport)
        assert result.summary == "Test summary"


def test_analyzer_tracks_skipped_files(mock_provider, tmp_path):
    """Test analyzer tracks files that fail to read."""
    # Create a batch with an unreadable file (nonexistent)
    test_file = tmp_path / "test.py"
    test_file.write_text("def foo(): pass")
    bad_file = tmp_path / "nonexistent.py"

    batch = FileBatch(files=[test_file, bad_file], batch_number=1, total_batches=1)

    with patch("codereview.analyzer.ProviderFactory") as mock_factory:
        mock_factory.return_value.create_provider.return_value = mock_provider

        analyzer = CodeAnalyzer(model_name="opus")
        analyzer.analyze_batch(batch)

        # Should have tracked the skipped file
        assert len(analyzer.skipped_files) == 1
        assert "nonexistent.py" in analyzer.skipped_files[0][0]


def test_analyzer_properties_delegate_to_provider(mock_provider):
    """Test properties delegate to provider."""
    with patch("codereview.analyzer.ProviderFactory") as mock_factory:
        mock_factory.return_value.create_provider.return_value = mock_provider

        analyzer = CodeAnalyzer(model_name="opus")

        assert analyzer.total_input_tokens == 100
        assert analyzer.total_output_tokens == 50
        assert analyzer.get_model_display_name() == "Test Model"
        cost = analyzer.estimate_cost()
        assert cost["total_cost"] == 1.75


def test_analyzer_reset_state(mock_provider, sample_batch):
    """Test reset_state delegates to provider and clears skipped files."""
    with patch("codereview.analyzer.ProviderFactory") as mock_factory:
        mock_factory.return_value.create_provider.return_value = mock_provider

        analyzer = CodeAnalyzer(model_name="opus")

        # Add some skipped files
        analyzer.skipped_files = [("file1.py", "error1"), ("file2.py", "error2")]

        # Reset state
        analyzer.reset_state()

        # Verify provider reset was called
        mock_provider.reset_state.assert_called_once()

        # Verify skipped files were cleared
        assert len(analyzer.skipped_files) == 0


def test_analyzer_with_custom_factory(mock_provider):
    """Test analyzer accepts custom provider factory."""
    custom_factory = Mock()
    custom_factory.create_provider.return_value = mock_provider

    analyzer = CodeAnalyzer(model_name="opus", provider_factory=custom_factory)

    assert analyzer.factory == custom_factory
    custom_factory.create_provider.assert_called_once_with(
        "opus", None, callbacks=None, project_context=None
    )


def test_passes_project_context_to_provider() -> None:
    """Should pass project_context to provider factory."""
    with patch("codereview.analyzer.ProviderFactory") as mock_factory_class:
        mock_factory = Mock()
        mock_provider = Mock()
        mock_factory.create_provider.return_value = mock_provider
        mock_factory_class.return_value = mock_factory

        readme_content = "# Test Project"
        analyzer = CodeAnalyzer(model_name="opus", project_context=readme_content)

        mock_factory.create_provider.assert_called_once_with(
            "opus", None, callbacks=None, project_context=readme_content
        )
        assert analyzer is not None  # Verify analyzer was created
