"""Tests for error handling and retry logic."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from codereview.analyzer import CodeAnalyzer
from codereview.batcher import FileBatch
from codereview.models import CodeReviewReport


@pytest.fixture
def sample_batch(tmp_path):
    """Create a sample file batch for testing."""
    test_file = tmp_path / "test_file.py"
    test_file.write_text("def test(): pass")
    return FileBatch(files=[test_file], batch_number=1, total_batches=1, total_tokens=100)


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = Mock()
    provider.total_input_tokens = 0
    provider.total_output_tokens = 0
    provider.reset_state.return_value = None
    return provider


class TestErrorPropagation:
    """Test error propagation from provider to analyzer.

    Note: Retry logic is tested in test_bedrock_provider.py since it's
    implemented in the provider layer, not the analyzer layer.
    """

    def test_successful_analysis(self, mock_provider, sample_batch):
        """Test that successful analysis works correctly."""
        mock_report = CodeReviewReport(
            summary="Test summary",
            metrics={},
            issues=[],
            system_design_insights="",
            recommendations=[],
        )

        mock_provider.analyze_batch.return_value = mock_report

        with patch("codereview.analyzer.ProviderFactory") as mock_factory:
            mock_factory.return_value.create_provider.return_value = mock_provider
            analyzer = CodeAnalyzer(model_name="opus")
            result = analyzer.analyze_batch(sample_batch)

        assert result == mock_report
        assert mock_provider.analyze_batch.call_count == 1

    def test_throttling_error_propagated(self, mock_provider, sample_batch):
        """Test that throttling errors from provider are propagated."""
        throttling_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )

        mock_provider.analyze_batch.side_effect = throttling_error

        with patch("codereview.analyzer.ProviderFactory") as mock_factory:
            mock_factory.return_value.create_provider.return_value = mock_provider
            analyzer = CodeAnalyzer(model_name="opus")

            with pytest.raises(ClientError) as exc_info:
                analyzer.analyze_batch(sample_batch, max_retries=3)

        assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"

    def test_access_denied_error_propagated(self, mock_provider, sample_batch):
        """Test that access denied errors are propagated."""
        access_error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "InvokeModel",
        )

        mock_provider.analyze_batch.side_effect = access_error

        with patch("codereview.analyzer.ProviderFactory") as mock_factory:
            mock_factory.return_value.create_provider.return_value = mock_provider
            analyzer = CodeAnalyzer(model_name="opus")

            with pytest.raises(ClientError) as exc_info:
                analyzer.analyze_batch(sample_batch, max_retries=3)

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    def test_generic_exception_propagated(self, mock_provider, sample_batch):
        """Test that generic exceptions are propagated."""
        generic_error = ValueError("Something went wrong")

        mock_provider.analyze_batch.side_effect = generic_error

        with patch("codereview.analyzer.ProviderFactory") as mock_factory:
            mock_factory.return_value.create_provider.return_value = mock_provider
            analyzer = CodeAnalyzer(model_name="opus")

            with pytest.raises(ValueError) as exc_info:
                analyzer.analyze_batch(sample_batch, max_retries=3)

        assert str(exc_info.value) == "Something went wrong"

    def test_max_retries_parameter_passed_to_provider(self, mock_provider, sample_batch):
        """Test that max_retries parameter is passed to provider."""
        mock_report = CodeReviewReport(
            summary="Test",
            metrics={},
            issues=[],
            system_design_insights="",
            recommendations=[],
        )
        mock_provider.analyze_batch.return_value = mock_report

        with patch("codereview.analyzer.ProviderFactory") as mock_factory:
            mock_factory.return_value.create_provider.return_value = mock_provider
            analyzer = CodeAnalyzer(model_name="opus")
            analyzer.analyze_batch(sample_batch, max_retries=5)

        # Verify max_retries was passed to provider
        call_kwargs = mock_provider.analyze_batch.call_args[1]
        assert call_kwargs["max_retries"] == 5


class TestErrorMessages:
    """Test error message generation."""

    def test_client_error_response_structure(self):
        """Test that ClientError has expected response structure."""
        error = ClientError(
            {"Error": {"Code": "TestError", "Message": "Test message"}}, "TestOperation"
        )

        assert error.response["Error"]["Code"] == "TestError"
        assert error.response["Error"]["Message"] == "Test message"

    def test_access_denied_error(self):
        """Test AccessDeniedException handling."""
        error = ClientError(
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "User is not authorized",
                }
            },
            "InvokeModel",
        )

        error_code = error.response.get("Error", {}).get("Code", "")
        assert error_code == "AccessDeniedException"

    def test_resource_not_found_error(self):
        """Test ResourceNotFoundException handling."""
        error = ClientError(
            {
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Model not found",
                }
            },
            "InvokeModel",
        )

        error_code = error.response.get("Error", {}).get("Code", "")
        assert error_code == "ResourceNotFoundException"


