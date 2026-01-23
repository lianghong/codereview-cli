"""Tests for error handling and retry logic."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from codereview.analyzer import CodeAnalyzer
from codereview.batcher import FileBatch
from codereview.models import CodeReviewReport


@pytest.fixture
def sample_batch():
    """Create a sample file batch for testing."""
    files = [Path("test_file.py")]
    return FileBatch(files=files, batch_number=1, total_batches=1, total_tokens=100)


@pytest.fixture
def mock_analyzer():
    """Create a mocked CodeAnalyzer."""
    with patch("codereview.analyzer.ChatBedrockConverse"):
        analyzer = CodeAnalyzer()
        return analyzer


class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    def test_successful_analysis_on_first_try(self, mock_analyzer, sample_batch):
        """Test that successful analysis doesn't retry."""
        mock_report = CodeReviewReport(
            summary="Test summary",
            metrics={},
            issues=[],
            system_design_insights="",
            recommendations=[],
        )

        mock_analyzer.model.invoke = Mock(return_value=mock_report)

        result = mock_analyzer.analyze_batch(sample_batch)

        assert result == mock_report
        assert mock_analyzer.model.invoke.call_count == 1

    def test_retry_on_throttling_exception(self, mock_analyzer, sample_batch):
        """Test retry logic for throttling exceptions."""
        mock_report = CodeReviewReport(
            summary="Test summary",
            metrics={},
            issues=[],
            system_design_insights="",
            recommendations=[],
        )

        # First call raises throttling error, second succeeds
        throttling_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )

        mock_analyzer.model.invoke = Mock(side_effect=[throttling_error, mock_report])

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = mock_analyzer.analyze_batch(sample_batch, max_retries=3)

        assert result == mock_report
        assert mock_analyzer.model.invoke.call_count == 2

    def test_exhausted_retries_raises_error(self, mock_analyzer, sample_batch):
        """Test that exhausted retries raise the last error."""
        throttling_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )

        mock_analyzer.model.invoke = Mock(side_effect=throttling_error)

        with patch("time.sleep"):
            with pytest.raises(ClientError) as exc_info:
                mock_analyzer.analyze_batch(sample_batch, max_retries=2)

        assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"
        # Should try initial + 2 retries = 3 times
        assert mock_analyzer.model.invoke.call_count == 3

    def test_exponential_backoff_timing(self, mock_analyzer, sample_batch):
        """Test that exponential backoff increases wait time."""
        throttling_error = ClientError(
            {
                "Error": {
                    "Code": "TooManyRequestsException",
                    "Message": "Too many requests",
                }
            },
            "InvokeModel",
        )

        mock_analyzer.model.invoke = Mock(side_effect=throttling_error)

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ClientError):
                mock_analyzer.analyze_batch(sample_batch, max_retries=3)

            # Should have called sleep with 1, 2, 4 seconds (2^0, 2^1, 2^2)
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert sleep_calls == [1, 2, 4]

    def test_non_throttling_error_no_retry(self, mock_analyzer, sample_batch):
        """Test that non-throttling errors don't trigger retries."""
        access_error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "InvokeModel",
        )

        mock_analyzer.model.invoke = Mock(side_effect=access_error)

        with pytest.raises(ClientError) as exc_info:
            mock_analyzer.analyze_batch(sample_batch, max_retries=3)

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
        # Should only try once (no retries for non-throttling errors)
        assert mock_analyzer.model.invoke.call_count == 1

    def test_generic_exception_no_retry(self, mock_analyzer, sample_batch):
        """Test that generic exceptions don't trigger retries."""
        generic_error = ValueError("Something went wrong")

        mock_analyzer.model.invoke = Mock(side_effect=generic_error)

        with pytest.raises(ValueError):
            mock_analyzer.analyze_batch(sample_batch, max_retries=3)

        # Should only try once
        assert mock_analyzer.model.invoke.call_count == 1


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


class TestMaxRetriesParameter:
    """Test max_retries parameter behavior."""

    def test_zero_retries(self, mock_analyzer, sample_batch):
        """Test that max_retries=0 means no retries."""
        throttling_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )

        mock_analyzer.model.invoke = Mock(side_effect=throttling_error)

        with patch("time.sleep"):
            with pytest.raises(ClientError):
                mock_analyzer.analyze_batch(sample_batch, max_retries=0)

        # Should only try once (initial attempt, no retries)
        assert mock_analyzer.model.invoke.call_count == 1

    def test_custom_max_retries(self, mock_analyzer, sample_batch):
        """Test custom max_retries value."""
        throttling_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )

        mock_analyzer.model.invoke = Mock(side_effect=throttling_error)

        with patch("time.sleep"):
            with pytest.raises(ClientError):
                mock_analyzer.analyze_batch(sample_batch, max_retries=5)

        # Should try initial + 5 retries = 6 times
        assert mock_analyzer.model.invoke.call_count == 6
