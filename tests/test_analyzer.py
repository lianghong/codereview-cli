from unittest.mock import Mock, patch

import pytest

from codereview.analyzer import CodeAnalyzer
from codereview.batcher import FileBatch
from codereview.models import CodeReviewReport


@pytest.fixture
def mock_bedrock_client():
    """Mock AWS Bedrock client."""
    with patch("codereview.analyzer.ChatBedrockConverse") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_batch(tmp_path):
    """Create sample batch with test file."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def foo():\n    pass\n")

    return FileBatch(files=[test_file], batch_number=1, total_batches=1)


def test_analyzer_initialization(mock_bedrock_client):
    """Test analyzer can be initialized."""
    analyzer = CodeAnalyzer(region="us-west-2")
    assert analyzer is not None


def test_prepare_batch_context(sample_batch, mock_bedrock_client):
    """Test preparing context from batch."""
    analyzer = CodeAnalyzer()
    context = analyzer._prepare_batch_context(sample_batch)

    assert "test.py" in context
    assert "def foo()" in context
    assert "Batch 1/1" in context


def test_analyze_batch_returns_report(sample_batch, mock_bedrock_client):
    """Test analyze_batch returns CodeReviewReport."""
    # Mock LLM response
    mock_response = CodeReviewReport(
        summary="No issues found",
        metrics={"files": 1, "issues": 0},
        issues=[],
        system_design_insights="Simple code",
        recommendations=[],
    )

    mock_bedrock_client.with_structured_output.return_value.invoke.return_value = (
        mock_response
    )

    analyzer = CodeAnalyzer()
    result = analyzer.analyze_batch(sample_batch)

    assert isinstance(result, CodeReviewReport)
    assert result.summary == "No issues found"
