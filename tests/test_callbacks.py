"""Tests for LangChain callback handlers."""

from unittest.mock import Mock, patch
from uuid import uuid4

from rich.console import Console

from codereview.callbacks import ProgressCallbackHandler, StreamingCallbackHandler


class TestStreamingCallbackHandler:
    """Tests for StreamingCallbackHandler."""

    def test_initialization_with_defaults(self):
        """Test handler initialization with default parameters."""
        handler = StreamingCallbackHandler()

        assert handler.verbose is True
        assert handler._current_text == ""
        assert handler._token_count == 0
        assert handler._live is None

    def test_initialization_with_custom_console(self):
        """Test handler initialization with custom console."""
        console = Console(force_terminal=True)
        handler = StreamingCallbackHandler(console=console, verbose=False)

        assert handler.console == console
        assert handler.verbose is False

    def test_on_llm_start_creates_live_display(self):
        """Test on_llm_start creates live display when verbose."""
        handler = StreamingCallbackHandler(verbose=True)

        with patch.object(handler, "_live"):
            handler.on_llm_start({}, ["test prompt"], run_id=uuid4())

            # After on_llm_start, live should be set (not None)
            assert handler._current_text == ""
            assert handler._token_count == 0

    def test_on_llm_start_skipped_when_not_verbose(self):
        """Test on_llm_start does nothing when not verbose."""
        handler = StreamingCallbackHandler(verbose=False)

        handler.on_llm_start({}, ["test prompt"], run_id=uuid4())

        # Live should not be created
        assert handler._live is None

    def test_on_llm_new_token_updates_state(self):
        """Test on_llm_new_token updates internal state."""
        handler = StreamingCallbackHandler(verbose=True)

        # Simulate starting
        handler._current_text = ""
        handler._token_count = 0
        handler._live = Mock()

        handler.on_llm_new_token("Hello")
        handler.on_llm_new_token(" world")

        assert handler._current_text == "Hello world"
        assert handler._token_count == 2

    def test_on_llm_new_token_skipped_when_not_verbose(self):
        """Test on_llm_new_token does nothing when not verbose."""
        handler = StreamingCallbackHandler(verbose=False)

        handler.on_llm_new_token("Hello")

        assert handler._current_text == ""
        assert handler._token_count == 0

    def test_on_llm_end_stops_live_display(self):
        """Test on_llm_end stops live display."""
        handler = StreamingCallbackHandler(verbose=True)
        mock_live = Mock()
        handler._live = mock_live
        handler._token_count = 10

        handler.on_llm_end(Mock(), run_id=uuid4())

        mock_live.stop.assert_called_once()
        assert handler._live is None

    def test_on_llm_error_stops_live_display(self):
        """Test on_llm_error stops live display."""
        handler = StreamingCallbackHandler(verbose=True)
        mock_live = Mock()
        handler._live = mock_live

        handler.on_llm_error(Exception("Test error"), run_id=uuid4())

        mock_live.stop.assert_called_once()
        assert handler._live is None

    def test_cleanup_stops_live_display(self):
        """Test cleanup method stops live display."""
        handler = StreamingCallbackHandler(verbose=True)
        mock_live = Mock()
        handler._live = mock_live

        handler.cleanup()

        mock_live.stop.assert_called_once()
        assert handler._live is None

    def test_cleanup_safe_when_no_live(self):
        """Test cleanup is safe to call when no live display exists."""
        handler = StreamingCallbackHandler(verbose=True)

        # Should not raise
        handler.cleanup()

        assert handler._live is None

    def test_cleanup_handles_stop_exception(self):
        """Test cleanup handles exceptions from live.stop()."""
        handler = StreamingCallbackHandler(verbose=True)
        mock_live = Mock()
        mock_live.stop.side_effect = Exception("Stop failed")
        handler._live = mock_live

        # Should not raise
        handler.cleanup()

        assert handler._live is None


class TestProgressCallbackHandler:
    """Tests for ProgressCallbackHandler."""

    def test_initialization_with_defaults(self):
        """Test handler initialization with default parameters."""
        handler = ProgressCallbackHandler()

        assert handler._status is None

    def test_initialization_with_custom_console(self):
        """Test handler initialization with custom console."""
        console = Console(force_terminal=True)
        handler = ProgressCallbackHandler(console=console)

        assert handler.console == console

    def test_on_llm_start_creates_status(self):
        """Test on_llm_start creates status spinner."""
        handler = ProgressCallbackHandler()

        with patch.object(handler.console, "status") as mock_status:
            mock_status_instance = Mock()
            mock_status.return_value = mock_status_instance

            handler.on_llm_start({}, ["test prompt"], run_id=uuid4())

            mock_status.assert_called_once()
            mock_status_instance.start.assert_called_once()

    def test_on_llm_end_stops_status(self):
        """Test on_llm_end stops status spinner."""
        handler = ProgressCallbackHandler()
        mock_status = Mock()
        handler._status = mock_status

        handler.on_llm_end(Mock(), run_id=uuid4())

        mock_status.stop.assert_called_once()
        assert handler._status is None

    def test_on_llm_error_stops_status(self):
        """Test on_llm_error stops status spinner."""
        handler = ProgressCallbackHandler()
        mock_status = Mock()
        handler._status = mock_status

        handler.on_llm_error(Exception("Test error"), run_id=uuid4())

        mock_status.stop.assert_called_once()
        assert handler._status is None


class TestCallbackIntegration:
    """Integration tests for callbacks with providers."""

    def test_callbacks_passed_to_analyzer(self):
        """Test callbacks are passed through analyzer to provider."""
        from codereview.analyzer import CodeAnalyzer

        with patch("codereview.analyzer.ProviderFactory") as mock_factory:
            mock_provider = Mock()
            mock_factory.return_value.create_provider.return_value = mock_provider

            callback = StreamingCallbackHandler(verbose=False)
            analyzer = CodeAnalyzer(model_name="opus", callbacks=[callback])

            # Verify callbacks were passed to factory
            mock_factory.return_value.create_provider.assert_called_once_with(
                "opus", None, callbacks=[callback], project_context=None
            )
            assert analyzer.callbacks == [callback]
