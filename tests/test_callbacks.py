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
        assert handler._current_parts == []
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
            assert handler._current_parts == []
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
        handler._current_parts = []
        handler._token_count = 0
        handler._live = Mock()

        handler.on_llm_new_token("Hello")
        handler.on_llm_new_token(" world")

        assert "".join(handler._current_parts) == "Hello world"
        assert handler._token_count == 2

    def test_on_llm_new_token_accepts_list_shaped_token(self):
        """langchain-core 1.4.6+ may deliver token as a list of str/dict.

        The supertype signature is ``token: str | list[str | dict]``; a
        handler that assumes str crashes on ``"".join`` or renders
        "['Hello']" garbage. Coerce list parts to their text content.
        """
        handler = StreamingCallbackHandler(verbose=True)
        handler._current_parts = []
        handler._token_count = 0
        handler._live = Mock()

        handler.on_llm_new_token(["Hello", {"type": "text", "text": " world"}])

        assert "".join(handler._current_parts) == "Hello world"
        assert handler._token_count == 1

    def test_block_with_explicit_none_text_contributes_nothing(self):
        """A ``{"text": None}`` block must add nothing, not the literal "None".

        Some providers emit non-text content blocks (reasoning summaries,
        tool-call deltas) with ``"text"`` present but null. ``part.get("text",
        "")`` returns None for those — the default only applies to a *missing*
        key — and ``str(None)`` would splice "None" into the streamed panel.
        """
        handler = StreamingCallbackHandler(verbose=True)
        handler._current_parts = []
        handler._token_count = 0
        handler._live = Mock()

        handler.on_llm_new_token(
            [
                {"type": "text", "text": "Hello"},
                {"type": "reasoning", "text": None},
                {"type": "text", "text": " world"},
            ]
        )

        assert "".join(handler._current_parts) == "Hello world"
        assert "None" not in "".join(handler._current_parts)

    def test_block_without_a_text_key_contributes_nothing(self):
        """A block carrying no text at all is skipped, not stringified."""
        handler = StreamingCallbackHandler(verbose=True)
        handler._current_parts = []
        handler._token_count = 0
        handler._live = Mock()

        handler.on_llm_new_token([{"type": "tool_use", "id": "call_1"}, "ok"])

        assert "".join(handler._current_parts) == "ok"

    def test_non_string_block_text_is_still_rendered(self):
        """Only None is dropped; a non-str value is coerced as before."""
        handler = StreamingCallbackHandler(verbose=True)
        handler._current_parts = []
        handler._token_count = 0
        handler._live = Mock()

        handler.on_llm_new_token([{"type": "text", "text": 42}])

        assert "".join(handler._current_parts) == "42"

    def test_on_llm_new_token_skipped_when_not_verbose(self):
        """Test on_llm_new_token does nothing when not verbose."""
        handler = StreamingCallbackHandler(verbose=False)

        handler.on_llm_new_token("Hello")

        assert handler._current_parts == []
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


class TestProgressCallbackHandlerConcurrency:
    """One handler instance serves every concurrent batch.

    run_review builds a single ProgressCallbackHandler and hands it to the
    analyzer, whose batches run in a ThreadPoolExecutor — so these callbacks
    arrive interleaved from up to four threads with arbitrary overlap. A single
    unsynchronized _status slot made the second overlapping start orphan the
    first Status (no reference left to stop it), which permanently corrupted
    the console: rich's set_live appends to _live_stack and returns whether the
    live is topmost rather than raising, so nothing failed loudly.
    """

    def test_overlapping_runs_share_one_spinner(self):
        """A start while another run is live must not create a second Status."""
        handler = ProgressCallbackHandler()
        first, second = uuid4(), uuid4()

        with patch.object(handler.console, "status") as mock_status:
            mock_status.return_value = Mock()

            handler.on_llm_start({}, [], run_id=first)
            handler.on_llm_start({}, [], run_id=second)

            assert mock_status.call_count == 1, (
                "the second overlapping batch created a second Status; the first "
                "one is now unreachable and can never be stopped"
            )

    def test_spinner_survives_until_the_last_run_finishes(self):
        """The first end must not stop a spinner other batches still need."""
        handler = ProgressCallbackHandler()
        first, second = uuid4(), uuid4()

        with patch.object(handler.console, "status") as mock_status:
            status = Mock()
            mock_status.return_value = status

            handler.on_llm_start({}, [], run_id=first)
            handler.on_llm_start({}, [], run_id=second)
            handler.on_llm_end(Mock(), run_id=first)

            status.stop.assert_not_called()
            assert handler._status is status

            handler.on_llm_end(Mock(), run_id=second)

            status.stop.assert_called_once()
            assert handler._status is None

    def test_an_erroring_batch_does_not_stop_a_healthy_one(self):
        """on_llm_error is refcounted the same way as on_llm_end."""
        handler = ProgressCallbackHandler()
        failing, healthy = uuid4(), uuid4()

        with patch.object(handler.console, "status") as mock_status:
            status = Mock()
            mock_status.return_value = status

            handler.on_llm_start({}, [], run_id=failing)
            handler.on_llm_start({}, [], run_id=healthy)
            handler.on_llm_error(Exception("boom"), run_id=failing)

            status.stop.assert_not_called()

            handler.on_llm_end(Mock(), run_id=healthy)
            status.stop.assert_called_once()

    def test_unmatched_end_is_a_no_op_not_an_underflow(self):
        """An end for a run that never started must not kill a live spinner."""
        handler = ProgressCallbackHandler()
        live = uuid4()

        with patch.object(handler.console, "status") as mock_status:
            status = Mock()
            mock_status.return_value = status

            handler.on_llm_start({}, [], run_id=live)
            handler.on_llm_end(Mock(), run_id=uuid4())  # never started

            status.stop.assert_not_called()

    def test_cleanup_stops_the_spinner_with_runs_still_registered(self):
        """An aborted run never delivers on_llm_end; finally must still clean up."""
        handler = ProgressCallbackHandler()

        with patch.object(handler.console, "status") as mock_status:
            status = Mock()
            mock_status.return_value = status

            handler.on_llm_start({}, [], run_id=uuid4())
            handler.on_llm_start({}, [], run_id=uuid4())
            handler.cleanup()

            status.stop.assert_called_once()
            assert handler._status is None
            assert not handler._active_runs

    def test_concurrent_batches_leave_the_console_live_stack_empty(self):
        """End-to-end against real rich objects, not mocks.

        The mock-based tests above pin the handler's own bookkeeping; this one
        pins the consequence that actually broke the terminal — a stale entry
        left on Console._live_stack, its Live still `_started`, and its refresh
        thread never joined.
        """
        import io
        import threading
        import time

        console = Console(file=io.StringIO(), force_terminal=True, width=100)
        handler = ProgressCallbackHandler(console=console)
        threads_before = set(threading.enumerate())

        def batch(start_delay: float) -> None:
            run_id = uuid4()
            time.sleep(start_delay)
            handler.on_llm_start({}, [], run_id=run_id)
            time.sleep(0.2)  # outlives the other batch's start
            handler.on_llm_end(Mock(), run_id=run_id)

        workers = [
            threading.Thread(target=batch, args=(delay,)) for delay in (0.0, 0.05)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        assert console._live_stack == [], (
            "a Live was left on the console's stack after every batch finished; "
            "the enclosing Progress's clear_live() will now pop the wrong entry"
        )
        assert handler._status is None
        leaked = [
            t.name
            for t in threading.enumerate()
            if t not in threads_before and t not in workers and t.is_alive()
        ]
        assert not leaked, f"leaked rich refresh thread(s): {leaked}"


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
