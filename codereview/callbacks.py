"""LangChain callbacks for streaming output and progress tracking."""

import logging
import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Re-export BaseCallbackHandler for convenience
__all__ = ["BaseCallbackHandler", "StreamingCallbackHandler", "ProgressCallbackHandler"]


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM output tokens.

    Provides real-time feedback during LLM calls by displaying
    tokens as they arrive. Useful for long-running code reviews.

    Attributes:
        console: Rich console for output.
        verbose: Whether to display streaming output.
        _current_text: Accumulated text from tokens during streaming.
        _live: Rich Live display context for real-time updates.
        _token_count: Number of tokens received in current stream.
    """

    def __init__(self, console: Console | None = None, verbose: bool = True):
        """Initialize streaming callback handler.

        Args:
            console: Rich console for output (creates new one if None)
            verbose: Whether to display streaming output
        """
        self.console = console or Console()
        self.verbose = verbose
        self._current_parts: list[str] = []
        self._live: Live | None = None
        self._token_count = 0

    def cleanup(self) -> None:
        """Clean up any active Live display.

        Call this to ensure terminal state is restored if an error occurs
        during LLM processing. Safe to call multiple times.
        """
        if self._live:
            try:
                self._live.stop()
            # PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
            except OSError, RuntimeError:
                # OSError: terminal I/O errors (e.g., broken pipe, write blocking)
                # RuntimeError: threading issues during shutdown
                pass  # Best effort cleanup - expected failure modes
            except Exception:
                # Log unexpected errors but don't propagate during cleanup
                logging.debug(
                    "Unexpected error during Live display cleanup", exc_info=True
                )
            self._live = None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM starts generating."""
        # Suppress unused parameter warnings
        _ = (serialized, prompts, run_id, parent_run_id, tags, metadata, kwargs)

        if not self.verbose:
            return None

        self._current_parts = []
        self._token_count = 0
        try:
            self._live = Live(
                Panel(
                    Text("Analyzing...", style="dim"),
                    title="[cyan]AI Analysis[/cyan]",
                    border_style="cyan",
                ),
                console=self.console,
                refresh_per_second=10,
            )
            self._live.start()
        except Exception:
            # If Live creation fails, ensure we don't leave partial state
            self._live = None
            raise
        return None

    @staticmethod
    def _block_text(part: dict[str, Any]) -> str:
        """Text of one content block, or "" when it carries none.

        ``part.get("text", "")`` is not enough: a block can hold an explicit
        ``{"text": None}``, and ``str()`` would turn that into the literal
        ``"None"`` in the stream.
        """
        value = part.get("text")
        return str(value) if value is not None else ""

    def on_llm_new_token(
        self, token: str | list[str | dict[str, Any]], **kwargs: Any
    ) -> None:
        """Called for each new token generated.

        langchain-core types ``token`` as ``str | list[str | dict]`` — content
        blocks arrive as a list whose dict parts carry their text under
        ``"text"``. Coerce to plain text before appending.

        A block's ``"text"`` may be present but ``None`` (some providers emit
        that for non-text blocks such as reasoning or tool-call deltas), so the
        value is checked rather than defaulted: ``str(None)`` would splice a
        literal ``"None"`` into the streamed panel.
        """
        _ = kwargs  # Suppress unused parameter warning

        if not self.verbose or not self._live:
            return

        if isinstance(token, list):
            text = "".join(
                part if isinstance(part, str) else self._block_text(part)
                for part in token
            )
        else:
            text = token

        self._current_parts.append(text)
        self._token_count += 1

        # Update display with current text (truncated for readability)
        display_text = "".join(self._current_parts)
        if len(display_text) > 500:
            display_text = "..." + display_text[-500:]

        self._live.update(
            Panel(
                Text(display_text),
                title=f"[cyan]AI Analysis[/cyan] [dim]({self._token_count} tokens)[/dim]",
                border_style="cyan",
            )
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM finishes generating."""
        _ = (
            response,
            run_id,
            parent_run_id,
            kwargs,
        )  # Suppress unused parameter warning

        self.cleanup()

        if self.verbose:
            self.console.print(
                f"[green]✓[/green] Analysis complete ({self._token_count} tokens)"
            )
        return None

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM encounters an error."""
        _ = (run_id, parent_run_id, kwargs)  # Suppress unused parameter warning

        self.cleanup()

        if self.verbose:
            self.console.print(f"[red]✗[/red] Analysis error: {error}")
        return None


class ProgressCallbackHandler(BaseCallbackHandler):
    """Simplified callback handler for progress indication only.

    Shows a spinner during LLM calls without streaming individual tokens.
    Lower overhead than full streaming.

    **One handler instance serves every concurrent batch** — ``run_review``
    creates a single handler and passes it to the analyzer, whose batches run in
    a ``ThreadPoolExecutor``. So the LLM lifecycle callbacks arrive interleaved
    from several threads and can overlap arbitrarily (batch A starts, batch B
    starts, A ends, B ends). The display state is therefore refcounted by
    ``run_id`` under a lock, and at most one ``Status`` exists at a time:

    A single ``self._status`` slot that each ``on_llm_start`` overwrote leaked
    the previous ``Status`` — the reference was gone, so no ``stop()`` could
    ever reach it. Rich's ``Console.set_live`` appends to ``_live_stack`` and
    returns whether the live is topmost rather than raising, so the overlap
    produced no error: it left a permanently-``_started`` ``Live`` on the
    console's stack (plus its refresh thread, and a pushed render hook), and the
    enclosing ``Progress``'s own ``clear_live`` then popped the wrong entry.
    Terminal state stayed corrupted for the rest of the process.
    """

    def __init__(self, console: Console | None = None):
        """Initialize progress callback handler.

        Args:
            console: Rich console for output (creates new one if None)
        """
        self.console = console or Console()
        self._status: Any = None
        # Guards _status and _active_runs together: the pair must be consistent
        # or two concurrent starts race to create two spinners.
        self._lock = threading.Lock()
        # Refcount by run_id rather than an int: a set makes a duplicate start
        # idempotent and an unmatched end a no-op instead of an underflow that
        # would stop the spinner while other batches are still running.
        self._active_runs: set[UUID] = set()

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM starts generating.

        The first concurrent run starts the spinner; later overlapping runs
        just register themselves against it.
        """
        _ = (serialized, prompts, parent_run_id, tags, metadata, kwargs)

        with self._lock:
            self._active_runs.add(run_id)
            if self._status is not None:
                return None
            status = self.console.status(
                "[cyan]Analyzing code...[/cyan]",
                spinner="dots",
            )
            self._status = status
        status.start()
        return None

    def _finish_run(self, run_id: UUID) -> None:
        """Drop *run_id* and stop the spinner once no runs are left.

        Stopping outside the lock keeps terminal I/O off the critical section;
        only the thread that swapped ``_status`` to None holds a reference, so
        exactly one ``stop()`` happens.
        """
        with self._lock:
            self._active_runs.discard(run_id)
            if self._active_runs or self._status is None:
                return
            status, self._status = self._status, None
        status.stop()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM finishes generating."""
        _ = (response, parent_run_id, kwargs)

        self._finish_run(run_id)
        return None

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM encounters an error."""
        _ = (error, parent_run_id, kwargs)

        self._finish_run(run_id)
        return None

    def cleanup(self) -> None:
        """Clean up any active status display.

        Called from ``run_review``'s ``finally`` block, so it must stop the
        spinner regardless of how many runs are still registered — an aborted
        run never delivers its ``on_llm_end``.
        """
        with self._lock:
            self._active_runs.clear()
            status, self._status = self._status, None
        if status is None:
            return
        try:
            status.stop()
        # PEP 758 syntax (Python 3.14+): unparenthesized multi-exception catch
        except OSError, RuntimeError:
            # OSError: terminal I/O errors; RuntimeError: shutdown threading
            pass  # Best effort cleanup - expected failure modes
        except Exception:
            # cleanup() runs in a finally block during error handling; an
            # unexpected exception here would mask the original error.
            logging.debug(
                "Unexpected error during status display cleanup", exc_info=True
            )
