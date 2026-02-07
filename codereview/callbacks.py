"""LangChain callbacks for streaming output and progress tracking."""

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
            except (OSError, RuntimeError):  # fmt: skip
                # OSError: terminal I/O errors (e.g., broken pipe, write blocking)
                # RuntimeError: threading issues during shutdown
                pass  # Best effort cleanup - expected failure modes
            except Exception:
                # Log unexpected errors but don't propagate during cleanup
                import logging

                logging.debug(
                    "Unexpected error during Live display cleanup", exc_info=True
                )
            self._live = None

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection (not guaranteed to run)."""
        self.cleanup()

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

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called for each new token generated."""
        _ = kwargs  # Suppress unused parameter warning

        if not self.verbose or not self._live:
            return

        self._current_parts.append(token)
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
    """

    def __init__(self, console: Console | None = None):
        """Initialize progress callback handler.

        Args:
            console: Rich console for output (creates new one if None)
        """
        self.console = console or Console()
        self._status: Any = None

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
        _ = (serialized, prompts, run_id, parent_run_id, tags, metadata, kwargs)

        self._status = self.console.status(
            "[cyan]Analyzing code...[/cyan]",
            spinner="dots",
        )
        self._status.start()
        return None

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM finishes generating."""
        _ = (response, run_id, parent_run_id, kwargs)

        if self._status:
            self._status.stop()
            self._status = None
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
        _ = (error, run_id, parent_run_id, kwargs)

        if self._status:
            self._status.stop()
            self._status = None
        return None

    def cleanup(self) -> None:
        """Clean up any active status display."""
        if self._status:
            try:
                self._status.stop()
            except (OSError, RuntimeError):  # fmt: skip
                pass
            self._status = None

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection (not guaranteed to run)."""
        self.cleanup()
