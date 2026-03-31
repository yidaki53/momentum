"""Focus timer and break countdown logic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from momentum.models import TimerConfig


class SessionKind(str, Enum):
    """Supported timer session types."""

    FOCUS = "focus"
    BREAK = "break"


@dataclass(frozen=True)
class TimerSession:
    """A single timer request."""

    kind: SessionKind
    minutes: int
    label: str
    task_id: int | None = None


@dataclass(frozen=True)
class TimerOutcome:
    """Observable result of a timer run."""

    completed: bool
    elapsed_seconds: int
    total_seconds: int
    kind: SessionKind
    task_id: int | None = None
    completion_message: str | None = None


class ClockPort(Protocol):
    """Sleep boundary for timer progression."""

    def sleep_one_second(self) -> None:
        """Advance the clock by one second."""


class TimerProgressPort(Protocol):
    """Progress reporting boundary."""

    def start(self, *, label: str, total_seconds: int) -> None:
        """Open progress reporting for a timer."""

    def advance(self, seconds: int = 1) -> None:
        """Advance the rendered timer state."""

    def interrupted(self, *, elapsed_seconds: int, total_seconds: int) -> None:
        """Finalize progress reporting for an interrupted timer."""

    def complete(self) -> None:
        """Finalize progress reporting for a completed timer."""


class EncouragementPort(Protocol):
    """Completion message boundary."""

    def completion_message_for(self, kind: SessionKind) -> str:
        """Select a completion message for the given timer kind."""
        ...

    def deliver(self, message: str) -> None:
        """Deliver a completion message to the active UI."""
        ...


class SystemClock:
    """Production clock implementation backed by time.sleep."""

    def sleep_one_second(self) -> None:
        time.sleep(1)


class RichTimerProgress:
    """Rich console progress adapter."""

    def __init__(
        self,
        *,
        progress_factory: Callable[[], Any] | None = None,
        interruption_printer: Callable[[str], None] | None = None,
    ) -> None:
        from momentum.ui.display import create_timer_progress, print_warning

        self._progress_factory = progress_factory or create_timer_progress
        self._interruption_printer = interruption_printer or print_warning
        self._progress: Any | None = None
        self._task_id: Any | None = None

    def start(self, *, label: str, total_seconds: int) -> None:
        progress = self._progress_factory()
        progress.__enter__()
        self._progress = progress
        self._task_id = progress.add_task(label, total=total_seconds)

    def advance(self, seconds: int = 1) -> None:
        if self._progress is None or self._task_id is None:
            return
        self._progress.advance(self._task_id, seconds)

    def interrupted(self, *, elapsed_seconds: int, total_seconds: int) -> None:
        self._close_progress()
        self._interruption_printer("Timer stopped early.")

    def complete(self) -> None:
        self._close_progress()

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.__exit__(None, None, None)
            self._progress = None
            self._task_id = None


class ConsoleEncouragement:
    """Rich console encouragement adapter."""

    _focus_message: Callable[[], str]
    _break_message: Callable[[], str]
    _message_sink: Callable[[str], None]
    _bell: Callable[[], None]

    def __init__(
        self,
        *,
        focus_message: Callable[[], str] | None = None,
        break_message: Callable[[], str] | None = None,
        message_sink: Callable[[str], None] | None = None,
        bell: Callable[[], None] | None = None,
    ) -> None:
        from momentum.encouragement import get_break_message, get_nudge
        from momentum.ui.display import console, print_nudge

        resolved_focus_message: Callable[[], str] = focus_message or get_nudge
        resolved_break_message: Callable[[], str] = break_message or get_break_message

        self._focus_message = resolved_focus_message
        self._break_message = resolved_break_message
        self._message_sink = message_sink or print_nudge
        self._bell = bell or (lambda: console.print("\a", end=""))

    def completion_message_for(self, kind: SessionKind) -> str:
        message: str | None
        if kind == SessionKind.BREAK:
            message = self._break_message()
        else:
            message = self._focus_message()
        return message or "Session complete."

    def deliver(self, message: str) -> None:
        self._bell()
        self._message_sink(message)


class TimerService:
    """Deep timer service that owns countdown orchestration."""

    def __init__(
        self,
        clock: ClockPort,
        progress: TimerProgressPort,
        encouragement: EncouragementPort,
    ) -> None:
        self._clock = clock
        self._progress = progress
        self._encouragement = encouragement

    def run(self, session: TimerSession) -> TimerOutcome:
        """Run a focus or break timer session."""
        total_seconds = session.minutes * 60
        label = session.label
        if session.task_id is not None:
            label = f"{session.label} (task #{session.task_id})"

        elapsed_seconds = 0
        self._progress.start(label=label, total_seconds=total_seconds)
        try:
            for _ in range(total_seconds):
                self._clock.sleep_one_second()
                elapsed_seconds += 1
                self._progress.advance(1)
        except KeyboardInterrupt:
            self._progress.interrupted(
                elapsed_seconds=elapsed_seconds,
                total_seconds=total_seconds,
            )
            return TimerOutcome(
                completed=False,
                elapsed_seconds=elapsed_seconds,
                total_seconds=total_seconds,
                kind=session.kind,
                task_id=session.task_id,
            )

        self._progress.complete()
        completion_message = self._encouragement.completion_message_for(session.kind)
        self._encouragement.deliver(completion_message)
        return TimerOutcome(
            completed=True,
            elapsed_seconds=elapsed_seconds,
            total_seconds=total_seconds,
            kind=session.kind,
            task_id=session.task_id,
            completion_message=completion_message,
        )

    def run_focus(
        self,
        *,
        minutes: int = 15,
        task_id: int | None = None,
        label: str = "Focus",
    ) -> TimerOutcome:
        """Run a focus session with caller-friendly defaults."""
        return self.run(
            TimerSession(
                kind=SessionKind.FOCUS,
                minutes=minutes,
                label=label,
                task_id=task_id,
            )
        )

    def run_break(
        self,
        *,
        minutes: int = 5,
        label: str = "Break",
    ) -> TimerOutcome:
        """Run a break session with caller-friendly defaults."""
        return self.run(
            TimerSession(
                kind=SessionKind.BREAK,
                minutes=minutes,
                label=label,
            )
        )


def default_timer_service() -> TimerService:
    """Build the production timer service."""
    return TimerService(
        clock=SystemClock(),
        progress=RichTimerProgress(),
        encouragement=ConsoleEncouragement(),
    )


def _session_from_config(config: TimerConfig) -> TimerSession:
    """Convert compatibility config into a timer session."""
    return TimerSession(
        kind=SessionKind.BREAK if config.is_break else SessionKind.FOCUS,
        minutes=config.minutes,
        label=config.label,
        task_id=config.task_id,
    )


def run_timer(config: TimerConfig) -> bool:
    """Run a countdown timer. Returns True if completed, False if interrupted."""
    return default_timer_service().run(_session_from_config(config)).completed


def run_focus(minutes: int = 15, task_id: int | None = None) -> bool:
    """Run a focus session timer."""
    outcome = default_timer_service().run_focus(minutes=minutes, task_id=task_id)
    return outcome.completed


def run_break(minutes: int = 5) -> bool:
    """Run a break timer."""
    outcome = default_timer_service().run_break(minutes=minutes)
    return outcome.completed


__all__ = [
    "ClockPort",
    "ConsoleEncouragement",
    "EncouragementPort",
    "RichTimerProgress",
    "SessionKind",
    "SystemClock",
    "TimerOutcome",
    "TimerProgressPort",
    "TimerService",
    "TimerSession",
    "default_timer_service",
    "run_break",
    "run_focus",
    "run_timer",
]
