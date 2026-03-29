"""Tests for the timer module."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from momentum.models import TimerConfig
from momentum.timer import (
    SessionKind,
    TimerOutcome,
    TimerService,
    TimerSession,
    run_break,
    run_focus,
    run_timer,
)


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def start(self, *, label: str, total_seconds: int) -> None:
        self.events.append(("start", label, total_seconds))

    def advance(self, seconds: int = 1) -> None:
        self.events.append(("advance", seconds))

    def interrupted(self, *, elapsed_seconds: int, total_seconds: int) -> None:
        self.events.append(("interrupted", elapsed_seconds, total_seconds))

    def complete(self) -> None:
        self.events.append(("complete",))


class FakeClock:
    def __init__(self, *, interrupt_after: int | None = None) -> None:
        self.calls = 0
        self._interrupt_after = interrupt_after

    def sleep_one_second(self) -> None:
        if self._interrupt_after is not None and self.calls >= self._interrupt_after:
            raise KeyboardInterrupt()
        self.calls += 1


@dataclass
class RecordingEncouragement:
    focus_message: str = "Keep going"
    break_message: str = "Take a breath"
    delivered: list[str] | None = None

    def __post_init__(self) -> None:
        if self.delivered is None:
            self.delivered = []

    def completion_message_for(self, kind: SessionKind) -> str:
        return self.break_message if kind == SessionKind.BREAK else self.focus_message

    def deliver(self, message: str) -> None:
        assert self.delivered is not None
        self.delivered.append(message)


class TestTimerService:
    def test_focus_session_completes_at_boundary(self) -> None:
        clock = FakeClock()
        progress = RecordingProgress()
        encouragement = RecordingEncouragement()
        service = TimerService(
            clock=clock, progress=progress, encouragement=encouragement
        )

        outcome = service.run(
            TimerSession(
                kind=SessionKind.FOCUS,
                minutes=1,
                label="Focus",
                task_id=42,
            )
        )

        assert outcome == TimerOutcome(
            completed=True,
            elapsed_seconds=60,
            total_seconds=60,
            kind=SessionKind.FOCUS,
            task_id=42,
            completion_message="Keep going",
        )
        assert clock.calls == 60
        assert progress.events[0] == ("start", "Focus (task #42)", 60)
        assert progress.events[-1] == ("complete",)
        assert progress.events.count(("advance", 1)) == 60
        assert encouragement.delivered == ["Keep going"]

    def test_interrupt_returns_incomplete_outcome(self) -> None:
        clock = FakeClock(interrupt_after=3)
        progress = RecordingProgress()
        encouragement = RecordingEncouragement()
        service = TimerService(
            clock=clock, progress=progress, encouragement=encouragement
        )

        outcome = service.run_break(minutes=1)

        assert outcome == TimerOutcome(
            completed=False,
            elapsed_seconds=3,
            total_seconds=60,
            kind=SessionKind.BREAK,
            task_id=None,
            completion_message=None,
        )
        assert progress.events[0] == ("start", "Break", 60)
        assert progress.events[-1] == ("interrupted", 3, 60)
        assert encouragement.delivered == []


class TestCompatibilityWrappers:
    @patch("momentum.timer.default_timer_service")
    def test_run_timer_uses_default_service(self, mock_factory) -> None:
        service = mock_factory.return_value
        service.run.return_value = TimerOutcome(
            completed=True,
            elapsed_seconds=60,
            total_seconds=60,
            kind=SessionKind.FOCUS,
        )

        config = TimerConfig(minutes=1, label="Test")
        assert run_timer(config) is True
        service.run.assert_called_once()

    @patch("momentum.timer.default_timer_service")
    def test_run_focus(self, mock_factory) -> None:
        service = mock_factory.return_value
        service.run_focus.return_value = TimerOutcome(
            completed=True,
            elapsed_seconds=60,
            total_seconds=60,
            kind=SessionKind.FOCUS,
            task_id=7,
            completion_message="Keep going",
        )

        result = run_focus(minutes=1, task_id=7)
        assert result is True
        service.run_focus.assert_called_once_with(minutes=1, task_id=7)

    @patch("momentum.timer.default_timer_service")
    def test_run_break(self, mock_factory) -> None:
        service = mock_factory.return_value
        service.run_break.return_value = TimerOutcome(
            completed=True,
            elapsed_seconds=60,
            total_seconds=60,
            kind=SessionKind.BREAK,
            completion_message="Take a breath",
        )

        result = run_break(minutes=1)
        assert result is True
        service.run_break.assert_called_once_with(minutes=1)
