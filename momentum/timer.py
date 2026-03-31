"""Backward compatibility shim for timer module."""

from momentum.domain.timer import (
    ClockPort,
    ConsoleEncouragement,
    EncouragementPort,
    RichTimerProgress,
    SessionKind,
    SystemClock,
    TimerOutcome,
    TimerProgressPort,
    TimerService,
    TimerSession,
    default_timer_service,
)
from momentum.models import TimerConfig


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
