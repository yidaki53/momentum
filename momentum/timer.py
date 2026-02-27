"""Focus timer and break countdown logic."""

from __future__ import annotations

import time

from momentum.display import console, create_timer_progress, print_nudge
from momentum.encouragement import get_break_message, get_nudge
from momentum.models import TimerConfig


def run_timer(config: TimerConfig) -> bool:
    """Run a countdown timer. Returns True if completed, False if interrupted."""
    total_seconds = config.minutes * 60
    label = config.label
    if config.task_id is not None:
        label = f"{config.label} (task #{config.task_id})"

    progress = create_timer_progress()

    try:
        with progress:
            task = progress.add_task(label, total=total_seconds)
            for _ in range(total_seconds):
                time.sleep(1)
                progress.advance(task, 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Timer stopped early.[/yellow]")
        return False

    # Bell notification
    console.print("\a", end="")

    if config.is_break:
        print_nudge(get_break_message())
    else:
        print_nudge(get_nudge())

    return True


def run_focus(minutes: int = 15, task_id: int | None = None) -> bool:
    """Run a focus session timer."""
    config = TimerConfig(minutes=minutes, label="Focus", task_id=task_id)
    return run_timer(config)


def run_break(minutes: int = 5) -> bool:
    """Run a break timer."""
    config = TimerConfig(minutes=minutes, label="Break", is_break=True)
    return run_timer(config)
