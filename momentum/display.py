"""Rich terminal formatting helpers."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from momentum.models import StatusSummary, Task, TaskStatus

console = Console()

_STATUS_STYLE: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "dim",
    TaskStatus.ACTIVE: "bold cyan",
    TaskStatus.DONE: "green",
}

_STATUS_ICON: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "[ ]",
    TaskStatus.ACTIVE: "[~]",
    TaskStatus.DONE: "[x]",
}


def print_task(task: Task, indent: int = 0) -> None:
    """Print a single task with status indicator."""
    prefix = "  " * indent
    icon = _STATUS_ICON[task.status]
    style = _STATUS_STYLE[task.status]
    console.print(f"{prefix}{icon} [bold]#{task.id}[/bold] {task.title}", style=style)


def print_task_list(tasks: list[Task], title: str = "Tasks") -> None:
    """Print a list of tasks in a panel."""
    if not tasks:
        console.print(Panel("No tasks.", title=title, border_style="dim"))
        return

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("status", width=3)
    table.add_column("id", width=5)
    table.add_column("title")

    for task in tasks:
        style = _STATUS_STYLE[task.status]
        table.add_row(
            _STATUS_ICON[task.status],
            f"#{task.id}",
            task.title,
            style=style,
        )

    console.print(Panel(table, title=title, border_style="blue"))


def print_status(summary: StatusSummary) -> None:
    """Print the full status dashboard."""
    # Today panel
    today = summary.today
    total_today = today.tasks_completed + len(summary.pending_tasks) + len(summary.active_tasks)
    progress_pct = (today.tasks_completed / total_today * 100) if total_today > 0 else 0

    lines: list[str] = [
        f"Tasks completed today: {today.tasks_completed}",
        f"Focus time today: {today.focus_minutes} min",
        f"Progress: {progress_pct:.0f}%",
        "",
        f"This week: {summary.week_tasks_completed} tasks, {summary.week_focus_minutes} min focused",
        f"Streak: {summary.streak_days} day{'s' if summary.streak_days != 1 else ''}",
    ]
    console.print(Panel("\n".join(lines), title="Status", border_style="green"))

    # Active tasks
    if summary.active_tasks:
        print_task_list(summary.active_tasks, title="Active")

    # Pending tasks
    if summary.pending_tasks:
        print_task_list(summary.pending_tasks, title="Pending")


def print_nudge(message: str) -> None:
    """Print an encouragement message in a styled panel."""
    text = Text(message, justify="center")
    console.print(Panel(text, border_style="magenta", padding=(1, 4)))


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]{message}[/green]")


def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[blue]{message}[/blue]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]{message}[/yellow]")


def create_timer_progress() -> Progress:
    """Create a Rich progress bar for the timer."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )
