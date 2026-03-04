"""Momentum CLI -- a gentle tool for executive dysfunction support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from momentum import db, display, encouragement, timer
from momentum.models import AssessmentType, FocusSessionCreate, TaskCreate, TaskStatus

app = typer.Typer(
    name="momentum",
    help="A gentle tool to help you get things done, one small step at a time.",
    no_args_is_help=True,
)


def _conn() -> db.sqlite3.Connection:
    """Get a database connection (convenience wrapper)."""
    return db.get_connection()


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------


@app.command()
def add(title: str = typer.Argument(..., help="What do you need to do?")) -> None:
    """Add a new task."""
    conn = _conn()
    task_in = TaskCreate(title=title)
    task = db.add_task(conn, task_in)
    display.print_success(f"Added task #{task.id}: {task.title}")
    conn.close()


@app.command(name="break-down")
def break_down(
    task_id: int = typer.Argument(..., help="ID of the task to break down"),
) -> None:
    """Break a task into smaller sub-steps (interactive)."""
    conn = _conn()
    parent = db.get_task(conn, task_id)
    if parent is None:
        display.print_warning(f"Task #{task_id} not found.")
        conn.close()
        raise typer.Exit(1)

    display.print_info(f'Breaking down: "{parent.title}"')
    display.print_info("Enter sub-steps one per line. Empty line to finish.")

    count = 0
    while True:
        step = typer.prompt("  Sub-step", default="", show_default=False)
        if not step.strip():
            break
        sub = TaskCreate(title=step.strip(), parent_id=parent.id)
        created = db.add_task(conn, sub)
        display.print_success(f"  Added #{created.id}: {created.title}")
        count += 1

    if count == 0:
        display.print_info("No sub-steps added.")
    else:
        display.print_success(f"Created {count} sub-step{'s' if count != 1 else ''}.")
    conn.close()


@app.command()
def done(
    task_id: int = typer.Argument(..., help="ID of the task to mark complete"),
) -> None:
    """Mark a task as done."""
    conn = _conn()
    task = db.complete_task(conn, task_id)
    if task is None:
        display.print_warning(f"Task #{task_id} not found.")
        conn.close()
        raise typer.Exit(1)
    display.print_success(f"Completed: {task.title}")
    display.print_nudge(encouragement.get_nudge())
    conn.close()


@app.command(name="list")
def list_tasks(
    all_tasks: bool = typer.Option(
        False, "--all", "-a", help="Include completed tasks"
    ),
) -> None:
    """List your tasks."""
    conn = _conn()
    if all_tasks:
        tasks = db.list_tasks(conn)
    else:
        pending = db.list_tasks(conn, status=TaskStatus.PENDING)
        active = db.list_tasks(conn, status=TaskStatus.ACTIVE)
        tasks = active + pending

    # Group by parent
    top_level = [t for t in tasks if not t.is_subtask]
    display.print_task_list(top_level, title="Tasks")

    # Show subtasks under their parents
    parents_shown: set[int] = set()
    for task in tasks:
        if task.is_subtask and task.parent_id not in parents_shown:
            parent = db.get_task(conn, task.parent_id)  # type: ignore[arg-type]
            if parent:
                subs = [t for t in tasks if t.parent_id == parent.id]
                if subs:
                    display.print_task_list(
                        subs, title=f"  Sub-steps of #{parent.id}: {parent.title}"
                    )
                    parents_shown.add(parent.id)
    conn.close()


# ---------------------------------------------------------------------------
# Focus & breaks
# ---------------------------------------------------------------------------


@app.command()
def focus(
    minutes: int = typer.Option(
        15, "--minutes", "-m", help="Focus duration in minutes"
    ),
    task_id: Optional[int] = typer.Option(
        None, "--task", "-t", help="Task ID to focus on"
    ),
) -> None:
    """Start a focus timer."""
    conn = _conn()
    if task_id is not None:
        task = db.get_task(conn, task_id)
        if task is None:
            display.print_warning(f"Task #{task_id} not found.")
            conn.close()
            raise typer.Exit(1)
        db.set_task_active(conn, task_id)
        display.print_info(f'Focusing on: "{task.title}" for {minutes} min')
    else:
        display.print_info(f"Starting {minutes}-minute focus session.")

    completed = timer.run_focus(minutes=minutes, task_id=task_id)
    if completed:
        session_in = FocusSessionCreate(task_id=task_id, duration_minutes=minutes)
        db.log_focus_session(conn, session_in)
        display.print_success("Focus session logged.")

        should_break = typer.confirm("Take a 5-minute break?", default=True)
        if should_break:
            timer.run_break()
    conn.close()


@app.command(name="take-break")
def take_break(
    minutes: int = typer.Option(5, "--minutes", "-m", help="Break duration in minutes"),
) -> None:
    """Take a break. You have earned it."""
    display.print_info(f"Break time: {minutes} minutes.")
    timer.run_break(minutes=minutes)


# ---------------------------------------------------------------------------
# Status & encouragement
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """See how your day is going."""
    conn = _conn()
    summary = db.get_status(conn)
    display.print_status(summary)
    conn.close()


@app.command()
def nudge() -> None:
    """Get a gentle encouragement message."""
    display.print_nudge(encouragement.get_nudge())


@app.command()
def start() -> None:
    """A gentle way to begin. Suggests a small step to get going."""
    conn = _conn()
    summary = db.get_status(conn)

    # If there are active tasks, suggest continuing one
    if summary.active_tasks:
        task = summary.active_tasks[0]
        display.print_info(f'You were working on: "{task.title}"')
        cont = typer.confirm("Continue with this?", default=True)
        if cont:
            display.print_info("Starting a 15-minute focus session.")
            completed = timer.run_focus(minutes=15, task_id=task.id)
            if completed:
                session_in = FocusSessionCreate(task_id=task.id, duration_minutes=15)
                db.log_focus_session(conn, session_in)
            conn.close()
            return

    # If there are pending tasks, suggest the first one
    if summary.pending_tasks:
        task = summary.pending_tasks[0]
        display.print_info(f'How about starting with: "{task.title}"?')
        go = typer.confirm("Work on this?", default=True)
        if go:
            db.set_task_active(conn, task.id)
            display.print_info("Starting a 15-minute focus session.")
            completed = timer.run_focus(minutes=15, task_id=task.id)
            if completed:
                session_in = FocusSessionCreate(task_id=task.id, duration_minutes=15)
                db.log_focus_session(conn, session_in)
            conn.close()
            return

    # No tasks -- ask them to name one small thing
    display.print_nudge("What is one small thing you could do right now?")
    thing = typer.prompt("Just one small thing", default="", show_default=False)
    if thing.strip():
        task_in = TaskCreate(title=thing.strip())
        task = db.add_task(conn, task_in)
        db.set_task_active(conn, task.id)
        display.print_success(f"Added and started: {task.title}")
        go = typer.confirm("Focus on it for 15 minutes?", default=True)
        if go:
            completed = timer.run_focus(minutes=15, task_id=task.id)
            if completed:
                session_in = FocusSessionCreate(task_id=task.id, duration_minutes=15)
                db.log_focus_session(conn, session_in)
    else:
        display.print_nudge(encouragement.get_nudge())

    conn.close()


# ---------------------------------------------------------------------------
# Configuration & sync
# ---------------------------------------------------------------------------


@app.command()
def config(
    sync: Optional[str] = typer.Option(
        None,
        "--sync",
        help="Sync DB via cloud folder: onedrive, dropbox, google-drive",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Set a custom database file path",
    ),
    reset: bool = typer.Option(False, "--reset", help="Reset to default local DB"),
    show: bool = typer.Option(False, "--show", help="Show current config"),
) -> None:
    """Configure where your data is stored (for cloud sync)."""
    from momentum import config as cfg

    if sync:
        result = cfg.set_cloud_sync(sync)
        if result is None:
            display.print_warning(
                f"Could not find {sync} folder. "
                f"Use --db-path to set a custom path instead."
            )
            raise typer.Exit(1)
        display.print_success(f"Database will sync via {sync}: {result.db_path}")
    elif db_path:
        result = cfg.set_db_path(db_path)
        display.print_success(f"Database path set to: {result.db_path}")
    elif reset:
        cfg.reset_db_path()
        display.print_success("Reset to default local database.")
    elif show:
        current = cfg.load_config()
        resolved = cfg.get_db_path()
        if current.db_path:
            display.print_info(f"Database: {current.db_path}")
        else:
            display.print_info(f"Database: {resolved} (default)")
    else:
        display.print_info("Use --sync, --db-path, --reset, or --show.")


# ---------------------------------------------------------------------------
# Self-assessment tests
# ---------------------------------------------------------------------------


@app.command(name="test")
def run_test(
    stroop: bool = typer.Option(
        False, "--stroop", help="Take the Stroop colour-word test instead"
    ),
) -> None:
    """Take a self-assessment test (BDEFS or Stroop)."""
    import time as _time

    from momentum.assessments import (
        BDEFS_QUESTIONS,
        BDEFS_SCALE_LABELS,
        StroopResult,
        generate_stroop_trials,
        interpret_bdefs,
        interpret_stroop,
        score_bdefs,
        score_stroop,
    )

    conn = _conn()

    if stroop:
        # --- Stroop test ---
        from momentum.assessments import STROOP_INSTRUCTIONS

        display.print_info("Stroop Colour-Word Test")
        display.console.print()
        for paragraph in STROOP_INSTRUCTIONS.split("\n\n"):
            display.print_info(paragraph)
        display.console.print()
        typer.confirm("Ready?", default=True, abort=True)

        trials = generate_stroop_trials()
        correct = 0
        total_time = 0.0
        per_trial: list[tuple[bool, float]] = []

        for i, trial in enumerate(trials, 1):
            display.console.print(
                f"\n  [{trial.ink_colour}]{trial.word.upper()}[/{trial.ink_colour}]"
            )
            t0 = _time.monotonic()
            answer = typer.prompt(f"  ({i}/{len(trials)}) Colour")
            elapsed = _time.monotonic() - t0
            total_time += elapsed
            is_correct = answer.strip().lower() == trial.ink_colour
            per_trial.append((is_correct, elapsed))
            if is_correct:
                correct += 1
                display.print_success("  Correct!")
            else:
                display.print_warning(f"  The colour was {trial.ink_colour}.")

        result = StroopResult(
            trials=len(trials),
            correct=correct,
            total_time_s=total_time,
            per_trial=per_trial,
        )
        create_model = score_stroop(result)
        saved = db.save_assessment(conn, create_model)
        display.print_info(
            f"\nResult: {correct}/{len(trials)} correct, "
            f"avg {result.avg_time_s:.1f}s per trial"
        )
        display.print_nudge(
            interpret_stroop(correct, len(trials), int(result.avg_time_s * 1000))
        )
    else:
        # --- BDEFS self-report ---
        from momentum.assessments import BDEFS_INSTRUCTIONS

        display.print_info("BDEFS-style Executive Function Self-Assessment")
        display.console.print()
        for paragraph in BDEFS_INSTRUCTIONS.split("\n\n"):
            display.print_info(paragraph)
        display.console.print()
        display.print_info("Rate each statement:")
        for label in BDEFS_SCALE_LABELS:
            display.print_info(f"  {label}")
        display.console.print()

        answers: dict[str, list[int]] = {}
        for domain, questions in BDEFS_QUESTIONS.items():
            display.console.print(f"\n[bold]{domain}[/bold]")
            domain_answers: list[int] = []
            for q in questions:
                while True:
                    raw = typer.prompt(f"  {q} (1-4)")
                    try:
                        val = int(raw)
                        if 1 <= val <= 4:
                            domain_answers.append(val)
                            break
                    except ValueError:
                        pass
                    display.print_warning("  Please enter 1, 2, 3, or 4.")
            answers[domain] = domain_answers

        from momentum.assessments import domain_advice

        create_model = score_bdefs(answers)
        saved = db.save_assessment(conn, create_model)
        display.print_info(f"\nTotal score: {saved.score}/{saved.max_score}")
        for d, s in saved.domain_scores.items():
            n_qs = len(BDEFS_QUESTIONS[d])
            display.print_info(f"  {d}: {s}/{n_qs * 4}")
            advice = domain_advice(d, s, n_qs * 4)
            if advice:
                display.console.print(f"    [dim italic]{advice}[/dim italic]")
        display.print_nudge(interpret_bdefs(saved.score, saved.max_score))

    conn.close()


@app.command(name="test-results")
def test_results(
    test_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by type: bdefs or stroop"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results to show"),
) -> None:
    """View past self-assessment results."""
    from momentum.assessments import (
        BDEFS_QUESTIONS as _BQ,
    )
    from momentum.assessments import (
        domain_advice,
        interpret_bdefs,
        interpret_stroop,
    )

    conn = _conn()
    atype = None
    if test_type:
        try:
            atype = AssessmentType(test_type.lower())
        except ValueError:
            display.print_warning(
                f"Unknown type '{test_type}'. Use 'bdefs' or 'stroop'."
            )
            conn.close()
            raise typer.Exit(1)

    results = db.list_assessments(conn, assessment_type=atype, limit=limit)
    if not results:
        display.print_info("No assessment results found.")
        conn.close()
        return

    for r in results:
        taken = r.taken_at.strftime("%Y-%m-%d %H:%M")
        display.console.print(
            f"\n[bold]#{r.id}[/bold] {r.assessment_type.value.upper()}  ({taken})"
        )
        display.print_info(f"  Score: {r.score}/{r.max_score}")
        if r.assessment_type == AssessmentType.BDEFS:
            for d, s in r.domain_scores.items():
                n_qs = len(_BQ.get(d, []))
                display.print_info(f"    {d}: {s}")
                if n_qs:
                    advice = domain_advice(d, s, n_qs * 4)
                    if advice:
                        display.console.print(
                            f"      [dim italic]{advice}[/dim italic]"
                        )
            display.print_info(f"  {interpret_bdefs(r.score, r.max_score)}")
        elif r.assessment_type == AssessmentType.STROOP:
            avg_ms = r.domain_scores.get("avg_time_ms", 0)
            display.print_info(f"  Avg response: {avg_ms}ms")
            display.print_info(f"  {interpret_stroop(r.score, r.max_score, avg_ms)}")

    conn.close()


# ---------------------------------------------------------------------------
# About & Science
# ---------------------------------------------------------------------------


@app.command()
def about() -> None:
    """Show information about Momentum."""
    display.print_info("Momentum v0.1.0")
    display.console.print()
    display.print_info(
        "A gentle tool to help people with executive dysfunction "
        "get back on track, one small step at a time."
    )
    display.console.print()
    display.print_info(
        "Created by Robin \u00d6berg -- Data Scientist, MSc Social Anthropology, "
        "MSc Applied Cultural Analysis."
    )
    display.console.print()
    display.print_info("Copyright \u00a9 2026 Robin \u00d6berg.")
    display.print_info("Licensed under the MIT License.")
    display.console.print()
    display.print_info("https://github.com/yidaki53/momentum")


@app.command()
def science() -> None:
    """Display the scientific rationale behind Momentum."""
    science_path = Path(__file__).resolve().parent.parent / "SCIENCE.md"
    if not science_path.exists():
        display.print_warning("SCIENCE.md not found.")
        raise typer.Exit(1)
    content = science_path.read_text(encoding="utf-8")
    display.console.print()
    from rich.markdown import Markdown

    display.console.print(Markdown(content))


# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------


@app.command(name="delete-results")
def delete_results() -> None:
    """Delete all self-assessment results."""
    conn = _conn()
    count = len(db.list_assessments(conn, limit=9999))
    if count == 0:
        display.print_info("No assessment results to delete.")
        conn.close()
        return
    confirm = typer.confirm(
        f"Delete all {count} assessment result{'s' if count != 1 else ''}?",
        default=False,
    )
    if confirm:
        deleted = db.delete_all_assessments(conn)
        display.print_success(
            f"Deleted {deleted} assessment result{'s' if deleted != 1 else ''}."
        )
    else:
        display.print_info("Cancelled.")
    conn.close()


@app.command(name="delete-tasks")
def delete_tasks() -> None:
    """Delete all tasks and focus session history."""
    conn = _conn()
    count = len(db.list_tasks(conn))
    if count == 0:
        display.print_info("No tasks to delete.")
        conn.close()
        return
    confirm = typer.confirm(
        f"Delete all {count} task{'s' if count != 1 else ''} and related focus sessions?",
        default=False,
    )
    if confirm:
        deleted = db.delete_all_tasks(conn)
        display.print_success(f"Deleted {deleted} task{'s' if deleted != 1 else ''}.")
    else:
        display.print_info("Cancelled.")
    conn.close()


@app.command(name="browse-db")
def browse_db(
    table: str = typer.Argument(
        ..., help="Table to browse: tasks, assessments, sessions, daily_log"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of rows"),
) -> None:
    """Browse database entries."""
    conn = _conn()
    if table == "tasks":
        tasks = db.list_tasks(conn)
        for t in tasks[:limit]:
            status = t.status.value
            display.console.print(
                f"  #{t.id}  [{status}]  {t.title}  (created: {t.created_at:%Y-%m-%d})"
            )
    elif table == "assessments":
        results = db.list_assessments(conn, limit=limit)
        for r in results:
            display.console.print(
                f"  #{r.id}  {r.assessment_type.value}  "
                f"score={r.score}/{r.max_score}  ({r.taken_at:%Y-%m-%d %H:%M})"
            )
    elif table == "sessions":
        sessions = db.list_focus_sessions(conn, limit=limit)
        for s in sessions:
            display.console.print(
                f"  #{s.id}  {s.duration_minutes}min  task={s.task_id}  "
                f"({s.completed_at:%Y-%m-%d %H:%M})"
            )
    elif table == "daily_log":
        logs = db.list_all_daily_logs(conn)
        for log in logs[:limit]:
            display.console.print(
                f"  {log.date}  tasks={log.tasks_completed}  "
                f"focus={log.focus_minutes}min"
            )
    else:
        display.print_warning(
            f"Unknown table '{table}'. Use: tasks, assessments, sessions, daily_log"
        )
    conn.close()


@app.command(name="delete-entry")
def delete_entry(
    table: str = typer.Argument(
        ..., help="Table: tasks, assessments, sessions, daily_log"
    ),
    entry_id: str = typer.Argument(..., help="ID (or date for daily_log) to delete"),
) -> None:
    """Delete a specific database entry by table and ID."""
    conn = _conn()
    deleted = False
    if table == "tasks":
        deleted = db.delete_task(conn, int(entry_id))
    elif table == "assessments":
        deleted = db.delete_assessment(conn, int(entry_id))
    elif table == "sessions":
        deleted = db.delete_focus_session(conn, int(entry_id))
    elif table == "daily_log":
        deleted = db.delete_daily_log(conn, entry_id)
    else:
        display.print_warning(
            f"Unknown table '{table}'. Use: tasks, assessments, sessions, daily_log"
        )
        conn.close()
        raise typer.Exit(1)

    if deleted:
        display.print_success(f"Deleted {table} entry {entry_id}.")
    else:
        display.print_warning(f"Entry {entry_id} not found in {table}.")
    conn.close()


# ---------------------------------------------------------------------------
# GUI & autostart (delegate to submodules)
# ---------------------------------------------------------------------------


@app.command()
def gui() -> None:
    """Open the Momentum GUI dashboard."""
    from momentum.gui import run_gui

    run_gui()


@app.command()
def autostart(
    enable: bool = typer.Option(False, "--enable", help="Enable autostart on login"),
    disable: bool = typer.Option(False, "--disable", help="Disable autostart"),
    show_status: bool = typer.Option(False, "--status", help="Show autostart status"),
) -> None:
    """Manage autostart on login."""
    from momentum.autostart import (
        disable_autostart,
        enable_autostart,
        get_autostart_status,
    )

    if enable:
        result = enable_autostart()
        if result.systemd_enabled or result.xdg_enabled:
            display.print_success("Autostart enabled.")
        else:
            display.print_warning("Could not enable autostart.")
    elif disable:
        disable_autostart()
        display.print_success("Autostart disabled.")
    elif show_status:
        result = get_autostart_status()
        display.print_info(
            f"Systemd service: {'enabled' if result.systemd_enabled else 'not found'}"
        )
        display.print_info(
            f"XDG autostart: {'enabled' if result.xdg_enabled else 'not found'}"
        )
        if result.service_path:
            display.print_info(f"  Service: {result.service_path}")
        if result.desktop_entry_path:
            display.print_info(f"  Desktop entry: {result.desktop_entry_path}")
    else:
        display.print_info("Use --enable, --disable, or --status.")


if __name__ == "__main__":
    app()
