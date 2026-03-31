"""Momentum CLI -- a gentle tool for executive dysfunction support."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from momentum import db, encouragement
from momentum.domain import timer
from momentum.models import AssessmentType, TaskStatus
from momentum.services import (
    AssessmentService,
    PersonalisationService,
    SessionService,
    TaskServiceFactory,
    status_service,
    task_service,
)
from momentum.ui import display

if TYPE_CHECKING:
    from momentum.assessments import PersonalisationProfile

app = typer.Typer(
    name="momentum",
    help="A gentle tool to help you get things done, one small step at a time.",
    no_args_is_help=True,
)


def _conn() -> sqlite3.Connection:
    """Get a database connection (convenience wrapper)."""
    return db.get_connection()


def _personalisation_profile(
    conn: sqlite3.Connection,
) -> PersonalisationProfile:
    """Return behavior defaults from the latest BIS/BAS result."""
    return PersonalisationService(conn).profile()


def _timer_service() -> timer.TimerService:
    """Build the production timer service for CLI commands."""
    return timer.default_timer_service()


def _task_service(conn: sqlite3.Connection) -> TaskServiceFactory:
    """Build task workflow operations for command handlers."""
    return task_service(conn)


def _assessment_service(conn: sqlite3.Connection) -> AssessmentService:
    """Build assessment workflow service for command handlers."""
    return AssessmentService(conn)


def _run_gui() -> None:
    """Import and run GUI lazily to keep CLI startup and tests lightweight."""
    from momentum.gui import run_gui

    run_gui()


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------


@app.command()
def add(title: str = typer.Argument(..., help="What do you need to do?")) -> None:
    """Add a new task."""
    conn = _conn()
    task = _task_service(conn).add_task(title)
    display.print_success(f"Added task #{task.id}: {task.title}")
    conn.close()


@app.command(name="break-down")
def break_down(
    task_id: int = typer.Argument(..., help="ID of the task to break down"),
) -> None:
    """Break a task into smaller sub-steps (interactive)."""
    conn = _conn()
    tasks = _task_service(conn)
    parent = tasks.get_task(task_id)
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
        created = tasks.add_subtask(parent.id, step.strip())
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
    from momentum.assessments import personalised_nudge

    conn = _conn()
    task = _task_service(conn).complete_task(task_id)
    if task is None:
        display.print_warning(f"Task #{task_id} not found.")
        conn.close()
        raise typer.Exit(1)
    display.print_success(f"Completed: {task.title}")
    profile = _personalisation_profile(conn)
    display.print_nudge(personalised_nudge(encouragement.get_nudge(), profile))
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
    minutes: Optional[int] = typer.Option(
        None,
        "--minutes",
        "-m",
        help="Focus duration in minutes (defaults to personalized value)",
    ),
    task_id: Optional[int] = typer.Option(
        None, "--task", "-t", help="Task ID to focus on"
    ),
) -> None:
    """Start a focus timer."""
    from momentum.assessments import personalised_nudge

    conn = _conn()
    profile = _personalisation_profile(conn)
    sessions = SessionService(conn, _timer_service())
    tasks = _task_service(conn)
    focus_minutes = minutes if minutes is not None else profile.focus_minutes
    if task_id is not None:
        task = tasks.get_task(task_id)
        if task is None:
            display.print_warning(f"Task #{task_id} not found.")
            conn.close()
            raise typer.Exit(1)
        tasks.activate_task(task_id)
        display.print_info(f'Focusing on: "{task.title}" for {focus_minutes} min')
    else:
        display.print_info(f"Starting {focus_minutes}-minute focus session.")

    completed = sessions.run_focus(minutes=focus_minutes, task_id=task_id)
    if completed:
        display.print_success("Focus session logged.")
        display.print_nudge(personalised_nudge(encouragement.get_nudge(), profile))

        should_break = typer.confirm(
            f"Take a {profile.break_minutes}-minute break?",
            default=True,
        )
        if should_break:
            sessions.run_break(minutes=profile.break_minutes)
    conn.close()


@app.command(name="take-break")
def take_break(
    minutes: int = typer.Option(5, "--minutes", "-m", help="Break duration in minutes"),
) -> None:
    """Take a break. You have earned it."""
    display.print_info(f"Break time: {minutes} minutes.")
    conn = _conn()
    SessionService(conn, _timer_service()).run_break(minutes=minutes)
    conn.close()


# ---------------------------------------------------------------------------
# Status & encouragement
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """See how your day is going."""
    conn = _conn()
    summary = status_service(conn)()
    display.print_status(summary)
    conn.close()


@app.command()
def nudge() -> None:
    """Get a gentle encouragement message."""
    from momentum.assessments import personalised_nudge

    conn = _conn()
    profile = _personalisation_profile(conn)
    display.print_nudge(personalised_nudge(encouragement.get_nudge(), profile))
    conn.close()


@app.command()
def start() -> None:
    """A gentle way to begin. Suggests a small step to get going."""
    from momentum.assessments import personalised_nudge

    conn = _conn()
    profile = _personalisation_profile(conn)
    sessions = SessionService(conn, _timer_service())
    tasks = _task_service(conn)
    focus_minutes = profile.focus_minutes

    # If there are active tasks, suggest continuing one
    active = tasks.first_active_task()
    if active is not None:
        task = active
        display.print_info(f'You were working on: "{task.title}"')
        cont = typer.confirm("Continue with this?", default=True)
        if cont:
            display.print_info(f"Starting a {focus_minutes}-minute focus session.")
            sessions.run_focus(minutes=focus_minutes, task_id=task.id)
            conn.close()
            return

    # If there are pending tasks, suggest the first one
    pending = tasks.first_pending_task()
    if pending is not None:
        task = pending
        display.print_info(f'How about starting with: "{task.title}"?')
        go = typer.confirm("Work on this?", default=True)
        if go:
            tasks.activate_task(task.id)
            if profile.suggest_breakdown:
                display.print_info(
                    "If this feels overwhelming, use break-down to create tiny steps first."
                )
            display.print_info(f"Starting a {focus_minutes}-minute focus session.")
            sessions.run_focus(minutes=focus_minutes, task_id=task.id)
            conn.close()
            return

    # No tasks -- ask them to name one small thing
    display.print_nudge(
        personalised_nudge("What is one small thing you could do right now?", profile)
    )
    thing = typer.prompt("Just one small thing", default="", show_default=False)
    if thing.strip():
        task = tasks.add_task(thing.strip())
        tasks.activate_task(task.id)
        display.print_success(f"Added and started: {task.title}")
        go = typer.confirm(f"Focus on it for {focus_minutes} minutes?", default=True)
        if go:
            sessions.run_focus(minutes=focus_minutes, task_id=task.id)
    else:
        display.print_nudge(personalised_nudge(encouragement.get_nudge(), profile))

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
    bisbas: bool = typer.Option(
        False, "--bisbas", help="Take the BIS/BAS motivational profile test"
    ),
) -> None:
    """Take a self-assessment test (BDEFS, Stroop, or BIS/BAS)."""
    import time as _time

    from momentum.assessments import (
        BDEFS_QUESTIONS,
        BDEFS_SCALE_LABELS,
        BISBAS_QUESTIONS,
        BISBAS_SCALE_LABELS,
        StroopResult,
        bisbas_domain_advice,
        generate_stroop_trials,
        interpret_bdefs,
        interpret_bisbas,
        interpret_stroop,
        personalise_from_bisbas,
        score_bdefs,
        score_bisbas,
        score_stroop,
    )

    conn = _conn()
    assessments = _assessment_service(conn)

    if stroop and bisbas:
        display.print_warning("Choose only one mode: --stroop or --bisbas.")
        conn.close()
        raise typer.Exit(1)

    if bisbas:
        from momentum.assessments import BISBAS_INSTRUCTIONS

        display.print_info("BIS/BAS Motivational Profile")
        display.console.print()
        for paragraph in BISBAS_INSTRUCTIONS.split("\n\n"):
            display.print_info(paragraph)
        display.console.print()
        display.print_info("Rate each statement:")
        for label in BISBAS_SCALE_LABELS:
            display.print_info(f"  {label}")
        display.console.print()

        bisbas_answers: dict[str, list[int]] = {}
        for domain, questions in BISBAS_QUESTIONS.items():
            display.console.print(f"\n[bold]{domain}[/bold]")
            bisbas_domain_answers: list[int] = []
            for q in questions:
                while True:
                    raw = typer.prompt(f"  {q} (1-4)")
                    try:
                        val = int(raw)
                        if 1 <= val <= 4:
                            bisbas_domain_answers.append(val)
                            break
                    except ValueError:
                        pass
                    display.print_warning("  Please enter 1, 2, 3, or 4.")
            bisbas_answers[domain] = bisbas_domain_answers

        create_model = score_bisbas(bisbas_answers)
        saved = assessments.save_result(create_model)
        display.print_info(f"\nTotal score: {saved.score}/{saved.max_score}")
        for domain, score in saved.domain_scores.items():
            n_qs = len(BISBAS_QUESTIONS.get(domain, []))
            max_domain = n_qs * 4 if n_qs else 1
            display.print_info(f"  {domain}: {score}/{max_domain}")
            advice = bisbas_domain_advice(domain, score, max_domain)
            if advice:
                display.console.print(f"    [dim italic]{advice}[/dim italic]")

        profile = personalise_from_bisbas(saved.domain_scores)
        display.print_info(
            "  Personalized defaults: "
            f"focus={profile.focus_minutes}m, break={profile.break_minutes}m"
        )
        display.print_nudge(
            interpret_bisbas(saved.score, saved.max_score, saved.domain_scores)
        )
    elif stroop:
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
        assessments.save_result(create_model)
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

        bdefs_answers: dict[str, list[int]] = {}
        for domain, questions in BDEFS_QUESTIONS.items():
            display.console.print(f"\n[bold]{domain}[/bold]")
            bdefs_domain_answers: list[int] = []
            for q in questions:
                while True:
                    raw = typer.prompt(f"  {q} (1-4)")
                    try:
                        val = int(raw)
                        if 1 <= val <= 4:
                            bdefs_domain_answers.append(val)
                            break
                    except ValueError:
                        pass
                    display.print_warning("  Please enter 1, 2, 3, or 4.")
            bdefs_answers[domain] = bdefs_domain_answers

        from momentum.assessments import domain_advice

        create_model = score_bdefs(bdefs_answers)
        saved = assessments.save_result(create_model)
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
        None, "--type", "-t", help="Filter by type: bdefs, stroop, or bisbas"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results to show"),
) -> None:
    """View past self-assessment results."""
    conn = _conn()
    assessments = _assessment_service(conn)
    atype = None
    if test_type:
        try:
            atype = AssessmentType(test_type.lower())
        except ValueError:
            display.print_warning(
                f"Unknown type '{test_type}'. Use 'bdefs', 'stroop', or 'bisbas'."
            )
            conn.close()
            raise typer.Exit(1)

    entries = assessments.history_entries(assessment_type=atype, limit=limit)
    if not entries:
        display.print_info("No assessment results found.")
        conn.close()
        return

    for entry in entries:
        display.console.print()
        display.console.print(entry.header)
        for line in entry.lines:
            display.console.print(f"[blue]{line}[/blue]")

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
    assessments = _assessment_service(conn)
    count = assessments.count_results()
    if count == 0:
        display.print_info("No assessment results to delete.")
        conn.close()
        return
    confirm = typer.confirm(
        f"Delete all {count} assessment result{'s' if count != 1 else ''}?",
        default=False,
    )
    if confirm:
        deleted = assessments.delete_all_results()
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
        for row in _assessment_service(conn).browse_rows(limit=limit):
            display.console.print(row)
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
    _run_gui()


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
