"""SQLite database layer. All public functions return Pydantic models."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from momentum.config import get_db_path as _config_get_db_path
from momentum.models import (
    AssessmentResult,
    AssessmentResultCreate,
    AssessmentType,
    DailyLog,
    FocusSession,
    FocusSessionCreate,
    StatusSummary,
    Task,
    TaskCreate,
    TaskStatus,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    parent_id   INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
    status      TEXT    NOT NULL DEFAULT 'pending',
    created_at  TEXT    NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS focus_sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    duration_minutes INTEGER NOT NULL,
    completed_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_log (
    date             TEXT PRIMARY KEY,
    tasks_completed  INTEGER NOT NULL DEFAULT 0,
    focus_minutes    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    type            TEXT    NOT NULL,
    score           INTEGER NOT NULL,
    max_score       INTEGER NOT NULL,
    domain_scores   TEXT    NOT NULL DEFAULT '{}',
    taken_at        TEXT    NOT NULL
);
"""


def _get_db_path() -> Path:
    """Return the database file path from config (or default)."""
    return _config_get_db_path()


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a connection and ensure the schema exists."""
    path = db_path or _get_db_path()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _row_to_task(row: sqlite3.Row) -> Task:
    """Convert a database row to a Task model."""
    return Task(
        id=row["id"],
        title=row["title"],
        parent_id=row["parent_id"],
        status=TaskStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        completed_at=(
            datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
        ),
    )


def add_task(conn: sqlite3.Connection, task_in: TaskCreate) -> Task:
    """Insert a new task and return it as a model."""
    now = datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO tasks (title, parent_id, status, created_at) VALUES (?, ?, ?, ?)",
        (task_in.title, task_in.parent_id, TaskStatus.PENDING.value, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _row_to_task(row)


def get_task(conn: sqlite3.Connection, task_id: int) -> Optional[Task]:
    """Fetch a single task by ID."""
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_task(row) if row else None


def list_tasks(
    conn: sqlite3.Connection,
    status: Optional[TaskStatus] = None,
    parent_id: Optional[int] = None,
) -> list[Task]:
    """List tasks, optionally filtered by status and/or parent."""
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[str | int] = []
    if status is not None:
        query += " AND status = ?"
        params.append(status.value)
    if parent_id is not None:
        query += " AND parent_id = ?"
        params.append(parent_id)
    query += " ORDER BY created_at ASC"
    rows = conn.execute(query, params).fetchall()
    return [_row_to_task(r) for r in rows]


def complete_task(conn: sqlite3.Connection, task_id: int) -> Optional[Task]:
    """Mark a task as done and update the daily log."""
    now = datetime.now()
    conn.execute(
        "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?",
        (TaskStatus.DONE.value, now.isoformat(), task_id),
    )
    # Update daily log
    today_str = date.today().isoformat()
    conn.execute(
        """INSERT INTO daily_log (date, tasks_completed, focus_minutes)
           VALUES (?, 1, 0)
           ON CONFLICT(date) DO UPDATE SET tasks_completed = tasks_completed + 1""",
        (today_str,),
    )
    conn.commit()
    return get_task(conn, task_id)


def set_task_active(conn: sqlite3.Connection, task_id: int) -> Optional[Task]:
    """Set a task to active status."""
    conn.execute(
        "UPDATE tasks SET status = ? WHERE id = ?",
        (TaskStatus.ACTIVE.value, task_id),
    )
    conn.commit()
    return get_task(conn, task_id)


def uncomplete_task(conn: sqlite3.Connection, task_id: int) -> Optional[Task]:
    """Revert a completed task back to pending."""
    conn.execute(
        "UPDATE tasks SET status = ?, completed_at = NULL WHERE id = ?",
        (TaskStatus.PENDING.value, task_id),
    )
    conn.commit()
    return get_task(conn, task_id)


def get_subtasks(conn: sqlite3.Connection, parent_id: int) -> list[Task]:
    """Get all sub-tasks of a parent task."""
    return list_tasks(conn, parent_id=parent_id)


# ---------------------------------------------------------------------------
# Focus sessions
# ---------------------------------------------------------------------------


def _row_to_session(row: sqlite3.Row) -> FocusSession:
    """Convert a database row to a FocusSession model."""
    return FocusSession(
        id=row["id"],
        task_id=row["task_id"],
        duration_minutes=row["duration_minutes"],
        completed_at=datetime.fromisoformat(row["completed_at"]),
    )


def log_focus_session(
    conn: sqlite3.Connection, session_in: FocusSessionCreate
) -> FocusSession:
    """Record a completed focus session and update the daily log."""
    now = datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO focus_sessions (task_id, duration_minutes, completed_at) VALUES (?, ?, ?)",
        (session_in.task_id, session_in.duration_minutes, now),
    )
    # Update daily log
    today_str = date.today().isoformat()
    conn.execute(
        """INSERT INTO daily_log (date, tasks_completed, focus_minutes)
           VALUES (?, 0, ?)
           ON CONFLICT(date) DO UPDATE SET focus_minutes = focus_minutes + ?""",
        (today_str, session_in.duration_minutes, session_in.duration_minutes),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM focus_sessions WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    return _row_to_session(row)


# ---------------------------------------------------------------------------
# Daily log & status
# ---------------------------------------------------------------------------


def get_daily_log(conn: sqlite3.Connection, for_date: date) -> DailyLog:
    """Get the daily log for a specific date, returning zeros if none exists."""
    row = conn.execute(
        "SELECT * FROM daily_log WHERE date = ?", (for_date.isoformat(),)
    ).fetchone()
    if row:
        return DailyLog(
            date=date.fromisoformat(row["date"]),
            tasks_completed=row["tasks_completed"],
            focus_minutes=row["focus_minutes"],
        )
    return DailyLog(date=for_date, tasks_completed=0, focus_minutes=0)


def _calculate_streak(conn: sqlite3.Connection) -> int:
    """Count consecutive days (ending today or yesterday) with at least one task completed."""
    rows = conn.execute(
        "SELECT date FROM daily_log WHERE tasks_completed > 0 ORDER BY date DESC"
    ).fetchall()
    if not rows:
        return 0

    streak = 0
    check_date = date.today()
    dates = {date.fromisoformat(r["date"]) for r in rows}

    # Allow streak to start from today or yesterday
    if check_date not in dates:
        check_date -= timedelta(days=1)
        if check_date not in dates:
            return 0

    while check_date in dates:
        streak += 1
        check_date -= timedelta(days=1)

    return streak


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------


def save_assessment(
    conn: sqlite3.Connection, result_in: AssessmentResultCreate
) -> AssessmentResult:
    """Save a completed assessment and return the stored result."""
    import json as _json

    now = datetime.now().isoformat()
    domain_json = _json.dumps(result_in.domain_scores)
    cur = conn.execute(
        "INSERT INTO assessments (type, score, max_score, domain_scores, taken_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            result_in.assessment_type.value,
            result_in.score,
            result_in.max_score,
            domain_json,
            now,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM assessments WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    return _row_to_assessment(row)


def list_assessments(
    conn: sqlite3.Connection,
    assessment_type: Optional[AssessmentType] = None,
    limit: int = 20,
) -> list[AssessmentResult]:
    """List past assessment results, most recent first."""
    query = "SELECT * FROM assessments"
    params: list[str | int] = []
    if assessment_type is not None:
        query += " WHERE type = ?"
        params.append(assessment_type.value)
    query += " ORDER BY taken_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [_row_to_assessment(r) for r in rows]


def _row_to_assessment(row: sqlite3.Row) -> AssessmentResult:
    """Convert a database row to an AssessmentResult model."""
    import json as _json

    return AssessmentResult(
        id=row["id"],
        assessment_type=AssessmentType(row["type"]),
        score=row["score"],
        max_score=row["max_score"],
        domain_scores=_json.loads(row["domain_scores"]),
        taken_at=datetime.fromisoformat(row["taken_at"]),
    )


# ---------------------------------------------------------------------------
# Daily log & status
# ---------------------------------------------------------------------------


def get_status(conn: sqlite3.Connection) -> StatusSummary:
    """Build the full status summary."""
    today_log = get_daily_log(conn, date.today())

    # Week totals
    week_start = date.today() - timedelta(days=date.today().weekday())
    rows = conn.execute(
        "SELECT COALESCE(SUM(tasks_completed), 0) as tc, COALESCE(SUM(focus_minutes), 0) as fm "
        "FROM daily_log WHERE date >= ?",
        (week_start.isoformat(),),
    ).fetchone()
    week_tasks = rows["tc"] if rows else 0
    week_focus = rows["fm"] if rows else 0

    pending = list_tasks(conn, status=TaskStatus.PENDING)
    active = list_tasks(conn, status=TaskStatus.ACTIVE)
    streak = _calculate_streak(conn)

    return StatusSummary(
        today=today_log,
        week_tasks_completed=week_tasks,
        week_focus_minutes=week_focus,
        streak_days=streak,
        pending_tasks=pending,
        active_tasks=active,
    )
