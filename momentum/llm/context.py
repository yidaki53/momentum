"""Builds user context from the database for LLM prompt injection."""

from __future__ import annotations

import sqlite3

from momentum import db
from momentum.assessments import (
    bisbas_effective_max_score,
    bisbas_normalized_total_score,
    interpret_bdefs,
    interpret_stroop,
)
from momentum.models import AssessmentType, TaskStatus


def build_user_context(conn: sqlite3.Connection) -> str:
    """Build a compact, local-only view of the user's Momentum history.

    The context is deliberately short enough for a tiny on-device model, but
    includes recent completed work, activity, repeated assessment scores, and
    the current task profile.  This lets optional snippets talk about trends
    without sending the full database or a long transcript to the model.
    """
    parts: list[str] = []

    active_tasks = db.list_tasks(conn, status=TaskStatus.ACTIVE)
    pending_tasks = db.list_tasks(conn, status=TaskStatus.PENDING)
    completed_tasks = db.list_tasks(conn, status=TaskStatus.DONE)
    status = db.get_status(conn)
    done_today = status.today.tasks_completed

    if active_tasks:
        parts.append(f"Active tasks: {', '.join(t.title for t in active_tasks[:5])}")
    if pending_tasks:
        parts.append(
            f"Pending tasks ({len(pending_tasks)}): "
            f"{', '.join(t.title for t in pending_tasks[:5])}"
        )
    if completed_tasks:
        recent_done = ", ".join(t.title for t in completed_tasks[-5:])
        parts.append(f"Recently completed tasks: {recent_done}")
    parts.append(
        f"Activity: {done_today} task(s) completed today; "
        f"{status.today.focus_minutes} focus minute(s) today; "
        f"{status.week_tasks_completed} task(s) and "
        f"{status.week_focus_minutes} focus minute(s) this week; "
        f"streak {status.streak_days} day(s)"
    )

    daily_logs = db.list_all_daily_logs(conn)[:7]
    if daily_logs:
        activity = ", ".join(
            f"{entry.date.isoformat()}: {entry.tasks_completed} tasks/"
            f"{entry.focus_minutes}m"
            for entry in reversed(daily_logs)
        )
        parts.append(f"Recent daily activity: {activity}")

    sessions = db.list_focus_sessions(conn, limit=5)
    if sessions:
        parts.append(
            "Recent focus sessions: "
            + ", ".join(
                f"{s.completed_at.strftime('%Y-%m-%d')} ({s.duration_minutes}m)"
                for s in sessions
            )
        )

    def _history_line(label: str, results: list[object]) -> None:
        if not results:
            return
        values = []
        for result in results[:4]:
            taken = getattr(result, "taken_at", None)
            when = taken.strftime("%Y-%m-%d") if taken is not None else "unknown date"
            values.append(f"{when} {result.score}/{result.max_score}")
        latest = results[0]
        line = f"{label} history: {'; '.join(values)}"
        if len(results) > 1:
            delta = latest.score - results[1].score
            direction = "up" if delta > 0 else "down" if delta < 0 else "unchanged"
            line += f" (latest change {direction} by {abs(delta)} points)"
        parts.append(line)

    bdefs = db.list_assessments(conn, AssessmentType.BDEFS, limit=4)
    bisbas = db.list_assessments(conn, AssessmentType.BISBAS, limit=4)
    stroop = db.list_assessments(conn, AssessmentType.STROOP, limit=4)
    if bdefs:
        latest = bdefs[0]
        parts.append(
            f"Executive function profile: {interpret_bdefs(latest.score, latest.max_score)}"
        )
        for domain, score in latest.domain_scores.items():
            parts.append(f"  {domain}: {score}")
    if bisbas:
        latest = bisbas[0]
        parts.append(
            f"Motivation profile (BIS/BAS): "
            f"{bisbas_normalized_total_score(latest.score)}/"
            f"{bisbas_effective_max_score()}"
        )
    if stroop:
        latest = stroop[0]
        avg_ms = latest.domain_scores.get("avg_time_ms", 0)
        parts.append(
            f"Cognitive processing (Stroop): "
            f"{interpret_stroop(latest.score, latest.max_score, avg_ms)}"
        )
    _history_line("BDEFS", bdefs)
    _history_line("BIS/BAS", bisbas)
    _history_line("Stroop", stroop)

    act_entries = db.list_act_journal_entries(conn, limit=3)
    if act_entries:
        parts.append("Recent ACT check-ins:")
        for entry in act_entries:
            parts.append(f"  Values: {entry.values_focus[:60]}")
            parts.append(f"    Committed action: {entry.committed_action[:60]}")

    return "\n".join(parts)


def build_chat_history(
    conn: sqlite3.Connection, limit: int = 6
) -> list[dict[str, str]]:
    """Return recent chat messages as a list of {role, content} dicts."""
    messages = db.list_llm_chat_messages(conn, limit=limit)
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]
