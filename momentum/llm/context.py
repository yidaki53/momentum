"""Builds user context from the database for LLM prompt injection."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Optional

from momentum import db
from momentum.assessments import (
    bisbas_effective_domain_max_score,
    bisbas_effective_max_score,
    bisbas_normalized_domain_score,
    bisbas_normalized_total_score,
    interpret_bdefs,
    interpret_bisbas,
    interpret_stroop,
    profile_from_latest_assessments,
)
from momentum.models import AssessmentType, TaskStatus


def build_user_context(conn: sqlite3.Connection) -> str:
    """Build a concise user context string from database state."""
    parts: list[str] = []

    # --- Tasks ---
    active_tasks = db.list_tasks(conn, status=TaskStatus.ACTIVE)
    pending_tasks = db.list_tasks(conn, status=TaskStatus.PENDING)
    done_today = db.get_daily_log(conn, date.today()).tasks_completed

    if active_tasks:
        parts.append(
            f"Active tasks: {', '.join(t.title for t in active_tasks[:5])}"
        )
    if pending_tasks:
        parts.append(
            f"Pending tasks ({len(pending_tasks)}): "
            f"{', '.join(t.title for t in pending_tasks[:5])}"
        )
    parts.append(f"Tasks completed today: {done_today}")

    # --- Streak ---
    status = db.get_status(conn)
    parts.append(f"Current streak: {status.streak_days} day(s)")
    parts.append(f"Focus minutes today: {status.today.focus_minutes}")

    # --- Assessments ---
    latest_bdefs = db.list_assessments(conn, AssessmentType.BDEFS, limit=1)
    latest_bisbas = db.list_assessments(conn, AssessmentType.BISBAS, limit=1)
    latest_stroop = db.list_assessments(conn, AssessmentType.STROOP, limit=1)

    if latest_bdefs:
        r = latest_bdefs[0]
        parts.append(
            f"Executive function profile: {interpret_bdefs(r.score, r.max_score)}"
        )
        for d, s in r.domain_scores.items():
            parts.append(f"  {d}: {s}")

    if latest_bisbas:
        r = latest_bisbas[0]
        parts.append(
            f"Motivation profile (BIS/BAS): "
            f"{bisbas_normalized_total_score(r.score)}/{bisbas_effective_max_score()}"
        )

    if latest_stroop:
        r = latest_stroop[0]
        avg_ms = r.domain_scores.get("avg_time_ms", 0)
        parts.append(
            f"Cognitive processing (Stroop): {interpret_stroop(r.score, r.max_score, avg_ms)}"
        )

    # --- ACT journal ---
    act_entries = db.list_act_journal_entries(conn, limit=3)
    if act_entries:
        parts.append("Recent ACT check-ins:")
        for e in act_entries:
            parts.append(f"  - Values: {e.values_focus[:60]}")
            parts.append(f"    Committed action: {e.committed_action[:60]}")

    return "\n".join(parts)


def build_chat_history(
    conn: sqlite3.Connection, limit: int = 6
) -> list[dict[str, str]]:
    """Return recent chat messages as a list of {role, content} dicts."""
    messages = db.list_llm_chat_messages(conn, limit=limit)
    return [
        {"role": m.role, "content": m.content} for m in reversed(messages)
    ]