"""Builds user context from the database for LLM prompt injection."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from momentum import db
from momentum.assessments import (
    bisbas_effective_max_score,
    bisbas_normalized_total_score,
    interpret_bdefs,
    interpret_stroop,
)
from momentum.models import AssessmentResult, AssessmentType, TaskStatus


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

    def _history_line(label: str, results: Sequence[AssessmentResult]) -> None:
        if not results:
            return
        values = []
        for result in results[:4]:
            taken = result.taken_at
            when = taken.strftime("%Y-%m-%d")
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


# Explicit task-creation marker.
#
# The coach may ask to create a task, but a 1.1B model must not be allowed to
# write to the database on its own initiative: a hallucinated "[[task: ...]]"
# would otherwise invent work the user never agreed to. So the model is told to
# emit this marker only when the user explicitly asks, the marker is stripped
# from the visible reply, and creation still requires a confirmation tap.
TASK_MARKER = "[[task:"

_TASK_MARKER_RE = re.compile(
    r"\[\[\s*task\s*:\s*(?P<title>[^\]\|]+?)\s*(?:\|[^\]]*)?\]\]",
    re.IGNORECASE,
)


def extract_task_requests(text: str) -> list[str]:
    """Return the task titles a coach reply asked to create.

    The markers are removed from the text so the user sees ordinary prose, and
    each title is de-duplicated case-insensitively and length-capped: a runaway
    generation must not be able to flood the task list.
    """
    seen: set[str] = set()
    titles: list[str] = []
    for match in _TASK_MARKER_RE.finditer(text or ""):
        title = " ".join(match.group("title").split()).strip()
        if not title or len(title) > 120:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        titles.append(title)
    return titles


def strip_task_markers(text: str) -> str:
    """Return *text* with any task markers removed, tidied for display."""
    cleaned = _TASK_MARKER_RE.sub("", text or "")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def recency_halflife_weight(age: int, halflife: float) -> float:
    """Return an exponential recency weight for a message *age* turns back.

    A weight of 1.0 is the most recent message, 0.5 is one halflife old, and so
    on. Exponential decay is used rather than a linear ramp because the value of
    a conversational turn falls off quickly once it is several turns back, while
    the immediately preceding turns matter almost equally.
    """
    if halflife <= 0:
        return 1.0
    return float(0.5 ** (max(0, age) / halflife))


def build_chat_history(
    conn: sqlite3.Connection,
    limit: int = 6,
    *,
    max_candidates: int = 40,
    recency_halflife: float = 6.0,
) -> list[dict[str, str]]:
    """Return chat messages weighted toward recent turns.

    The whole conversation stays in SQLite -- nothing is dropped on save. What
    changes here is *selection*: rather than taking the last ``limit`` messages
    strictly by timestamp, up to ``max_candidates`` recent messages are scored
    by recency (exponential decay with ``recency_halflife`` turns) and the
    highest-scoring ``limit`` are returned.

    Scoring by age rather than truncating by age means a still-relevant message
    from earlier in the conversation can survive while a run of trivial
    turn-taking ("ok", "thanks") from the last minute is dropped. The newest
    message is always included regardless of score, because dropping the turn
    being replied to would break the conversation outright.

    The result is always returned oldest-first, as the prompt builder expects.
    """
    candidates = db.list_llm_chat_messages(conn, limit=max_candidates)
    if len(candidates) <= limit:
        ordered = list(reversed(candidates))
        return [{"role": m.role, "content": m.content} for m in ordered]

    # ``candidates`` is newest-first, so the enumeration index is the age in
    # turns: index 0 is the newest message.
    scored = [
        (recency_halflife_weight(index, recency_halflife), index, message)
        for index, message in enumerate(candidates)
    ]
    # Highest weight first, keeping newest-first ordering as the tie-break so
    # equal-weight messages are chosen by recency rather than arbitrarily.
    scored.sort(key=lambda item: (-item[0], item[1]))

    chosen = scored[:limit]
    # Index 0 is the NEWEST message, so descending index is oldest-first --
    # the order the prompt builder and the chat transcript both expect.
    chosen.sort(key=lambda item: item[1], reverse=True)

    return [
        {"role": message.role, "content": message.content} for _, _, message in chosen
    ]
