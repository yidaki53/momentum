"""Export Momentum data as plain-text, diffable documents.

Both output formats are deliberately dependency-free, human-editable, and
usable with plain text tools -- the properties that matter for a
self-hosted, open-source, plain-text-first app:

* **Org mode** (``.org``) -- ``#+TITLE`` properties plus Org's ``*`` heading
  rules. Use this when the data should be greppable, editable in Emacs, and
  diffable in git.
* **Markdown** (``.md``) -- ATX headings and tables. Broadest interoperability
  (GitHub, Obsidian, any renderer).

Neither format is a lossy *storage* format; SQLite remains the source of truth.
These are projections of it, so an export can always be regenerated and the
exported files are safe to edit, diff, and commit alongside a dotfiles repo.

Every user-authored string is escaped for the target syntax, because task titles
and chat messages are free text and a stray ``|`` would otherwise corrupt the
table structure.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from momentum import db
from momentum.models import TaskStatus

__all__ = [
    "ExportFormat",
    "export_markdown",
    "export_org",
    "export_to_path",
    "render_document",
]

MAX_ASSESSMENTS = 100
MAX_CHAT_MESSAGES = 500
MAX_SESSIONS = 200


class ExportFormat:
    """Known export formats."""

    MARKDOWN = "markdown"
    ORG = "org"

    ALL = (MARKDOWN, ORG)


def _md_escape(text: str) -> str:
    """Escape a user string for a Markdown table cell."""
    return (
        (text or "")
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _org_escape(text: str) -> str:
    """Escape a user string for an Org mode table cell.

    ``|`` separates columns, and newlines end a row, so both are neutralised.
    """
    return (
        (text or "")
        .replace("\\", "\\\\")
        .replace("|", "\\vert{}")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _iso(value: object) -> str:
    """Render a datetime for a document, tolerating None."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _heading(level: int, text: str, fmt: str) -> str:
    """Return a heading line for the requested format."""
    if fmt == ExportFormat.MARKDOWN:
        return "#" * level + " " + text
    return "*" * level + " " + text


def _table(
    headers: list[str],
    rows: Iterable[list[str]],
    *,
    fmt: str,
) -> list[str]:
    """Render a pipe-aligned table in the requested format."""
    escape = _md_escape if fmt == ExportFormat.MARKDOWN else _org_escape
    header_cells = [escape(h) for h in headers]
    body = [[escape(cell) for cell in row] for row in rows]

    if fmt == ExportFormat.MARKDOWN:
        lines = [
            "| " + " | ".join(header_cells) + " |",
            "| " + " | ".join("---" for _ in header_cells) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return lines

    widths = [
        max([len(h)] + [len(row[i]) for row in body])
        for i, h in enumerate(header_cells)
    ]

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [_row(header_cells), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(_row(row) for row in body)
    return lines


def _tasks_section(conn: sqlite3.Connection, fmt: str) -> list[str]:
    """Render the Tasks table."""
    lines = [_heading(2, "Tasks", fmt), ""]
    tasks = db.list_tasks(conn)
    if not tasks:
        return lines + ["No tasks recorded yet.", ""]
    rows = [
        [
            str(t.id),
            t.title,
            t.status.value if isinstance(t.status, TaskStatus) else str(t.status),
            _iso(t.created_at),
            _iso(t.completed_at),
        ]
        for t in tasks
    ]
    return (
        lines
        + _table(["ID", "Title", "Status", "Created", "Completed"], rows, fmt=fmt)
        + [""]
    )


def _sessions_section(conn: sqlite3.Connection, fmt: str) -> list[str]:
    """Render the Focus Sessions table."""
    lines = [_heading(2, "Focus Sessions", fmt), ""]
    sessions = db.list_focus_sessions(conn, limit=MAX_SESSIONS)
    if not sessions:
        return lines + ["No focus sessions recorded yet.", ""]
    rows = [
        [_iso(s.completed_at), str(s.duration_minutes), str(s.task_id or "")]
        for s in sessions
    ]
    return lines + _table(["Completed", "Minutes", "Task ID"], rows, fmt=fmt) + [""]


def _assessments_section(conn: sqlite3.Connection, fmt: str) -> list[str]:
    """Render the Assessments table."""
    lines = [_heading(2, "Assessments", fmt), ""]
    results = db.list_assessments(conn, limit=MAX_ASSESSMENTS)
    if not results:
        return lines + ["No assessments recorded yet.", ""]
    rows = []
    for a in results:
        kind = getattr(a.assessment_type, "value", None) or str(a.assessment_type)
        rows.append([_iso(a.taken_at), kind, str(a.score), str(a.max_score)])
    return lines + _table(["Taken", "Type", "Score", "Max"], rows, fmt=fmt) + [""]


def _act_section(conn: sqlite3.Connection, fmt: str) -> list[str]:
    """Render the ACT journal as dated subsections."""
    lines = [_heading(2, "ACT Journal", fmt), ""]
    entries = db.list_act_journal_entries(conn, limit=MAX_ASSESSMENTS)
    if not entries:
        return lines + ["No ACT journal entries yet.", ""]
    for entry in entries:
        stamp = _iso(entry.created_at)
        lines += [_heading(3, stamp, fmt), ""]
        for label, value in (
            ("Values", entry.values_focus),
            ("Challenge", entry.challenge_context),
            ("Thoughts and feelings", entry.thoughts_feelings),
            ("Defusion reframe", entry.defusion_reframe),
            ("Committed action", entry.committed_action),
        ):
            text = (value or "").strip()
            if text:
                lines.append(f"- {label}: {text}")
        lines.append("")
    return lines


def _chat_section(conn: sqlite3.Connection, fmt: str) -> list[str]:
    """Render the AI Coach transcript, oldest first."""
    lines = [_heading(2, "AI Coach Chat", fmt), ""]
    messages = db.list_llm_chat_messages(conn, limit=MAX_CHAT_MESSAGES)
    if not messages:
        return lines + ["No chat messages yet.", ""]
    for message in reversed(messages):
        who = "You" if message.role == "user" else "Coach"
        stamp = _iso(message.created_at)
        label = (
            f"**{who}** ({stamp})"
            if fmt == ExportFormat.MARKDOWN
            else f"{who} ({stamp})"
        )
        lines += [_heading(3, label, fmt), "", message.content, ""]
    return lines


def render_document(conn: sqlite3.Connection, fmt: str = ExportFormat.MARKDOWN) -> str:
    """Render the whole Momentum database as a single document."""
    if fmt not in ExportFormat.ALL:
        raise ValueError(
            f"Unknown export format {fmt!r}; expected one of "
            f"{', '.join(ExportFormat.ALL)}"
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if fmt == ExportFormat.ORG:
        head = ["#+TITLE: Momentum Export", f"#+DATE: {now}", ""]
    else:
        head = ["# Momentum Export", "", f"_Exported {now}_", ""]

    lines = head
    lines += _tasks_section(conn, fmt)
    lines += _sessions_section(conn, fmt)
    lines += _assessments_section(conn, fmt)
    lines += _act_section(conn, fmt)
    lines += _chat_section(conn, fmt)
    return "\n".join(lines).rstrip() + "\n"


def export_markdown(conn: sqlite3.Connection) -> str:
    """Render the database as Markdown."""
    return render_document(conn, ExportFormat.MARKDOWN)


def export_org(conn: sqlite3.Connection) -> str:
    """Render the database as Org mode."""
    return render_document(conn, ExportFormat.ORG)


def export_to_path(
    db_path: Path,
    destination: Path,
    fmt: str = ExportFormat.MARKDOWN,
) -> Path:
    """Write an export of *db_path* to *destination* and return the path."""
    conn = db.get_connection(Path(db_path))
    try:
        text = render_document(conn, fmt)
    finally:
        conn.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination
