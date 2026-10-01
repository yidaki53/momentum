"""Tests for the coach's task markers and context contents.

The coach may propose tasks, but a 1.1B model must never write to the
database on its own initiative: a hallucinated marker would invent work the
user never agreed to. These tests pin the guarantee that markers are parsed,
de-duplicated, stripped from the visible reply, and -- because the UI always
confirms first -- never create anything by themselves.
"""

from __future__ import annotations

from pathlib import Path

from momentum import db
from momentum.llm import context as ctx_mod
from momentum.models import TaskCreate


def test_extracts_task_markers() -> None:
    reply = (
        "Here are two small steps.\n\n"
        "[[task: Write the opening paragraph]]\n"
        "[[task: Email the draft to Sam]]\n"
    )
    assert ctx_mod.extract_task_requests(reply) == [
        "Write the opening paragraph",
        "Email the draft to Sam",
    ]


def test_markers_are_stripped_from_the_visible_reply() -> None:
    reply = "Let's start small.\n\n[[task: Draft one bullet]]\n"
    shown = ctx_mod.strip_task_markers(reply)
    assert "[[task:" not in shown
    assert shown == "Let's start small."


def test_reply_without_markers_is_untouched() -> None:
    reply = "That sounds hard. What would the smallest step be?"
    assert ctx_mod.extract_task_requests(reply) == []
    assert ctx_mod.strip_task_markers(reply) == reply


def test_duplicate_markers_are_collapsed() -> None:
    """A rambling model repeating itself must not flood the task list."""
    reply = "[[task: Email Sam]]\n[[task: email sam ]]\n[[task:Email Sam]]"
    assert ctx_mod.extract_task_requests(reply) == ["Email Sam"]


def test_absurdly_long_marker_is_rejected() -> None:
    """A runaway generation cannot create a 10k-character 'task'."""
    reply = "[[task: " + ("x" * 500) + "]]"
    assert ctx_mod.extract_task_requests(reply) == []


def test_malformed_marker_is_ignored() -> None:
    assert ctx_mod.extract_task_requests("[[task:]]") == []
    assert ctx_mod.extract_task_requests("[[task: no closing bracket") == []


def test_parsing_alone_creates_no_tasks(tmp_path: Path) -> None:
    """Parsing is pure: nothing reaches the database without confirmation."""
    conn = db.get_connection(tmp_path / "c.db")
    try:
        reply = "[[task: Something the model made up]]"
        assert ctx_mod.extract_task_requests(reply)
        assert ctx_mod.strip_task_markers(reply) == ""
        assert db.list_tasks(conn) == []
    finally:
        conn.close()


def test_prompt_tells_the_model_when_to_use_the_marker() -> None:
    """The model is instructed explicitly, including when *not* to use it."""
    from momentum.llm import prompts

    system = prompts.CHAT_SYSTEM_PROMPT
    assert "[[task:" in system
    assert "only when the user explicitly" in system.lower()


def test_context_includes_active_pending_and_completed_tasks(
    tmp_path: Path,
) -> None:
    """The coach is told about work in progress and work already finished."""
    conn = db.get_connection(tmp_path / "c.db")
    try:
        db.add_task(conn, TaskCreate(title="Finish the quarterly summary"))
        active = db.list_tasks(conn)[0]
        db.set_task_active(conn, active.id)

        done = db.add_task(conn, TaskCreate(title="Send the invoice"))
        db.complete_task(conn, done.id)
        db.add_task(conn, TaskCreate(title="Book the venue"))

        context = ctx_mod.build_user_context(conn)

        assert "Finish the quarterly summary" in context  # in progress
        assert "Book the venue" in context  # still pending
        assert "Send the invoice" in context  # already completed
    finally:
        conn.close()
