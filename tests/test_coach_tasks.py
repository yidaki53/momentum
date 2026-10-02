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


def test_prompt_never_shows_the_marker_unprompted() -> None:
    """The task marker must be invisible unless the user asked for a task.

    A 0.5B model given a worked example of a structured output reproduces it:
    with the marker permanently in the system prompt, "hi" came back as an
    invented task. This is the regression that caused it.
    """
    from momentum.llm import prompts

    assert "[[task:" not in prompts.CHAT_SYSTEM_PROMPT

    ordinary = prompts.build_chat_prompt("hi", "ctx", [])
    assert not any("[[task:" in m["content"] for m in ordinary)

    asking = prompts.build_chat_prompt("add a task to call Sam", "ctx", [])
    assert any("[[task:" in m["content"] for m in asking)


def test_task_intent_detection() -> None:
    from momentum.llm import prompts

    for message in ("hi", "hello", "how are you?", "what should I do today?"):
        assert not prompts.asks_for_task_creation(message), message
    for message in (
        "add a task to call Sam",
        "create a todo for the report",
        "make me a task for the laundry",
    ):
        assert prompts.asks_for_task_creation(message), message


_NARRATION = (
    "[This user is now creating a task for today. They will write a short "
    'introduction, such as "Today\'s Focus: [task]".]'
)


def test_narrated_reply_is_detected_but_task_markers_are_not() -> None:
    assert ctx_mod.is_narrated_reply(_NARRATION)
    assert not ctx_mod.is_narrated_reply("[[task: Call Sam]]")
    assert not ctx_mod.is_narrated_reply("Hi! How is your day going?")


def test_chat_prompt_drops_poisoned_history() -> None:
    """The phone's saved history: a narrated reply, orphaned 'hi's, and the
    current message already saved before the prompt was built."""
    from momentum.llm import prompts

    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": _NARRATION},
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "hello"},
    ]
    turns = [
        m
        for m in prompts.build_chat_prompt("hello", "ctx", history)
        if m["role"] != "system"
    ]

    assert turns == [{"role": "user", "content": "hello"}]


def test_chat_prompt_keeps_a_normal_exchange() -> None:
    from momentum.llm import prompts

    history = [
        {"role": "user", "content": "I can't start"},
        {"role": "assistant", "content": "That's common. Try two minutes."},
        {"role": "user", "content": "ok"},
    ]
    turns = [
        m
        for m in prompts.build_chat_prompt("ok", "ctx", history)
        if m["role"] != "system"
    ]

    assert turns == history
