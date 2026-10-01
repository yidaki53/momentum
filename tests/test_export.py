"""Tests for the plain-text exporters.

The exports exist so a self-hosted, open-source, plain-text-first app can keep
its data in files a human can read, diff, and commit. That property only holds
if the documents are structurally sound and if user text cannot break them, so
these tests check both the shape of the output and the escaping of free text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momentum import db, export
from momentum.models import (
    ActJournalEntryCreate,
    AssessmentResultCreate,
    AssessmentType,
    FocusSessionCreate,
    LlmChatMessageCreate,
    TaskCreate,
)


@pytest.fixture()
def conn(tmp_path: Path):
    connection = db.get_connection(tmp_path / "momentum.db")
    db.add_task(connection, TaskCreate(title="Write the report"))
    db.log_focus_session(connection, FocusSessionCreate(duration_minutes=25))
    db.add_act_journal_entry(
        connection,
        ActJournalEntryCreate(
            values_focus="creativity",
            challenge_context="starting is hard",
            thoughts_feelings="I am behind",
            defusion_reframe="I notice I am behind",
            committed_action="Write two sentences",
        ),
    )
    db.save_assessment(
        connection,
        AssessmentResultCreate(
            assessment_type=AssessmentType.BDEFS,
            score=20,
            max_score=80,
            domain_scores={"self-management": 5},
        ),
    )
    db.add_llm_chat_message(
        connection, LlmChatMessageCreate(role="user", content="hi there")
    )
    db.add_llm_chat_message(
        connection, LlmChatMessageCreate(role="assistant", content="Hello!")
    )
    yield connection
    connection.close()


def test_org_export_has_title_and_headings(conn) -> None:
    text = export.export_org(conn)
    assert text.startswith("#+TITLE: Momentum Export")
    assert "* Tasks" in text
    assert "* AI Coach Chat" in text


def test_markdown_export_has_headings_and_table(conn) -> None:
    text = export.export_markdown(conn)
    assert text.startswith("# Momentum Export")
    assert "## Tasks" in text
    assert "| ID | Title | Status |" in text


def test_every_section_is_present_in_both_formats(conn) -> None:
    for text in (export.export_org(conn), export.export_markdown(conn)):
        for heading in ("Tasks", "Focus Sessions", "Assessments", "ACT Journal"):
            assert heading in text
        assert "AI Coach Chat" in text


def test_pipe_in_task_title_cannot_break_the_table(conn) -> None:
    """A ``|`` in free text would otherwise add a phantom column."""
    db.add_task(conn, TaskCreate(title="a | b | c"))
    for text in (export.export_org(conn), export.export_markdown(conn)):
        # The escaped form must appear and the raw triple-pipe must not.
        assert "a | b | c" not in text
        assert ("a \\| b \\| c" in text) or ("a \\vert{} b \\vert{} c" in text)


def test_multi_line_chat_is_preserved_and_contained(conn) -> None:
    """Chat bodies are free text and keep their line breaks.

    Escaping applies to table cells only; collapsing a chat message's newlines
    would mangle the conversation. What matters is that the multi-line block
    stays under its own heading and does not swallow the following section.
    """
    db.add_llm_chat_message(
        conn, LlmChatMessageCreate(role="user", content="line one\nline two")
    )
    text = export.export_markdown(conn)
    assert "line one\nline two" in text

    # The newlines live inside the final chat block, not before it.
    body_at = text.index("### **You**")
    assert text.index("line one") > body_at


def test_chat_is_exported_oldest_first(conn) -> None:
    text = export.export_markdown(conn)
    assert text.index("hi there") < text.index("Hello!")


def test_unknown_format_is_rejected(conn) -> None:
    with pytest.raises(ValueError):
        export.render_document(conn, "latex")


def test_export_handles_an_empty_database(tmp_path: Path) -> None:
    empty = db.get_connection(tmp_path / "empty.db")
    try:
        text = export.export_markdown(empty)
        assert "No tasks recorded yet." in text
    finally:
        empty.close()


def test_export_to_path_writes_the_file(tmp_path: Path) -> None:
    db_path = tmp_path / "momentum.db"
    connection = db.get_connection(db_path)
    db.add_task(connection, TaskCreate(title="export me"))
    connection.close()

    dest = tmp_path / "out" / "momentum.org"
    written = export.export_to_path(db_path, dest, export.ExportFormat.ORG)

    assert written == dest
    assert "#+TITLE: Momentum Export" in dest.read_text(encoding="utf-8")
