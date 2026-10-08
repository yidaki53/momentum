"""Tests for the AI Coach schematic feedback signal store.

Signals are stored as short, structured tuples — never free text. The tests
below pin that behaviour: a signal log is visible, exportable and deletable,
but it cannot carry a user's words.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from momentum import db
from momentum.models import CoachSignal


@pytest.fixture()
def conn(tmp_path: Path):
    """Provide a fresh database for each signal-store test."""
    db_path = tmp_path / "signal_test.db"
    connection = db.get_connection(db_path=db_path)
    yield connection
    connection.close()


def test_schema_is_created(conn) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='coach_signals'"
    ).fetchone()
    assert row is not None


def test_add_and_get(conn) -> None:
    created = db.add_coach_signal(
        conn,
        intent="stop-medication",
        verdict="non-responsive",
        failure="answered with an unrelated task action",
        user_data_fields_used=["active_tasks", "bddefs_score"],
    )
    assert isinstance(created, CoachSignal)
    assert created.intent == "stop-medication"
    assert created.verdict == "non-responsive"
    assert created.failure == "answered with an unrelated task action"
    assert created.user_data_fields_used == ["active_tasks", "bddefs_score"]

    fetched = db.list_coach_signals(conn)
    assert len(fetched) == 1
    assert fetched[0].id == created.id


def test_data_fields_always_list(conn) -> None:
    db.add_coach_signal(conn, "task-creation", "rejected", "no marker emitted")
    fetched = db.list_coach_signals(conn)
    assert isinstance(fetched[0].user_data_fields_used, list)
    assert fetched[0].user_data_fields_used == []


def test_export_is_json_serialisable_without_free_text(conn) -> None:
    db.add_coach_signal(
        conn,
        "stop-medication",
        "non-responsive",
        "short reason",
        user_data_fields_used=["active_tasks"],
    )
    export = db.export_coach_signals(conn)
    assert len(export) == 1
    row = export[0]
    assert set(row) == {
        "id",
        "intent",
        "verdict",
        "failure",
        "user_data_fields_used",
        "created_at",
    }
    assert isinstance(row["user_data_fields_used"], list)
    assert "free-text user content" not in repr(row)


def test_delete_single_and_clear(conn) -> None:
    first = db.add_coach_signal(conn, "a", "v", "f1")
    second = db.add_coach_signal(conn, "a", "v", "f2")
    assert db.delete_coach_signal(conn, second.id) is True
    rows = db.list_coach_signals(conn)
    assert len(rows) == 1
    assert rows[0].id == first.id
    assert rows[0].intent == "a"
    assert rows[0].verdict == "v"
    assert rows[0].failure == "f1"
    assert rows[0].user_data_fields_used == []
    assert db.clear_coach_signals(conn) == 1


def test_list_respects_limit(conn) -> None:
    for i in range(5):
        db.add_coach_signal(conn, "intent", "verdict", f"failure {i}")
    assert len(db.list_coach_signals(conn, limit=2)) == 2


def test_config_accessors_roundtrip(tmp_path: Path) -> None:
    from momentum import config as cfg

    original = cfg.load_config()
    try:
        conf = cfg.set_signal_collection_enabled(True)
        assert conf.signal_collection_enabled is True
        assert cfg.load_config().signal_collection_enabled is True

        conf = cfg.set_signal_collection_enabled(False)
        assert conf.signal_collection_enabled is False
    finally:
        cfg.set_signal_collection_enabled(original.signal_collection_enabled)


# ---------------------------------------------------------------------------
# context_fields_used: what a signal records as "fields used"
# ---------------------------------------------------------------------------


def test_context_fields_used_returns_names_only() -> None:
    from momentum.llm.context import context_fields_used

    context = (
        "Active tasks: Call Sam about the warranty\n"
        "Pending tasks (1): Email the supervisor\n"
        "Recently completed tasks: Read two papers\n"
        "Activity: 3 task(s) completed today; 45 focus minute(s) today; "
        "streak 2 day(s)\n"
        "BDEFS history: 2026-01-01 51/70; 2025-12-01 48/70\n"
        "Recent ACT check-ins:\n"
        "  Values: family\n"
        "    Committed action: call the plumber\n"
    )
    fields = context_fields_used(context)
    assert fields == [
        "active_tasks",
        "pending_tasks",
        "completed_tasks",
        "activity",
        "bdefs_history",
        "act_journal",
    ]
    # The names must not smuggle any of the context's values along with them.
    joined = " ".join(fields)
    for value in ("Sam", "supervisor", "papers", "plumber", "51/70"):
        assert value not in joined


def test_context_fields_used_tolerates_empty_input() -> None:
    from momentum.llm.context import context_fields_used

    assert context_fields_used("") == []
    assert context_fields_used(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Wiring contracts: the reply flows must actually record signals.
#
# The closures in gui.py and the handlers in mobile/main.py are not unit
# testable without a running UI, so these pin the wiring at the source level
# the same way test_mobile_ui_contract does.
# ---------------------------------------------------------------------------


def _source(rel_path: str) -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / rel_path).read_text(encoding="utf-8")


def _function_sources(src: str, name: str) -> list[str]:
    """Return the source of every function named *name* anywhere in *src*."""
    tree = ast.parse(src)
    segments: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(src, node)
            if segment is not None:
                segments.append(segment)
    assert segments, f"function {name!r} not found"
    return segments


def _some_function_source(src: str, name: str) -> str:
    segments = _function_sources(src, name)
    assert segments
    return segments[0]


def test_desktop_reply_flow_records_signals() -> None:
    src = _source("momentum/gui.py")

    done = _some_function_source(src, "_on_done")
    for verdict in ("narrated", "task-proposed", "empty"):
        assert f'"{verdict}"' in done, verdict
    assert "_report_feedback(" in done

    err = _some_function_source(src, "_on_error")
    assert "_report_feedback(" in err
    assert '"generation-error"' in err

    send = _some_function_source(src, "_send_message")
    assert "asks_for_task_creation" in send
    assert "context_fields_used" in send
    assert '"engine-unavailable"' in send


def test_desktop_report_feedback_is_opt_in_and_renders() -> None:
    src = _source("momentum/gui.py")
    body = _some_function_source(src, "_report_feedback")
    assert "signal_collection_enabled" in body
    assert "_render_signal_log()" in body
    assert "user_data_fields_used" in body


def test_mobile_reply_flow_records_signals() -> None:
    src = _source("mobile/main.py")

    finish = _some_function_source(src, "_finish_reply")
    for verdict in ("narrated", "task-proposed", "empty"):
        assert f'"{verdict}"' in finish, verdict
    assert finish.count("_report_feedback(") >= 3

    fail = _some_function_source(src, "_fail_reply")
    assert "_report_feedback(" in fail
    assert '"generation-error"' in fail

    send = _some_function_source(src, "send_message")
    assert "_asks_for_task_creation(" in send
    assert "_context_fields_used(" in send


def test_mobile_report_feedback_is_opt_in_and_renders() -> None:
    src = _source("mobile/main.py")
    body = _some_function_source(src, "_report_feedback")
    assert "signal_collection_enabled" in body
    assert "_render_signal_log()" in body
    assert "user_data_fields_used" in body


def test_mobile_log_list_and_hide_handlers_exist() -> None:
    """The List/Close buttons are bound at render time; both must exist."""
    src = _source("mobile/main.py")
    _function_sources(src, "_list_signal_log")
    _function_sources(src, "_hide_signal_log")
