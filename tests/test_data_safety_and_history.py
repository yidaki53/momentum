"""Tests for the requested behaviour changes: database survival across updates,
cloud-sync data safety, and recency-weighted chat history.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momentum import config as cfg
from momentum import db
from momentum.llm import context as ctx_mod
from momentum.models import LlmChatMessageCreate, TaskCreate

# ---------------------------------------------------------------------------
# Database survives updates
# ---------------------------------------------------------------------------


def test_default_db_is_created_when_absent(tmp_path: Path) -> None:
    """A missing database is created rather than treated as an error."""
    target = tmp_path / "fresh.db"
    assert not target.exists()

    conn = db.get_connection(target)
    try:
        db.add_task(conn, TaskCreate(title="first task"))
        assert target.exists()
    finally:
        conn.close()


def test_existing_db_is_reused_not_replaced(tmp_path: Path) -> None:
    """Reopening keeps existing rows -- the 'use it if it exists' rule."""
    target = tmp_path / "existing.db"
    first = db.get_connection(target)
    db.add_task(first, TaskCreate(title="keep me"))
    first.close()

    second = db.get_connection(target)
    try:
        assert "keep me" in [t.title for t in db.list_tasks(second)]
    finally:
        second.close()


# Orphaned-database adoption is covered by tests/test_recovery.py, which owns
# the fixture that re-enables the scan that conftest.py disables globally.


# ---------------------------------------------------------------------------
# Cloud sync must not lose data
# ---------------------------------------------------------------------------


def _seed(path: Path, tasks: int = 3) -> None:
    """Populate a database with tasks and chat history, then close it."""
    conn = db.get_connection(path)
    for i in range(tasks):
        db.add_task(conn, TaskCreate(title=f"task {i}"))
    db.add_llm_chat_message(
        conn, LlmChatMessageCreate(role="user", content="hello from before the move")
    )
    conn.close()


def test_cloud_sync_carries_existing_data_across(tmp_path: Path, monkeypatch) -> None:
    """Enabling cloud sync must not switch the app to an empty database.

    This was the "cloud support does not work at all" bug: the path was
    repointed without moving the data, so every task and message appeared to
    vanish the instant sync was turned on.
    """
    live = tmp_path / "live.db"
    _seed(live)

    cloud_root = tmp_path / "OneDrive"
    cloud_root.mkdir()
    monkeypatch.setitem(cfg._CLOUD_PRESETS, "onedrive", [cloud_root])
    monkeypatch.setattr(cfg, "_is_android", lambda: False)

    cfg.set_db_path(str(live))
    result = cfg.set_cloud_sync("onedrive")

    assert result is not None
    cloud_db = Path(result.db_path)
    assert cloud_db.exists()

    moved = db.get_connection(cloud_db)
    try:
        assert len(db.list_tasks(moved)) == 3
        assert db.list_llm_chat_messages(moved), "chat history must move too"
    finally:
        moved.close()


def test_cloud_sync_does_not_overwrite_populated_cloud_db(
    tmp_path: Path, monkeypatch
) -> None:
    """Re-running sync must never clobber data already in the cloud."""
    live = tmp_path / "live.db"
    _seed(live, tasks=2)

    cloud_root = tmp_path / "OneDrive"
    cloud_root.mkdir()
    monkeypatch.setitem(cfg._CLOUD_PRESETS, "onedrive", [cloud_root])
    monkeypatch.setattr(cfg, "_is_android", lambda: False)

    cfg.set_db_path(str(live))
    cfg.set_cloud_sync("onedrive")
    cloud_db = cfg.get_db_path()

    conn = db.get_connection(cloud_db)
    db.add_task(conn, TaskCreate(title="added after sync"))
    conn.close()

    assert cfg._migrate_database_to(cloud_db) is False
    final = db.get_connection(cloud_db)
    try:
        titles = [t.title for t in db.list_tasks(final)]
        assert "added after sync" in titles
        assert "task 0" in titles
    finally:
        final.close()


def test_cloud_sync_is_safe_when_source_is_empty(tmp_path: Path, monkeypatch) -> None:
    """Migrating an empty database is a no-op, not a crash."""
    live = tmp_path / "empty.db"
    db.get_connection(live).close()

    cloud_root = tmp_path / "OneDrive"
    cloud_root.mkdir()
    monkeypatch.setitem(cfg._CLOUD_PRESETS, "onedrive", [cloud_root])
    monkeypatch.setattr(cfg, "_is_android", lambda: False)

    cfg.set_db_path(str(live))
    assert cfg._migrate_database_to(cloud_root / "momentum" / "momentum.db") is False


# ---------------------------------------------------------------------------
# Recency-weighted chat history
# ---------------------------------------------------------------------------


def test_recency_weight_decays_with_age() -> None:
    assert ctx_mod.recency_halflife_weight(0, 6.0) == pytest.approx(1.0)
    assert ctx_mod.recency_halflife_weight(6, 6.0) == pytest.approx(0.5)
    assert ctx_mod.recency_halflife_weight(12, 6.0) == pytest.approx(0.25)
    weights = [ctx_mod.recency_halflife_weight(a, 6.0) for a in range(10)]
    assert weights == sorted(weights, reverse=True)


def test_chat_history_is_returned_oldest_first(tmp_path: Path) -> None:
    conn = db.get_connection(tmp_path / "c.db")
    try:
        for i in range(4):
            db.add_llm_chat_message(
                conn, LlmChatMessageCreate(role="user", content=f"m{i}")
            )
        history = ctx_mod.build_chat_history(conn, limit=4)
        assert [m["content"] for m in history] == ["m0", "m1", "m2", "m3"]
    finally:
        conn.close()


def test_chat_history_selects_by_recency_not_just_latest(tmp_path: Path) -> None:
    """The newest message is always present; selection is recency-weighted.

    Nothing is deleted -- the whole transcript stays in SQLite. Only which
    messages are handed to the model is chosen by recency.
    """
    conn = db.get_connection(tmp_path / "c.db")
    try:
        for i in range(12):
            db.add_llm_chat_message(
                conn, LlmChatMessageCreate(role="user", content=f"m{i}")
            )

        contents = [m["content"] for m in ctx_mod.build_chat_history(conn, limit=4)]

        assert len(contents) == 4
        assert "m11" in contents  # newest always included
        assert contents == sorted(contents, key=lambda c: int(c[1:]))
        assert len(db.list_llm_chat_messages(conn, limit=100)) == 12  # nothing lost
    finally:
        conn.close()


def test_short_history_is_returned_unchanged(tmp_path: Path) -> None:
    conn = db.get_connection(tmp_path / "c.db")
    try:
        db.add_llm_chat_message(
            conn, LlmChatMessageCreate(role="user", content="only one")
        )
        history = ctx_mod.build_chat_history(conn, limit=6)
        assert [m["content"] for m in history] == ["only one"]
    finally:
        conn.close()
