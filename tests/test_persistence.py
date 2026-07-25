"""Update-safe persistence: user data survives an app update on Android.

Android preserves app-private internal storage (``getFilesDir()``) across app
updates by design. This test makes that guarantee explicit and regression-proof:
it writes a realistic data set to a DB + config rooted at an Android-style
app-private path, then simulates an update by closing the connection (the
running process is replaced) and re-opening from the *same* path (new process,
same preserved storage), asserting every item round-trips.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

from momentum import db
from momentum.config import get_db_path, load_config, save_config
from momentum.models import (
    ActJournalEntryCreate,
    AppConfig,
    AssessmentResultCreate,
    AssessmentType,
    FocusSessionCreate,
    LlmChatMessageCreate,
    TaskCreate,
    TaskStatus,
    ThemeMode,
    TimerCycleMode,
    WindowPosition,
)


def _android_paths(root: Path):
    """Android app-private layout: <filesdir>/data/{config,db}."""
    data_dir = root / "data"
    return (
        data_dir,
        data_dir / "config",
        data_dir / "config" / "config.json",
        data_dir / "db",
    )


def _android_patches(root: Path):
    """Patch momentum.config to resolve all storage under ``root`` as Android.

    ``_DATA_DIR`` is only created at module load when ``_is_android()`` is true,
    so it is not patched here (storage resolves via ``_DB_DIR`` / ``_CONFIG_DIR``
    / ``_CONFIG_FILE``, which always exist). ``_is_android`` and
    ``_android_data_dir`` are patched for any runtime callers.
    """
    _data_dir, config_dir, config_file, db_dir = _android_paths(root)
    return (
        patch("momentum.config._is_android", return_value=True),
        patch("momentum.config._android_data_dir", return_value=_data_dir),
        patch("momentum.config._CONFIG_DIR", config_dir),
        patch("momentum.config._CONFIG_FILE", config_file),
        patch("momentum.config._DB_DIR", db_dir),
        patch("momentum.config._LEGACY_CONFIG_FILES", []),
        patch("momentum.config._LEGACY_DB_FILES", []),
        patch.dict(os.environ, {"ANDROID_ARGUMENT": "private=/tmp"}, clear=False),
    )


def _write_realistic_dataset() -> None:
    """Write a representative user dataset: config + every DB table."""
    # Config with non-default values across every persisted field.
    save_config(
        AppConfig(
            db_path=None,
            window_position=WindowPosition.TOP_LEFT,
            theme_mode=ThemeMode.LIGHT,
            accessibility_large_text=True,
            accessibility_high_contrast=True,
            accessibility_reduce_visual_load=True,
            timer_cycle_mode=TimerCycleMode.AUTO,
            check_updates_at_startup=False,
            last_update_check_unix=1700000000,
            show_llm_welcome=False,
            llm_model="qwen",
        )
    )

    conn = db.get_connection()
    try:
        task = db.add_task(conn, TaskCreate(title="Write introduction"))
        db.set_task_active(conn, task.id)
        sub = db.add_task(conn, TaskCreate(title="Draft outline", parent_id=task.id))

        # Completing a task writes the daily_log row for today.
        db.complete_task(conn, sub.id)

        # A focus session also updates the daily_log.
        db.log_focus_session(
            conn, FocusSessionCreate(task_id=task.id, duration_minutes=25)
        )

        # Assessment with multi-domain scores (JSON-serialised).
        db.save_assessment(
            conn,
            AssessmentResultCreate(
                assessment_type=AssessmentType.BDEFS,
                score=42,
                max_score=80,
                domain_scores={"self-management": 11, "self-organisation": 9},
            ),
        )

        # ACT journal entry (all five fields).
        db.add_act_journal_entry(
            conn,
            ActJournalEntryCreate(
                values_focus="Being present with my family",
                challenge_context="Avoiding the work I care about",
                thoughts_feelings="I feel overwhelmed and stuck",
                defusion_reframe="I am having the thought that I am stuck",
                committed_action="Open the document for two minutes",
            ),
        )

        # AI Coach chat history.
        db.add_llm_chat_message(
            conn,
            LlmChatMessageCreate(role="user", content="I can't get started today."),
        )
        db.add_llm_chat_message(
            conn,
            LlmChatMessageCreate(
                role="assistant",
                content="That is a hard place to be. Let's pick one tiny step.",
            ),
        )
    finally:
        conn.close()


def test_user_data_survives_an_update_on_android(tmp_path: Path) -> None:
    """Data written before an update is fully present after a reopen."""
    patches = _android_patches(tmp_path)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        # The DB path must resolve under the Android app-private tree.
        db_path = get_db_path()
        assert db_path == tmp_path / "data" / "db" / "momentum.db"

        _write_realistic_dataset()

        # --- Simulate the update: the running process is replaced. ---
        # We simply drop all in-memory state (a fresh get_connection below models
        # a new process opening the same preserved app-private file).

        # --- "After update": reopen from the same path and verify everything. ---
        # Config round-trips with every persisted field intact.
        cfg = load_config()
        assert cfg.window_position == WindowPosition.TOP_LEFT
        assert cfg.theme_mode == ThemeMode.LIGHT
        assert cfg.accessibility_large_text is True
        assert cfg.accessibility_high_contrast is True
        assert cfg.accessibility_reduce_visual_load is True
        assert cfg.timer_cycle_mode == TimerCycleMode.AUTO
        assert cfg.check_updates_at_startup is False
        assert cfg.last_update_check_unix == 1700000000
        assert cfg.show_llm_welcome is False
        assert cfg.llm_model == "qwen"

        conn = db.get_connection()
        try:
            # Tasks (parent + subtask) survive.
            tasks = db.list_tasks(conn)
            titles = {t.title for t in tasks}
            assert {"Write introduction", "Draft outline"} <= titles
            parent = next(t for t in tasks if t.title == "Write introduction")
            assert parent.status == TaskStatus.ACTIVE
            subtasks = db.get_subtasks(conn, parent.id)
            assert len(subtasks) == 1
            assert subtasks[0].status == TaskStatus.DONE
            assert subtasks[0].completed_at is not None

            # Focus session survives and is linked to the task.
            sessions = db.list_focus_sessions(conn)
            assert len(sessions) == 1
            assert sessions[0].task_id == parent.id
            assert sessions[0].duration_minutes == 25

            # Daily log (written by complete_task + log_focus_session) survives.
            today = db.get_daily_log(conn, date.today())
            assert today.tasks_completed == 1
            assert today.focus_minutes == 25

            # Assessment with domain_scores (JSON) survives intact.
            assessments = db.list_assessments(conn, AssessmentType.BDEFS)
            assert len(assessments) == 1
            r = assessments[0]
            assert r.score == 42
            assert r.max_score == 80
            assert r.domain_scores == {"self-management": 11, "self-organisation": 9}

            # ACT journal entry survives with all five fields.
            entries = db.list_act_journal_entries(conn)
            assert len(entries) == 1
            e = entries[0]
            assert e.values_focus == "Being present with my family"
            assert e.challenge_context == "Avoiding the work I care about"
            assert e.thoughts_feelings == "I feel overwhelmed and stuck"
            assert e.defusion_reframe == "I am having the thought that I am stuck"
            assert e.committed_action == "Open the document for two minutes"

            # AI Coach chat history survives (list_llm_chat_messages returns
            # newest-first; same-second timestamps make tie order undefined, so
            # assert the message set order-independently).
            messages = db.list_llm_chat_messages(conn, limit=10)
            assert len(messages) == 2
            pairs = {(m.role, m.content) for m in messages}
            assert ("user", "I can't get started today.") in pairs
            assert any(
                r == "assistant" and c.startswith("That is a hard place to be")
                for r, c in pairs
            )
        finally:
            conn.close()

        # The on-disk files are the actual proof: they exist at the app-private
        # path and were not wiped by the "update" (reopen).
        assert (tmp_path / "data" / "config" / "config.json").exists()
        assert db_path.exists()
