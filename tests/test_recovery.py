"""Tests for data recovery (orphaned-DB adoption) and backup/restore."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from momentum import db, recovery


@pytest.fixture(autouse=True)
def _enable_recovery(monkeypatch):
    """This suite opts IN to recovery (conftest disables it globally).

    Cloud presets are stubbed out too: the real machine may hold a genuine
    cloud-synced Momentum DB (e.g. ~/OneDrive/momentum/momentum.db) which
    must never leak into tmp targets.
    """
    monkeypatch.delenv("MOMENTUM_DISABLE_RECOVERY", raising=False)
    recovery.reset_recovery_state()
    with patch("momentum.config._CLOUD_PRESETS", {}):
        yield
    recovery.reset_recovery_state()


def _make_db(path: Path, titles: list[str]) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(db._SCHEMA)
    for title in titles:
        conn.execute(
            "INSERT INTO tasks (title, status, created_at) "
            "VALUES (?, 'pending', '2026-01-01T00:00:00')",
            (title,),
        )
    conn.commit()
    conn.close()


class TestCountUserRows:
    def test_counts_tasks_and_assessments(self, tmp_path: Path) -> None:
        target = tmp_path / "momentum.db"
        _make_db(target, ["a", "b"])
        assert recovery.count_user_rows(target) == (2, 0)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert recovery.count_user_rows(tmp_path / "nope.db") is None

    def test_non_db_file_returns_none(self, tmp_path: Path) -> None:
        junk = tmp_path / "momentum.db"
        junk.write_text("this is not sqlite")
        assert recovery.count_user_rows(junk) is None


class TestRecoverDefaultDb:
    def test_adopts_richest_orphaned_db(self, tmp_path: Path) -> None:
        """Empty default + richer DB under a legacy root -> adopted."""
        default = tmp_path / "data" / "db" / "momentum.db"
        default.parent.mkdir(parents=True)
        _make_db(default, [])  # empty schema-only DB at the new location
        orphan_root = tmp_path / "old_root"
        orphan = orphan_root / "momentum.db"
        orphan_root.mkdir(parents=True)
        _make_db(orphan, ["old task 1", "old task 2", "old task 3"])

        with (
            patch("momentum.config.app_data_root", return_value=tmp_path / "data"),
            patch("momentum.config._is_android", return_value=False),
            patch("momentum.config._DB_DIR", orphan_root),
        ):
            assert recovery.recover_default_db(default) is True

        counts = recovery.count_user_rows(default)
        assert counts == (3, 0)
        # The empty file that used to sit at the target is preserved aside.
        assert default.with_name("momentum.db.empty.bak").exists()

    def test_nonempty_default_is_never_touched(self, tmp_path: Path) -> None:
        default = tmp_path / "momentum.db"
        _make_db(default, ["keep me"])
        other = tmp_path / "sub" / "momentum.db"
        other.parent.mkdir()
        _make_db(other, ["x", "y", "z"])

        with (
            patch("momentum.config.app_data_root", return_value=tmp_path),
            patch("momentum.config._is_android", return_value=False),
            patch("momentum.config._DB_DIR", tmp_path),
        ):
            assert recovery.recover_default_db(default) is False
        assert recovery.count_user_rows(default) == (1, 0)

    def test_missing_default_with_orphan_adopts(self, tmp_path: Path) -> None:
        """No file at target at all (fresh install after layout change)."""
        default = tmp_path / "fresh" / "momentum.db"
        orphan_root = tmp_path / "legacy"
        orphan = orphan_root / "momentum.db"
        orphan_root.mkdir()
        _make_db(orphan, ["only task"])

        with (
            patch("momentum.config.app_data_root", return_value=tmp_path / "fresh"),
            patch("momentum.config._is_android", return_value=False),
            patch("momentum.config._DB_DIR", orphan_root),
        ):
            assert recovery.recover_default_db(default) is True
        assert recovery.count_user_rows(default) == (1, 0)

    def test_no_candidates_is_noop(self, tmp_path: Path) -> None:
        default = tmp_path / "data" / "momentum.db"
        default.parent.mkdir()
        with (
            patch("momentum.config.app_data_root", return_value=tmp_path / "data"),
            patch("momentum.config._is_android", return_value=False),
            patch("momentum.config._DB_DIR", tmp_path / "data"),
        ):
            assert recovery.recover_default_db(default) is False

    def test_runs_only_once_per_process(self, tmp_path: Path) -> None:
        default = tmp_path / "momentum.db"
        with (
            patch("momentum.config.app_data_root", return_value=tmp_path),
            patch("momentum.config._is_android", return_value=False),
            patch("momentum.config._DB_DIR", tmp_path),
        ):
            assert recovery.recover_default_db(default) is False  # first call
            assert recovery.recover_default_db(default) is False  # guarded

    def test_env_var_disables_recovery(self, tmp_path: Path) -> None:
        default = tmp_path / "data" / "momentum.db"
        default.parent.mkdir()
        _make_db(default, [])
        orphan_root = tmp_path / "legacy"
        orphan = orphan_root / "momentum.db"
        orphan_root.mkdir()
        _make_db(orphan, ["should not adopt"])

        with (
            patch("momentum.config.app_data_root", return_value=tmp_path / "data"),
            patch("momentum.config._is_android", return_value=False),
            patch("momentum.config._DB_DIR", orphan_root),
            patch.dict(os.environ, {"MOMENTUM_DISABLE_RECOVERY": "1"}),
        ):
            recovery.reset_recovery_state()
            assert recovery.recover_default_db(default) is False
        assert recovery.count_user_rows(default) == (0, 0)


class TestBackupRoundtrip:
    def test_export_then_restore(self, tmp_path: Path) -> None:
        import dataclasses
        import json

        from momentum.models import AppConfig

        p_cfg = tmp_path / "config"
        p_file = p_cfg / "config.json"
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        real_db = db_dir / "momentum.db"
        _make_db(real_db, ["task A", "task B"])

        p_cfg.mkdir(parents=True, exist_ok=True)
        p_file.write_text(json.dumps(dataclasses.asdict(AppConfig())))

        with (
            patch("momentum.config._CONFIG_DIR", p_cfg),
            patch("momentum.config._CONFIG_FILE", p_file),
            patch("momentum.config._DB_DIR", db_dir),
            patch("momentum.config.load_config", return_value=AppConfig()),
        ):
            saved = recovery.export_backup(tmp_path / "out")
            assert saved.exists()
            assert recovery.count_user_rows(saved) == (2, 0)
            assert list((tmp_path / "out").glob("config-backup-*.json"))

            # Replace the live DB with junk, then restore from the backup.
            real_db.unlink()
            _make_db(real_db, ["junk"])
            recovery.restore_backup(saved)
            assert recovery.count_user_rows(real_db) == (2, 0)
            # Previous data preserved aside.
            assert real_db.with_name("momentum.db.pre-restore").exists()

    def test_restore_rejects_non_db(self, tmp_path: Path) -> None:
        junk = tmp_path / "fake.db"
        junk.write_text("not a database")
        with pytest.raises(ValueError, match="Not a valid Momentum database"):
            recovery.restore_backup(junk)
