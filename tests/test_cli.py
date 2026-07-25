"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from momentum.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path: Path):
    """Redirect all CLI tests to a temporary database."""
    db_path = tmp_path / "test.db"
    with patch("momentum.db._get_db_path", return_value=db_path):
        yield


class TestAdd:
    def test_add_task(self) -> None:
        result = runner.invoke(app, ["add", "Write introduction"])
        assert result.exit_code == 0
        assert "Added task #1" in result.output

    def test_add_multiple(self) -> None:
        runner.invoke(app, ["add", "Task one"])
        result = runner.invoke(app, ["add", "Task two"])
        assert "Added task #2" in result.output


class TestDone:
    def test_complete_existing(self) -> None:
        runner.invoke(app, ["add", "Finish"])
        result = runner.invoke(app, ["done", "1"])
        assert result.exit_code == 0
        assert "Completed" in result.output

    def test_complete_nonexistent(self) -> None:
        result = runner.invoke(app, ["done", "999"])
        assert result.exit_code == 1


class TestList:
    def test_list_empty(self) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_list_with_tasks(self) -> None:
        runner.invoke(app, ["add", "Alpha"])
        runner.invoke(app, ["add", "Beta"])
        result = runner.invoke(app, ["list"])
        assert "Alpha" in result.output
        assert "Beta" in result.output


class TestStatus:
    def test_status_empty(self) -> None:
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Status" in result.output


class TestNudge:
    def test_nudge(self) -> None:
        result = runner.invoke(app, ["nudge"])
        assert result.exit_code == 0
        # Should print some non-empty message
        assert len(result.output.strip()) > 0


class TestConfig:
    def test_show_default(self) -> None:
        result = runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "Database" in result.output

    def test_no_flags(self) -> None:
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "--sync" in result.output or "--db-path" in result.output

    def test_set_db_path(self, tmp_path: Path) -> None:
        db = tmp_path / "custom.db"
        result = runner.invoke(app, ["config", "--db-path", str(db)])
        assert result.exit_code == 0
        assert "Database path set" in result.output

    def test_reset(self) -> None:
        result = runner.invoke(app, ["config", "--reset"])
        assert result.exit_code == 0
        assert "Reset" in result.output

    def test_sync_nonexistent_provider(self) -> None:
        result = runner.invoke(app, ["config", "--sync", "icloud"])
        assert result.exit_code == 1


class TestTestResults:
    def test_no_results(self) -> None:
        result = runner.invoke(app, ["test-results"])
        assert result.exit_code == 0
        assert "No assessment" in result.output

    def test_invalid_type(self) -> None:
        result = runner.invoke(app, ["test-results", "--type", "invalid"])
        assert result.exit_code == 1


class TestAutostart:
    @patch("momentum.autostart.enable_autostart")
    def test_enable(self, mock_enable) -> None:
        from momentum.models import AutostartStatus

        mock_enable.return_value = AutostartStatus(
            systemd_enabled=True, xdg_enabled=True
        )
        result = runner.invoke(app, ["autostart", "--enable"])
        assert result.exit_code == 0
        assert "enabled" in result.output.lower()

    @patch("momentum.autostart.disable_autostart")
    def test_disable(self, mock_disable) -> None:
        result = runner.invoke(app, ["autostart", "--disable"])
        assert result.exit_code == 0
        assert "disabled" in result.output.lower()

    @patch("momentum.autostart.get_autostart_status")
    def test_status(self, mock_status) -> None:
        from momentum.models import AutostartStatus

        mock_status.return_value = AutostartStatus()
        result = runner.invoke(app, ["autostart", "--status"])
        assert result.exit_code == 0

    def test_no_flags(self) -> None:
        result = runner.invoke(app, ["autostart"])
        assert result.exit_code == 0
        assert "--enable" in result.output or "Use" in result.output


class TestBreakDown:
    def test_nonexistent_task(self) -> None:
        result = runner.invoke(app, ["break-down", "999"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_no_substeps_added(self) -> None:
        runner.invoke(app, ["add", "Parent task"])
        result = runner.invoke(app, ["break-down", "1"], input="\n")
        assert result.exit_code == 0
        assert "No sub-steps" in result.output

    def test_add_substeps(self) -> None:
        runner.invoke(app, ["add", "Big task"])
        result = runner.invoke(app, ["break-down", "1"], input="Step A\nStep B\n\n")
        assert result.exit_code == 0
        assert "2 sub-steps" in result.output


class TestListAll:
    def test_list_all_includes_done(self) -> None:
        runner.invoke(app, ["add", "Task one"])
        runner.invoke(app, ["done", "1"])
        result = runner.invoke(app, ["list", "--all"])
        assert result.exit_code == 0
        assert "Task one" in result.output

    def test_list_with_subtasks(self) -> None:
        runner.invoke(app, ["add", "Parent"])
        runner.invoke(app, ["break-down", "1"], input="Child A\nChild B\n\n")
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Parent" in result.output
        assert "Child A" in result.output


class TestFocus:
    @patch("momentum.cli._timer_service")
    def test_focus_no_task(self, mock_service) -> None:
        mock_service.return_value.run_focus.return_value.completed = False
        result = runner.invoke(app, ["focus"])
        assert result.exit_code == 0
        assert "Starting" in result.output

    @patch("momentum.cli._timer_service")
    def test_focus_with_task(self, mock_service) -> None:
        mock_service.return_value.run_focus.return_value.completed = False
        runner.invoke(app, ["add", "Focus target"])
        result = runner.invoke(app, ["focus", "--task", "1"])
        assert result.exit_code == 0
        assert "Focusing on" in result.output

    def test_focus_nonexistent_task(self) -> None:
        result = runner.invoke(app, ["focus", "--task", "999"])
        assert result.exit_code == 1

    @patch("momentum.cli._timer_service")
    def test_focus_completed_accept_break(self, mock_service) -> None:
        service = mock_service.return_value
        service.run_focus.return_value.completed = True
        result = runner.invoke(app, ["focus"], input="y\n")
        assert result.exit_code == 0
        assert "logged" in result.output.lower()
        service.run_break.assert_called_once()

    @patch("momentum.cli._timer_service")
    def test_focus_completed_decline_break(self, mock_service) -> None:
        service = mock_service.return_value
        service.run_focus.return_value.completed = True
        result = runner.invoke(app, ["focus"], input="n\n")
        assert result.exit_code == 0
        assert "logged" in result.output.lower()
        service.run_break.assert_not_called()


class TestTakeBreak:
    @patch("momentum.cli._timer_service")
    def test_take_break(self, mock_service) -> None:
        result = runner.invoke(app, ["take-break"])
        assert result.exit_code == 0
        assert "Break time" in result.output
        mock_service.return_value.run_break.assert_called_once_with(minutes=5)

    @patch("momentum.cli._timer_service")
    def test_take_break_custom_minutes(self, mock_service) -> None:
        result = runner.invoke(app, ["take-break", "--minutes", "10"])
        assert result.exit_code == 0
        assert "10" in result.output
        mock_service.return_value.run_break.assert_called_once_with(minutes=10)


class TestConfigExtra:
    @patch("momentum.config.set_cloud_sync")
    def test_sync_success(self, mock_sync) -> None:
        from momentum.models import AppConfig

        mock_sync.return_value = AppConfig(db_path="/tmp/synced.db")
        result = runner.invoke(app, ["config", "--sync", "onedrive"])
        assert result.exit_code == 0
        assert "sync" in result.output.lower()

    def test_show_with_custom_path(self, tmp_path: Path) -> None:
        db = tmp_path / "custom.db"
        runner.invoke(app, ["config", "--db-path", str(db)])
        result = runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "custom.db" in result.output


class TestTestResultsWithData:
    def _save_bdefs(self) -> None:
        from momentum import db as _db
        from momentum.assessments import score_bdefs

        conn = _db.get_connection()
        answers = {
            "Time Management": [2, 2, 2],
            "Organisation & Problem-Solving": [2, 2, 2],
            "Self-Restraint": [1, 1, 1],
            "Self-Motivation": [3, 3, 3],
            "Emotion Regulation": [2, 2, 2],
        }
        create = score_bdefs(answers)
        _db.save_assessment(conn, create)
        conn.close()

    def _save_stroop(self) -> None:
        from momentum import db as _db
        from momentum.assessments import StroopResult, score_stroop

        conn = _db.get_connection()
        result = StroopResult(
            trials=10,
            correct=8,
            total_time_s=12.0,
            per_trial=[(True, 1.0)] * 8 + [(False, 2.0)] * 2,
        )
        create = score_stroop(result)
        _db.save_assessment(conn, create)
        conn.close()

    def test_results_with_bdefs(self) -> None:
        self._save_bdefs()
        result = runner.invoke(app, ["test-results"])
        assert result.exit_code == 0
        assert "BDEFS" in result.output
        assert "Score" in result.output
        assert "Time Management" in result.output

    def test_results_with_stroop(self) -> None:
        self._save_stroop()
        result = runner.invoke(app, ["test-results"])
        assert result.exit_code == 0
        assert "STROOP" in result.output
        assert "Avg response" in result.output

    def test_results_filter_bdefs(self) -> None:
        self._save_bdefs()
        self._save_stroop()
        result = runner.invoke(app, ["test-results", "--type", "bdefs"])
        assert result.exit_code == 0
        assert "BDEFS" in result.output
        assert "STROOP" not in result.output

    def test_results_filter_stroop(self) -> None:
        self._save_bdefs()
        self._save_stroop()
        result = runner.invoke(app, ["test-results", "--type", "stroop"])
        assert result.exit_code == 0
        assert "STROOP" in result.output

    def test_results_with_limit(self) -> None:
        self._save_bdefs()
        self._save_bdefs()
        result = runner.invoke(app, ["test-results", "--limit", "1"])
        assert result.exit_code == 0
        assert "BDEFS" in result.output


class TestAutostartExtra:
    @patch("momentum.autostart.enable_autostart")
    def test_enable_fails(self, mock_enable) -> None:
        from momentum.models import AutostartStatus

        mock_enable.return_value = AutostartStatus(
            systemd_enabled=False, xdg_enabled=False
        )
        result = runner.invoke(app, ["autostart", "--enable"])
        assert result.exit_code == 0
        assert "could not" in result.output.lower()

    @patch("momentum.autostart.get_autostart_status")
    def test_status_with_paths(self, mock_status) -> None:
        from momentum.models import AutostartStatus

        mock_status.return_value = AutostartStatus(
            systemd_enabled=True,
            xdg_enabled=True,
            service_path="/etc/systemd/user/momentum.service",
            desktop_entry_path="/etc/xdg/autostart/momentum.desktop",
        )
        result = runner.invoke(app, ["autostart", "--status"])
        assert result.exit_code == 0
        assert "enabled" in result.output
        assert "momentum.service" in result.output
        assert "momentum.desktop" in result.output


class TestStart:
    @patch("momentum.cli._timer_service")
    def test_start_with_active_task_continue(self, mock_service) -> None:
        mock_service.return_value.run_focus.return_value.completed = False
        runner.invoke(app, ["add", "Active thing"])
        # Manually set task active via focus
        runner.invoke(app, ["focus", "--task", "1"])
        result = runner.invoke(app, ["start"], input="y\n")
        assert result.exit_code == 0

    @patch("momentum.cli._timer_service")
    def test_start_with_pending_task_decline(self, mock_service) -> None:
        mock_service.return_value.run_focus.return_value.completed = False
        runner.invoke(app, ["add", "Pending thing"])
        result = runner.invoke(app, ["start"], input="n\n")
        assert result.exit_code == 0

    def test_start_no_tasks_empty_input(self) -> None:
        result = runner.invoke(app, ["start"], input="\n\n")
        assert result.exit_code == 0

    @patch("momentum.cli._timer_service")
    def test_start_no_tasks_add_one(self, mock_service) -> None:
        mock_service.return_value.run_focus.return_value.completed = False
        result = runner.invoke(app, ["start"], input="Do laundry\ny\n")
        assert result.exit_code == 0
        assert "Added" in result.output


class TestGui:
    @patch("momentum.cli._run_gui")
    def test_gui_command_invokes_run_gui(self, mock_gui) -> None:
        result = runner.invoke(app, ["gui"])
        assert result.exit_code == 0
        mock_gui.assert_called_once()


class TestUpdateCommands:
    @pytest.fixture(autouse=True)
    def _tmp_config(self, tmp_path: Path, monkeypatch) -> None:
        """Redirect config reads/writes to tmp_path so tests never touch the
        real ~/.config/momentum/config.json."""
        cfg_dir = tmp_path / "config"
        monkeypatch.setattr("momentum.config._CONFIG_DIR", cfg_dir)
        monkeypatch.setattr("momentum.config._CONFIG_FILE", cfg_dir / "config.json")

    def _mock_release(self, monkeypatch, version: str = "0.4.0") -> None:
        from momentum.ui import update_check

        monkeypatch.setattr(
            update_check,
            "fetch_latest_release",
            lambda timeout=5.0: update_check.ReleaseInfo(
                version=version, url="https://example.com/release"
            ),
        )

    def test_check_updates_up_to_date(self, monkeypatch) -> None:
        self._mock_release(monkeypatch, "0.4.0")
        result = runner.invoke(app, ["check-updates"])
        assert result.exit_code == 0
        assert "latest version" in result.output

    def test_check_updates_available(self, monkeypatch) -> None:
        self._mock_release(monkeypatch, "0.5.0")
        result = runner.invoke(app, ["check-updates"])
        assert result.exit_code == 0
        assert "0.5.0" in result.output
        assert "momentum update" in result.output

    def test_check_updates_network_error(self, monkeypatch) -> None:
        from momentum.ui import update_check

        def boom(timeout: float = 5.0):
            raise OSError("network down")

        monkeypatch.setattr(update_check, "fetch_latest_release", boom)
        result = runner.invoke(app, ["check-updates"])
        assert result.exit_code == 1

    def test_update_no_new_version(self, monkeypatch) -> None:
        self._mock_release(monkeypatch, "0.4.0")
        result = runner.invoke(app, ["update"])
        assert result.exit_code == 0
        assert "latest version" in result.output

    def test_update_falls_back_when_not_frozen(self, monkeypatch) -> None:
        # Dev/poetry runs are not frozen, so self-update must fall back to notify.
        self._mock_release(monkeypatch, "0.5.0")
        result = runner.invoke(app, ["update"])
        assert result.exit_code == 0
        assert "Automatic update is not available" in result.output
        assert "https://example.com/release" in result.output

    def test_startup_notice_when_update_available(self, monkeypatch) -> None:
        from momentum import config as cfg
        from momentum.ui import update_check

        # Opt-in check enabled, throttle forced elapsed (last_update_check_unix=0).
        cfg.save_config(
            cfg.AppConfig(check_updates_at_startup=True, last_update_check_unix=0)
        )
        monkeypatch.setattr(
            update_check,
            "fetch_latest_release",
            lambda timeout=5.0: update_check.ReleaseInfo(
                version="0.5.0", url="https://example.com/release"
            ),
        )
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "0.5.0" in result.output
        assert "momentum update" in result.output
