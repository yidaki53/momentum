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


class TestGui:
    @patch("momentum.gui.run_gui")
    def test_gui_command_invokes_run_gui(self, mock_gui) -> None:
        result = runner.invoke(app, ["gui"])
        assert result.exit_code == 0
        mock_gui.assert_called_once()
