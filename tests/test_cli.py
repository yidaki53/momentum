"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from momentum import db
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
