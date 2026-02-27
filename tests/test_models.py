"""Tests for Pydantic models."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from momentum.models import (
    AutostartStatus,
    DailyLog,
    FocusSession,
    FocusSessionCreate,
    StatusSummary,
    Task,
    TaskCreate,
    TaskStatus,
    TimerConfig,
)


class TestTaskStatus:
    def test_values(self) -> None:
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.ACTIVE.value == "active"
        assert TaskStatus.DONE.value == "done"

    def test_from_string(self) -> None:
        assert TaskStatus("pending") is TaskStatus.PENDING


class TestTask:
    def test_create_minimal(self) -> None:
        task = Task(id=1, title="Test task")
        assert task.status == TaskStatus.PENDING
        assert task.parent_id is None
        assert task.completed_at is None

    def test_is_subtask(self) -> None:
        parent = Task(id=1, title="Parent")
        child = Task(id=2, title="Child", parent_id=1)
        assert not parent.is_subtask
        assert child.is_subtask


class TestTaskCreate:
    def test_valid(self) -> None:
        tc = TaskCreate(title="Do something")
        assert tc.title == "Do something"
        assert tc.parent_id is None

    def test_empty_title_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskCreate(title="")

    def test_long_title_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskCreate(title="x" * 501)

    def test_with_parent(self) -> None:
        tc = TaskCreate(title="Sub-step", parent_id=5)
        assert tc.parent_id == 5


class TestFocusSessionCreate:
    def test_valid(self) -> None:
        fs = FocusSessionCreate(duration_minutes=15)
        assert fs.task_id is None

    def test_zero_minutes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FocusSessionCreate(duration_minutes=0)

    def test_over_120_minutes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FocusSessionCreate(duration_minutes=121)


class TestTimerConfig:
    def test_defaults(self) -> None:
        tc = TimerConfig(minutes=10)
        assert tc.label == "Focus"
        assert tc.is_break is False

    def test_break_config(self) -> None:
        tc = TimerConfig(minutes=5, label="Break", is_break=True)
        assert tc.is_break is True


class TestDailyLog:
    def test_create(self) -> None:
        dl = DailyLog(date=date.today(), tasks_completed=3, focus_minutes=45)
        assert dl.tasks_completed == 3

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DailyLog(date=date.today(), tasks_completed=-1, focus_minutes=0)


class TestStatusSummary:
    def test_defaults(self) -> None:
        summary = StatusSummary(
            today=DailyLog(date=date.today(), tasks_completed=0, focus_minutes=0),
        )
        assert summary.streak_days == 0
        assert summary.pending_tasks == []


class TestAutostartStatus:
    def test_defaults(self) -> None:
        status = AutostartStatus()
        assert not status.systemd_enabled
        assert not status.xdg_enabled
        assert status.service_path is None
