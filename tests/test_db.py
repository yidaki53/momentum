"""Tests for the database layer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from momentum import db
from momentum.models import (
    AssessmentResultCreate,
    AssessmentType,
    FocusSessionCreate,
    TaskCreate,
    TaskStatus,
)


@pytest.fixture()
def conn(tmp_path: Path):
    """Provide a fresh in-memory-like database for each test."""
    db_path = tmp_path / "test.db"
    connection = db.get_connection(db_path=db_path)
    yield connection
    connection.close()


class TestTasks:
    def test_add_and_get(self, conn) -> None:
        task_in = TaskCreate(title="Write tests")
        task = db.add_task(conn, task_in)
        assert task.id == 1
        assert task.title == "Write tests"
        assert task.status == TaskStatus.PENDING

        fetched = db.get_task(conn, task.id)
        assert fetched is not None
        assert fetched.title == task.title

    def test_get_nonexistent(self, conn) -> None:
        assert db.get_task(conn, 9999) is None

    def test_list_tasks_filter_by_status(self, conn) -> None:
        db.add_task(conn, TaskCreate(title="Task A"))
        t2 = db.add_task(conn, TaskCreate(title="Task B"))
        db.set_task_active(conn, t2.id)

        pending = db.list_tasks(conn, status=TaskStatus.PENDING)
        active = db.list_tasks(conn, status=TaskStatus.ACTIVE)
        assert len(pending) == 1
        assert len(active) == 1
        assert active[0].title == "Task B"

    def test_complete_task(self, conn) -> None:
        task = db.add_task(conn, TaskCreate(title="Finish"))
        completed = db.complete_task(conn, task.id)
        assert completed is not None
        assert completed.status == TaskStatus.DONE
        assert completed.completed_at is not None

    def test_complete_updates_daily_log(self, conn) -> None:
        task = db.add_task(conn, TaskCreate(title="Count me"))
        db.complete_task(conn, task.id)
        log = db.get_daily_log(conn, date.today())
        assert log.tasks_completed == 1

    def test_subtasks(self, conn) -> None:
        parent = db.add_task(conn, TaskCreate(title="Big task"))
        sub = db.add_task(conn, TaskCreate(title="Small step", parent_id=parent.id))
        subs = db.get_subtasks(conn, parent.id)
        assert len(subs) == 1
        assert subs[0].id == sub.id


class TestFocusSessions:
    def test_log_session(self, conn) -> None:
        session_in = FocusSessionCreate(duration_minutes=15)
        session = db.log_focus_session(conn, session_in)
        assert session.id == 1
        assert session.duration_minutes == 15

    def test_log_session_updates_daily_log(self, conn) -> None:
        db.log_focus_session(conn, FocusSessionCreate(duration_minutes=10))
        db.log_focus_session(conn, FocusSessionCreate(duration_minutes=15))
        log = db.get_daily_log(conn, date.today())
        assert log.focus_minutes == 25


class TestStatus:
    def test_empty_status(self, conn) -> None:
        summary = db.get_status(conn)
        assert summary.today.tasks_completed == 0
        assert summary.streak_days == 0
        assert summary.pending_tasks == []

    def test_status_with_activity(self, conn) -> None:
        task = db.add_task(conn, TaskCreate(title="Do thing"))
        db.complete_task(conn, task.id)
        db.log_focus_session(conn, FocusSessionCreate(duration_minutes=15))

        summary = db.get_status(conn)
        assert summary.today.tasks_completed == 1
        assert summary.today.focus_minutes == 15
        assert summary.streak_days == 1

    def test_daily_log_default(self, conn) -> None:
        log = db.get_daily_log(conn, date(2020, 1, 1))
        assert log.tasks_completed == 0
        assert log.focus_minutes == 0


class TestAssessments:
    def test_save_and_retrieve_bdefs(self, conn) -> None:
        create = AssessmentResultCreate(
            assessment_type=AssessmentType.BDEFS,
            score=30,
            max_score=60,
            domain_scores={"Time Management": 6, "Self-Restraint": 8},
        )
        saved = db.save_assessment(conn, create)
        assert saved.id == 1
        assert saved.score == 30
        assert saved.max_score == 60
        assert saved.domain_scores["Time Management"] == 6

    def test_save_and_retrieve_stroop(self, conn) -> None:
        create = AssessmentResultCreate(
            assessment_type=AssessmentType.STROOP,
            score=8,
            max_score=10,
            domain_scores={"correct": 8, "trials": 10, "avg_time_ms": 1200},
        )
        saved = db.save_assessment(conn, create)
        assert saved.assessment_type == AssessmentType.STROOP

    def test_list_assessments_all(self, conn) -> None:
        for i in range(3):
            db.save_assessment(
                conn,
                AssessmentResultCreate(
                    assessment_type=AssessmentType.BDEFS,
                    score=20 + i,
                    max_score=60,
                ),
            )
        results = db.list_assessments(conn)
        assert len(results) == 3
        # Most recent first
        assert results[0].score >= results[-1].score

    def test_list_assessments_filter_type(self, conn) -> None:
        db.save_assessment(
            conn,
            AssessmentResultCreate(
                assessment_type=AssessmentType.BDEFS, score=30, max_score=60
            ),
        )
        db.save_assessment(
            conn,
            AssessmentResultCreate(
                assessment_type=AssessmentType.STROOP, score=8, max_score=10
            ),
        )
        bdefs = db.list_assessments(conn, assessment_type=AssessmentType.BDEFS)
        assert len(bdefs) == 1
        assert bdefs[0].assessment_type == AssessmentType.BDEFS

    def test_list_assessments_limit(self, conn) -> None:
        for i in range(5):
            db.save_assessment(
                conn,
                AssessmentResultCreate(
                    assessment_type=AssessmentType.BDEFS,
                    score=20 + i,
                    max_score=60,
                ),
            )
        results = db.list_assessments(conn, limit=2)
        assert len(results) == 2
