"""Tests for application service layer."""

from __future__ import annotations

from dataclasses import dataclass

from momentum import db
from momentum.models import (
    AssessmentResultCreate,
    AssessmentType,
    FocusSessionCreate,
    TaskCreate,
)
from momentum.services import (
    AssessmentService,
    PersonalisationService,
    SessionService,
    StatusService,
    TaskService,
)


@dataclass
class FakeTimerService:
    completed: bool = True
    focus_calls: list[tuple[int, int | None]] | None = None
    break_calls: list[int] | None = None

    def __post_init__(self) -> None:
        if self.focus_calls is None:
            self.focus_calls = []
        if self.break_calls is None:
            self.break_calls = []

    def run_focus(self, *, minutes: int, task_id: int | None = None):  # type: ignore[no-untyped-def]
        assert self.focus_calls is not None
        self.focus_calls.append((minutes, task_id))
        return type("Outcome", (), {"completed": self.completed})()

    def run_break(self, *, minutes: int):  # type: ignore[no-untyped-def]
        assert self.break_calls is not None
        self.break_calls.append(minutes)


class TestPersonalisationService:
    def test_returns_defaults_without_bisbas(self) -> None:
        conn = db.get_connection(":memory:")

        profile = PersonalisationService(conn).profile()

        assert profile.focus_minutes == 15
        assert profile.break_minutes == 5
        conn.close()

    def test_loads_profile_from_latest_bisbas(self) -> None:
        conn = db.get_connection(":memory:")
        db.save_assessment(
            conn,
            AssessmentResultCreate(
                assessment_type=AssessmentType.BISBAS,
                score=20,
                max_score=20,
                domain_scores={
                    "Behavioral Inhibition (BIS)": 20,
                    "BAS Drive": 8,
                    "BAS Reward Responsiveness": 8,
                    "BAS Fun Seeking": 8,
                },
            ),
        )

        profile = PersonalisationService(conn).profile()

        assert profile.focus_minutes == 10
        assert profile.nudge_style == "reassuring"
        conn.close()


class TestSessionService:
    def test_persists_focus_when_completed(self) -> None:
        conn = db.get_connection(":memory:")
        timer = FakeTimerService(completed=True)
        task = db.add_task(conn, TaskCreate(title="Write draft"))

        result = SessionService(conn, timer).run_focus(minutes=12, task_id=task.id)

        sessions = db.list_focus_sessions(conn)
        assert result is True
        assert timer.focus_calls == [(12, task.id)]
        assert len(sessions) == 1
        assert sessions[0].duration_minutes == 12
        assert sessions[0].task_id == task.id
        conn.close()

    def test_does_not_persist_focus_when_interrupted(self) -> None:
        conn = db.get_connection(":memory:")
        timer = FakeTimerService(completed=False)

        result = SessionService(conn, timer).run_focus(minutes=8, task_id=None)

        sessions = db.list_focus_sessions(conn)
        assert result is False
        assert timer.focus_calls == [(8, None)]
        assert sessions == []
        conn.close()

    def test_runs_break_without_persistence(self) -> None:
        conn = db.get_connection(":memory:")
        timer = FakeTimerService(completed=True)

        SessionService(conn, timer).run_break(minutes=6)

        assert timer.break_calls == [6]
        assert db.list_focus_sessions(conn) == []
        conn.close()


class TestTaskService:
    def test_add_and_lookup_task(self) -> None:
        conn = db.get_connection(":memory:")
        tasks = TaskService(conn)

        created = tasks.add_task("Draft outline")
        loaded = tasks.get_task(created.id)

        assert loaded is not None
        assert loaded.id == created.id
        assert loaded.title == "Draft outline"
        conn.close()

    def test_add_subtask(self) -> None:
        conn = db.get_connection(":memory:")
        tasks = TaskService(conn)
        parent = tasks.add_task("Parent")

        sub = tasks.add_subtask(parent_id=parent.id, title="Child")

        assert sub.parent_id == parent.id
        assert sub.title == "Child"
        conn.close()

    def test_complete_and_activate(self) -> None:
        conn = db.get_connection(":memory:")
        tasks = TaskService(conn)
        created = tasks.add_task("Ship patch")

        active = tasks.activate_task(created.id)
        done = tasks.complete_task(created.id)

        assert active is not None
        assert done is not None
        assert done.status.value == "done"
        conn.close()

    def test_list_and_reopen(self) -> None:
        conn = db.get_connection(":memory:")
        tasks = TaskService(conn)
        created = tasks.add_task("Reopen me")
        tasks.complete_task(created.id)

        done = tasks.list_tasks(status=created.status.DONE)
        reopened = tasks.reopen_task(created.id)
        pending = tasks.list_tasks(status=created.status.PENDING)

        assert len(done) == 1
        assert reopened is not None
        assert reopened.status.value == "pending"
        assert len(pending) == 1
        conn.close()

    def test_first_active_and_pending(self) -> None:
        conn = db.get_connection(":memory:")
        tasks = TaskService(conn)
        one = tasks.add_task("One")
        two = tasks.add_task("Two")
        tasks.activate_task(two.id)

        first_pending = tasks.first_pending_task()
        first_active = tasks.first_active_task()

        assert first_pending is not None
        assert first_pending.id == one.id
        assert first_active is not None
        assert first_active.id == two.id
        conn.close()


class TestStatusService:
    def test_summary_empty_database(self) -> None:
        conn = db.get_connection(":memory:")

        summary = StatusService(conn).summary()

        assert summary.today.tasks_completed == 0
        assert summary.today.focus_minutes == 0
        assert summary.streak_days == 0
        conn.close()

    def test_summary_includes_completed_tasks_and_focus(self) -> None:
        conn = db.get_connection(":memory:")
        task = db.add_task(conn, TaskCreate(title="Write intro"))
        db.complete_task(conn, task.id)
        db.log_focus_session(
            conn,
            FocusSessionCreate(task_id=task.id, duration_minutes=15),
        )

        summary = StatusService(conn).summary()

        assert summary.today.tasks_completed == 1
        assert summary.today.focus_minutes == 15
        assert summary.week_tasks_completed >= 1
        assert summary.week_focus_minutes >= 15
        conn.close()


class TestAssessmentService:
    def test_save_and_list_results(self) -> None:
        conn = db.get_connection(":memory:")
        assessments = AssessmentService(conn)

        saved = assessments.save_result(
            AssessmentResultCreate(
                assessment_type=AssessmentType.BDEFS,
                score=12,
                max_score=60,
                domain_scores={"Time Management": 3},
            )
        )
        results = assessments.list_results(limit=5)

        assert saved.assessment_type == AssessmentType.BDEFS
        assert len(results) == 1
        assert results[0].id == saved.id
        conn.close()

    def test_history_entries_include_interpretation(self) -> None:
        conn = db.get_connection(":memory:")
        assessments = AssessmentService(conn)
        assessments.save_result(
            AssessmentResultCreate(
                assessment_type=AssessmentType.STROOP,
                score=8,
                max_score=10,
                domain_scores={"avg_time_ms": 1200},
            )
        )

        entries = assessments.history_entries(limit=5)

        assert len(entries) == 1
        assert "STROOP" in entries[0].header
        assert any("Avg response: 1200ms" in line for line in entries[0].lines)
        assert any("inhibitory control" in line.lower() for line in entries[0].lines)
        conn.close()

    def test_browse_rows_include_score_summary(self) -> None:
        conn = db.get_connection(":memory:")
        assessments = AssessmentService(conn)
        assessments.save_result(
            AssessmentResultCreate(
                assessment_type=AssessmentType.BDEFS,
                score=12,
                max_score=60,
                domain_scores={"Time Management": 3},
            )
        )

        rows = assessments.browse_rows(limit=5)

        assert len(rows) == 1
        assert "bdefs" in rows[0]
        assert "score=12/60" in rows[0]
        conn.close()

    def test_count_and_delete_results(self) -> None:
        conn = db.get_connection(":memory:")
        assessments = AssessmentService(conn)
        assessments.save_result(
            AssessmentResultCreate(
                assessment_type=AssessmentType.STROOP,
                score=8,
                max_score=10,
                domain_scores={"avg_time_ms": 1200},
            )
        )
        assessments.save_result(
            AssessmentResultCreate(
                assessment_type=AssessmentType.BISBAS,
                score=10,
                max_score=20,
                domain_scores={"BAS Drive": 3},
            )
        )

        count_before = assessments.count_results()
        deleted = assessments.delete_all_results()
        count_after = assessments.count_results()

        assert count_before == 2
        assert deleted == 2
        assert count_after == 0
        conn.close()
