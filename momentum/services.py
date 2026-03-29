"""Application services for CLI and future UI surfaces.

This module deepens common workflows so command handlers stay thin and testable.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from momentum import db
from momentum.models import (
    AssessmentResult,
    AssessmentResultCreate,
    AssessmentType,
    FocusSessionCreate,
    StatusSummary,
    Task,
    TaskCreate,
    TaskStatus,
)
from momentum.timer import TimerService

if TYPE_CHECKING:
    from momentum.assessments import PersonalisationProfile


@dataclass(frozen=True)
class AssessmentHistoryEntry:
    """Plain-text presentation data for an assessment history item."""

    header: str
    lines: list[str]


class PersonalisationService:
    """Resolve profile-dependent defaults and nudge variants."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def profile(self) -> PersonalisationProfile:
        """Return defaults from the latest BIS/BAS result or safe defaults."""
        from momentum.assessments import profile_from_latest_bisbas

        latest = db.list_assessments(
            self._conn,
            assessment_type=AssessmentType.BISBAS,
            limit=1,
        )
        return profile_from_latest_bisbas(latest[0] if latest else None)

    def personalise_nudge(self, message: str) -> str:
        """Apply profile-specific nudge phrasing to a base message."""
        from momentum.assessments import personalised_nudge

        return personalised_nudge(message, self.profile())


class SessionService:
    """Own focus/break session orchestration and persistence."""

    def __init__(
        self, conn: sqlite3.Connection, timer_service: TimerService | None = None
    ) -> None:
        self._conn = conn
        self._timer = timer_service

    def run_focus(self, *, minutes: int, task_id: int | None = None) -> bool:
        """Run a focus timer and persist a completed session."""
        assert self._timer is not None, "timer_service required for run_focus"
        outcome = self._timer.run_focus(minutes=minutes, task_id=task_id)
        if outcome.completed:
            db.log_focus_session(
                self._conn,
                FocusSessionCreate(task_id=task_id, duration_minutes=minutes),
            )
        return outcome.completed

    def run_break(self, *, minutes: int) -> None:
        """Run a break timer."""
        assert self._timer is not None, "timer_service required for run_break"
        self._timer.run_break(minutes=minutes)

    def log_focus(self, *, task_id: int | None, duration_minutes: int) -> None:
        """Persist a completed focus session (for surfaces with their own timer)."""
        db.log_focus_session(
            self._conn,
            FocusSessionCreate(task_id=task_id, duration_minutes=duration_minutes),
        )


class StatusService:
    """Own status summary retrieval for CLI and GUI surfaces."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def summary(self) -> StatusSummary:
        """Return the current dashboard summary."""
        return db.get_status(self._conn)


class TaskService:
    """Own task CRUD and state transitions used by command flows."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add_task(self, title: str) -> Task:
        """Create a top-level task."""
        return db.add_task(self._conn, TaskCreate(title=title))

    def add_subtask(self, *, parent_id: int, title: str) -> Task:
        """Create a subtask under a parent task."""
        return db.add_task(self._conn, TaskCreate(title=title, parent_id=parent_id))

    def get_task(self, task_id: int) -> Task | None:
        """Load a task by ID."""
        return db.get_task(self._conn, task_id)

    def list_tasks(self, *, status: TaskStatus | None = None) -> list[Task]:
        """List tasks with an optional status filter."""
        return db.list_tasks(self._conn, status=status)

    def complete_task(self, task_id: int) -> Task | None:
        """Mark a task as completed."""
        return db.complete_task(self._conn, task_id)

    def reopen_task(self, task_id: int) -> Task | None:
        """Mark a completed task as pending again."""
        return db.uncomplete_task(self._conn, task_id)

    def activate_task(self, task_id: int) -> Task | None:
        """Mark a task as active."""
        return db.set_task_active(self._conn, task_id)

    def first_active_task(self) -> Task | None:
        """Return the first active task if one exists."""
        active = db.list_tasks(self._conn, status=TaskStatus.ACTIVE)
        return active[0] if active else None

    def first_pending_task(self) -> Task | None:
        """Return the first pending task if one exists."""
        pending = db.list_tasks(self._conn, status=TaskStatus.PENDING)
        return pending[0] if pending else None

    def delete_all_tasks(self) -> int:
        """Delete all tasks and focus sessions and return deleted count."""
        return db.delete_all_tasks(self._conn)


class AssessmentService:
    """Own assessment result persistence and retrieval."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_result(self, result_in: AssessmentResultCreate) -> AssessmentResult:
        """Persist a completed assessment result."""
        return db.save_assessment(self._conn, result_in)

    def list_results(
        self,
        *,
        assessment_type: AssessmentType | None = None,
        limit: int = 10,
    ) -> list[AssessmentResult]:
        """List assessment results with optional type filter."""
        return db.list_assessments(
            self._conn,
            assessment_type=assessment_type,
            limit=limit,
        )

    def count_results(self) -> int:
        """Return total number of saved assessment results."""
        return len(self.list_results(limit=9999))

    def delete_all_results(self) -> int:
        """Delete all assessment results and return deleted count."""
        return db.delete_all_assessments(self._conn)

    def history_entries(
        self,
        *,
        assessment_type: AssessmentType | None = None,
        limit: int = 10,
    ) -> list[AssessmentHistoryEntry]:
        """Return formatted history entries for CLI rendering."""
        results = self.list_results(assessment_type=assessment_type, limit=limit)
        return [self._history_entry(result) for result in results]

    def browse_rows(self, *, limit: int = 20) -> list[str]:
        """Return compact assessment rows for database browsing."""
        results = self.list_results(limit=limit)
        return [
            (
                f"  #{result.id}  {result.assessment_type.value}  "
                f"score={result.score}/{result.max_score}  "
                f"({result.taken_at:%Y-%m-%d %H:%M})"
            )
            for result in results
        ]

    def _history_entry(self, result: AssessmentResult) -> AssessmentHistoryEntry:
        """Build a formatted history entry for one saved result."""
        from momentum.assessments import (
            BDEFS_QUESTIONS,
            BISBAS_QUESTIONS,
            bisbas_domain_advice,
            domain_advice,
            interpret_bdefs,
            interpret_bisbas,
            interpret_stroop,
        )

        taken = result.taken_at.strftime("%Y-%m-%d %H:%M")
        lines = [f"  Score: {result.score}/{result.max_score}"]

        if result.assessment_type == AssessmentType.BDEFS:
            for domain, score in result.domain_scores.items():
                max_domain = len(BDEFS_QUESTIONS.get(domain, [])) * 4
                lines.append(f"    {domain}: {score}")
                if max_domain:
                    advice = domain_advice(domain, score, max_domain)
                    if advice:
                        lines.append(f"      [dim italic]{advice}[/dim italic]")
            lines.append(f"  {interpret_bdefs(result.score, result.max_score)}")
        elif result.assessment_type == AssessmentType.STROOP:
            avg_ms = result.domain_scores.get("avg_time_ms", 0)
            lines.append(f"  Avg response: {avg_ms}ms")
            lines.append(
                f"  {interpret_stroop(result.score, result.max_score, avg_ms)}"
            )
        elif result.assessment_type == AssessmentType.BISBAS:
            for domain, score in result.domain_scores.items():
                question_count = len(BISBAS_QUESTIONS.get(domain, []))
                max_domain = question_count * 4 if question_count else 1
                lines.append(f"    {domain}: {score}/{max_domain}")
                advice = bisbas_domain_advice(domain, score, max_domain)
                if advice:
                    lines.append(f"      [dim italic]{advice}[/dim italic]")
            lines.append(
                f"  {interpret_bisbas(result.score, result.max_score, result.domain_scores)}"
            )

        return AssessmentHistoryEntry(
            header=(
                f"[bold]#{result.id}[/bold] "
                f"{result.assessment_type.value.upper()}  ({taken})"
            ),
            lines=lines,
        )
