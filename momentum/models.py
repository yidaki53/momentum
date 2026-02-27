"""Pydantic models — single source of truth for all data types."""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, enum.Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"


class Task(BaseModel):
    """A single task or sub-task."""

    id: int
    title: str
    parent_id: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def is_subtask(self) -> bool:
        return self.parent_id is not None


class TaskCreate(BaseModel):
    """Input model for creating a new task."""

    title: str = Field(min_length=1, max_length=500)
    parent_id: Optional[int] = None


class FocusSession(BaseModel):
    """A completed focus (pomodoro) session."""

    id: int
    task_id: Optional[int] = None
    duration_minutes: int = Field(gt=0)
    completed_at: datetime = Field(default_factory=datetime.now)


class FocusSessionCreate(BaseModel):
    """Input model for logging a focus session."""

    task_id: Optional[int] = None
    duration_minutes: int = Field(gt=0, le=120)


class DailyLog(BaseModel):
    """Aggregated daily activity summary."""

    date: date
    tasks_completed: int = Field(ge=0)
    focus_minutes: int = Field(ge=0)


class StatusSummary(BaseModel):
    """Dashboard data for the status command / GUI panel."""

    today: DailyLog
    week_tasks_completed: int = Field(default=0, ge=0)
    week_focus_minutes: int = Field(default=0, ge=0)
    streak_days: int = Field(default=0, ge=0)
    pending_tasks: list[Task] = Field(default_factory=list)
    active_tasks: list[Task] = Field(default_factory=list)


class TimerConfig(BaseModel):
    """Configuration for a focus or break timer."""

    minutes: int = Field(gt=0, le=120)
    label: str = "Focus"
    task_id: Optional[int] = None
    is_break: bool = False


class AutostartStatus(BaseModel):
    """Current state of autostart configuration."""

    systemd_enabled: bool = False
    xdg_enabled: bool = False
    service_path: Optional[str] = None
    desktop_entry_path: Optional[str] = None


class AssessmentType(str, enum.Enum):
    """Available self-assessment types."""

    BDEFS = "bdefs"  # Barkley Deficits in Executive Functioning Scale (self-report)
    STROOP = "stroop"  # Stroop Color and Word Test


class AssessmentResult(BaseModel):
    """A completed self-assessment result."""

    id: int
    assessment_type: AssessmentType
    score: int = Field(ge=0)
    max_score: int = Field(gt=0)
    domain_scores: dict[str, int] = Field(default_factory=dict)
    taken_at: datetime = Field(default_factory=datetime.now)


class AssessmentResultCreate(BaseModel):
    """Input model for saving an assessment result."""

    assessment_type: AssessmentType
    score: int = Field(ge=0)
    max_score: int = Field(gt=0)
    domain_scores: dict[str, int] = Field(default_factory=dict)


class WindowPosition(str, enum.Enum):
    """Where the GUI window appears on launch."""

    CENTRE = "centre"
    TOP_LEFT = "top-left"


class AppConfig(BaseModel):
    """Application configuration (persisted to ~/.config/momentum/config.json)."""

    db_path: Optional[str] = None  # None = use default (~/.local/share/momentum/)
    window_position: WindowPosition = WindowPosition.CENTRE
