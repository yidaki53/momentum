"""Dataclass models — single source of truth for all data types.

These were Pydantic ``BaseModel`` subclasses but are now stdlib
``dataclasses`` so the Android (Kivy/p4a) build does not need
``pydantic-core`` (a Rust crate with no Android wheel). Validation that
Pydantic performed via ``Field(...)`` constraints is reproduced in
``__post_init__`` methods that raise ``ValueError``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when *condition* is false."""
    if not condition:
        raise ValueError(message)


class TaskStatus(str, enum.Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"


@dataclass
class Task:
    """A single task or sub-task."""

    id: int
    title: str
    parent_id: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def is_subtask(self) -> bool:
        return self.parent_id is not None


@dataclass
class TaskCreate:
    """Input model for creating a new task."""

    title: str
    parent_id: Optional[int] = None

    def __post_init__(self) -> None:
        _require(1 <= len(self.title) <= 500, "title must be 1..500 characters")


@dataclass
class FocusSession:
    """A completed focus (pomodoro) session."""

    id: int
    duration_minutes: int
    task_id: Optional[int] = None
    completed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        _require(self.duration_minutes > 0, "duration_minutes must be > 0")


@dataclass
class FocusSessionCreate:
    """Input model for logging a focus session."""

    duration_minutes: int
    task_id: Optional[int] = None

    def __post_init__(self) -> None:
        _require(
            0 < self.duration_minutes <= 120,
            "duration_minutes must be > 0 and <= 120",
        )


@dataclass
class DailyLog:
    """Aggregated daily activity summary."""

    date: date
    tasks_completed: int
    focus_minutes: int

    def __post_init__(self) -> None:
        _require(self.tasks_completed >= 0, "tasks_completed must be >= 0")
        _require(self.focus_minutes >= 0, "focus_minutes must be >= 0")


@dataclass
class StatusSummary:
    """Dashboard data for the status command / GUI panel."""

    today: DailyLog
    week_tasks_completed: int = 0
    week_focus_minutes: int = 0
    streak_days: int = 0
    pending_tasks: list[Task] = field(default_factory=list)
    active_tasks: list[Task] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require(self.week_tasks_completed >= 0, "week_tasks_completed must be >= 0")
        _require(self.week_focus_minutes >= 0, "week_focus_minutes must be >= 0")
        _require(self.streak_days >= 0, "streak_days must be >= 0")


@dataclass
class TimerConfig:
    """Configuration for a focus or break timer."""

    minutes: int
    label: str = "Focus"
    task_id: Optional[int] = None
    is_break: bool = False

    def __post_init__(self) -> None:
        _require(0 < self.minutes <= 120, "minutes must be > 0 and <= 120")


@dataclass
class AutostartStatus:
    """Current state of autostart configuration."""

    systemd_enabled: bool = False
    xdg_enabled: bool = False
    service_path: Optional[str] = None
    desktop_entry_path: Optional[str] = None


class AssessmentType(str, enum.Enum):
    """Available self-assessment types."""

    BDEFS = "bdefs"  # Barkley Deficits in Executive Functioning Scale (self-report)
    STROOP = "stroop"  # Stroop Color and Word Test
    BISBAS = "bisbas"  # BIS/BAS motivational style questionnaire


@dataclass
class AssessmentResult:
    """A completed self-assessment result."""

    id: int
    assessment_type: AssessmentType
    score: int
    max_score: int
    domain_scores: dict[str, int] = field(default_factory=dict)
    taken_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        _require(self.score >= 0, "score must be >= 0")
        _require(self.max_score > 0, "max_score must be > 0")


@dataclass
class AssessmentResultCreate:
    """Input model for saving an assessment result."""

    assessment_type: AssessmentType
    score: int
    max_score: int
    domain_scores: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.score >= 0, "score must be >= 0")
        _require(self.max_score > 0, "max_score must be > 0")


class WindowPosition(str, enum.Enum):
    """Where the GUI window appears on launch."""

    CENTRE = "centre"
    TOP_LEFT = "top-left"


class ThemeMode(str, enum.Enum):
    """Visual theme preference."""

    DARK = "dark"
    LIGHT = "light"


class TimerCycleMode(str, enum.Enum):
    """How focus and break sessions are started."""

    MANUAL = "manual"
    AUTO = "auto"


@dataclass
class AppConfig:
    """Application configuration (persisted to ~/.config/momentum/config.json)."""

    db_path: Optional[str] = None  # None = use default (~/.local/share/momentum/)
    window_position: WindowPosition = WindowPosition.CENTRE
    theme_mode: ThemeMode = ThemeMode.DARK
    accessibility_large_text: bool = False
    accessibility_high_contrast: bool = False
    accessibility_reduce_visual_load: bool = False
    timer_cycle_mode: TimerCycleMode = TimerCycleMode.MANUAL
    check_updates_at_startup: bool = True
    last_update_check_unix: int = 0
    show_llm_welcome: bool = True
    llm_model: str = "tinyllama"

    def __post_init__(self) -> None:
        # Coerce string values (from JSON deserialisation) to their enum types,
        # matching the coercion Pydantic performed automatically.
        if not isinstance(self.window_position, WindowPosition):
            self.window_position = WindowPosition(self.window_position)
        if not isinstance(self.theme_mode, ThemeMode):
            self.theme_mode = ThemeMode(self.theme_mode)
        if not isinstance(self.timer_cycle_mode, TimerCycleMode):
            self.timer_cycle_mode = TimerCycleMode(self.timer_cycle_mode)


@dataclass
class ActJournalEntry:
    """A structured ACT-style journaling entry."""

    id: int
    values_focus: str
    challenge_context: str
    thoughts_feelings: str
    defusion_reframe: str
    committed_action: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ActJournalEntryCreate:
    """Input model for creating an ACT-style journaling entry."""

    values_focus: str
    challenge_context: str
    thoughts_feelings: str
    defusion_reframe: str
    committed_action: str

    def __post_init__(self) -> None:
        _require(
            1 <= len(self.values_focus) <= 1500,
            "values_focus must be 1..1500 characters",
        )
        _require(
            1 <= len(self.challenge_context) <= 2500,
            "challenge_context must be 1..2500 characters",
        )
        _require(
            1 <= len(self.thoughts_feelings) <= 2500,
            "thoughts_feelings must be 1..2500 characters",
        )
        _require(
            1 <= len(self.defusion_reframe) <= 2500,
            "defusion_reframe must be 1..2500 characters",
        )
        _require(
            1 <= len(self.committed_action) <= 1500,
            "committed_action must be 1..1500 characters",
        )


@dataclass
class LlmChatMessage:
    """A single message in the AI Coach chat history."""

    id: int
    role: str  # 'user' or 'assistant'
    content: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LlmChatMessageCreate:
    """Input model for saving a chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        _require(
            self.role in ("user", "assistant"),
            "role must be 'user' or 'assistant'",
        )
        _require(
            1 <= len(self.content) <= 10000,
            "content must be 1..10000 characters",
        )
