"""Pure scoring functions for assessments (no interpretation or personalization)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from momentum.models import AssessmentResultCreate, AssessmentType

# Try to import Cython-compiled versions; fall back to pure Python
_CYTHON_AVAILABLE = False
score_bdefs_cy: Callable[[dict[str, list[int]]], AssessmentResultCreate] | None = None
score_bisbas_cy: Callable[[dict[str, list[int]]], AssessmentResultCreate] | None = None
try:
    from momentum._assessments_cy import score_bdefs_cy, score_bisbas_cy

    _CYTHON_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# BDEFS-style self-report
# ---------------------------------------------------------------------------

BDEFS_SCALE = {1: "Never", 2: "Sometimes", 3: "Often", 4: "Very Often"}
BDEFS_SCALE_LABELS = [f"{k} - {v}" for k, v in BDEFS_SCALE.items()]

BDEFS_QUESTIONS: dict[str, list[str]] = {
    "Time Management": [
        "I have difficulty estimating how long a task will take.",
        "I procrastinate or put off doing things until the last minute.",
        "I have trouble completing tasks on time.",
    ],
    "Organisation & Problem-Solving": [
        "I find it hard to organise my thoughts before starting a task.",
        "I struggle to break large projects into manageable steps.",
        "I have trouble keeping my workspace or living area tidy.",
    ],
    "Self-Restraint": [
        "I act on impulse without thinking about the consequences.",
        "I have difficulty waiting my turn or being patient.",
        "I interrupt others or blurt things out before they finish speaking.",
    ],
    "Self-Motivation": [
        "I lack the drive to start tasks even when I know they are important.",
        "I find it hard to sustain effort on tasks that are not immediately rewarding.",
        "I need external pressure (deadlines, others reminding me) to get things done.",
    ],
    "Emotion Regulation": [
        "I become frustrated or upset more easily than others.",
        "I have trouble calming myself down when I am angry or stressed.",
        "My emotional reactions feel out of proportion to the situation.",
    ],
}

BDEFS_MIN_PER_ITEM = 1
BDEFS_MAX_PER_ITEM = 4


def bdefs_max_score() -> int:
    """Maximum possible BDEFS total score."""
    n_items = sum(len(qs) for qs in BDEFS_QUESTIONS.values())
    return n_items * BDEFS_MAX_PER_ITEM


def score_bdefs(answers: dict[str, list[int]]) -> AssessmentResultCreate:
    """Score a completed BDEFS questionnaire.

    Parameters
    ----------
    answers:
        Mapping of domain name -> list of integer responses (1-4 each).

    Returns
    -------
    AssessmentResultCreate ready to be saved to the database.
    """
    # Use Cython-compiled version if available
    if score_bdefs_cy is not None:
        return score_bdefs_cy(answers)

    # Pure Python fallback
    domain_scores: dict[str, int] = {}
    total = 0
    for domain, scores in answers.items():
        ds = sum(scores)
        domain_scores[domain] = ds
        total += ds

    return AssessmentResultCreate(
        assessment_type=AssessmentType.BDEFS,
        score=total,
        max_score=bdefs_max_score(),
        domain_scores=domain_scores,
    )


# ---------------------------------------------------------------------------
# BIS/BAS motivational profile
# ---------------------------------------------------------------------------

BISBAS_SCALE = {
    1: "Very false for me",
    2: "Somewhat false for me",
    3: "Somewhat true for me",
    4: "Very true for me",
}
BISBAS_SCALE_LABELS = [f"{k} - {v}" for k, v in BISBAS_SCALE.items()]

BISBAS_QUESTIONS: dict[str, list[str]] = {
    "Behavioral Inhibition (BIS)": [
        "I worry about making mistakes when I start something important.",
        "Criticism or disapproval can easily make me hold back.",
        "I often hesitate because I am concerned about negative outcomes.",
        "Uncertain situations tend to make me cautious and avoidant.",
        "When I feel pressured, I find it hard to initiate action.",
    ],
    "BAS Drive": [
        "When I decide on a goal, I keep working until I reach it.",
        "I stay focused on goals even when progress is slow.",
        "I usually push through obstacles when something matters to me.",
        "I find it easy to re-engage with a goal after interruptions.",
        "I actively pursue long-term goals even when effort is required.",
    ],
    "BAS Reward Responsiveness": [
        "Completing even a small step gives me a noticeable boost.",
        "I feel energised by seeing clear signs of progress.",
        "I respond strongly to praise or positive feedback.",
        "Small rewards help me keep momentum on difficult tasks.",
        "The possibility of success strongly increases my motivation.",
    ],
    "BAS Fun Seeking": [
        "I get bored quickly if tasks feel repetitive.",
        "I prefer starting tasks that feel interesting right away.",
        "Novelty helps me engage more than routine.",
        "I am more likely to begin tasks that feel stimulating.",
        "I often switch focus when something feels more engaging.",
    ],
}

BISBAS_MIN_PER_ITEM = 1
BISBAS_MAX_PER_ITEM = 4


def bisbas_max_score() -> int:
    """Maximum possible BIS/BAS total score."""
    n_items = sum(len(qs) for qs in BISBAS_QUESTIONS.values())
    return n_items * BISBAS_MAX_PER_ITEM


def score_bisbas(answers: dict[str, list[int]]) -> AssessmentResultCreate:
    """Score a completed BIS/BAS questionnaire."""
    # Use Cython-compiled version if available
    if score_bisbas_cy is not None:
        return score_bisbas_cy(answers)

    # Pure Python fallback
    domain_scores: dict[str, int] = {}
    total = 0
    for domain, scores in answers.items():
        ds = sum(scores)
        domain_scores[domain] = ds
        total += ds
    return AssessmentResultCreate(
        assessment_type=AssessmentType.BISBAS,
        score=total,
        max_score=bisbas_max_score(),
        domain_scores=domain_scores,
    )


# ---------------------------------------------------------------------------
# Stroop colour-word test
# ---------------------------------------------------------------------------

STROOP_COLOURS = ["red", "green", "blue", "yellow"]
STROOP_DEFAULT_TRIALS = 10


@dataclass
class StroopTrial:
    """A single Stroop trial: a colour word displayed in a different ink colour."""

    word: str
    ink_colour: str  # the *correct* answer

    def __post_init__(self) -> None:
        assert self.word != self.ink_colour


@dataclass
class StroopResult:
    """Aggregate result of a Stroop test session."""

    trials: int
    correct: int
    total_time_s: float
    per_trial: list[tuple[bool, float]] = field(default_factory=list)

    @property
    def accuracy_pct(self) -> float:
        return (self.correct / self.trials * 100) if self.trials else 0.0

    @property
    def avg_time_s(self) -> float:
        return (self.total_time_s / self.trials) if self.trials else 0.0


def generate_stroop_trials(n: int = STROOP_DEFAULT_TRIALS) -> list[StroopTrial]:
    """Generate *n* Stroop trials with mismatched word/colour pairs."""
    trials: list[StroopTrial] = []
    for _ in range(n):
        word = random.choice(STROOP_COLOURS)
        ink = random.choice([c for c in STROOP_COLOURS if c != word])
        trials.append(StroopTrial(word=word, ink_colour=ink))
    return trials


def score_stroop(result: StroopResult) -> AssessmentResultCreate:
    """Convert a StroopResult into an AssessmentResultCreate."""
    # Score = correct answers; max = total trials.
    return AssessmentResultCreate(
        assessment_type=AssessmentType.STROOP,
        score=result.correct,
        max_score=result.trials,
        domain_scores={
            "correct": result.correct,
            "trials": result.trials,
            "avg_time_ms": int(result.avg_time_s * 1000),
        },
    )
