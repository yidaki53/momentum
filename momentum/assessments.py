"""Self-assessment instruments for executive-dysfunction screening.

Two tests are provided:

* **BDEFS-style self-report** -- a brief questionnaire modelled on the
  Barkley Deficits in Executive Functioning Scale.  It covers five domains
  (time management, organisation & problem-solving, self-restraint,
  self-motivation, and emotion regulation) with three items each, rated on
  a 1-4 Likert scale (Never / Sometimes / Often / Very Often).

* **Stroop colour-word test** -- a timed task that measures the ability to
  inhibit automatic responses.  The participant names the *colour* of
  colour-words that are printed in a mismatched colour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from momentum.models import AssessmentResultCreate, AssessmentType

# ---------------------------------------------------------------------------
# BDEFS-style self-report
# ---------------------------------------------------------------------------

BDEFS_SCALE = {1: "Never", 2: "Sometimes", 3: "Often", 4: "Very Often"}
BDEFS_SCALE_LABELS = [f"{k} - {v}" for k, v in BDEFS_SCALE.items()]

# Each domain maps to a list of question strings.
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


def interpret_bdefs(score: int, max_score: int) -> str:
    """Return a plain-English interpretation of a BDEFS total score."""
    pct = score / max_score * 100
    if pct <= 25:
        return "Minimal difficulties -- your executive functioning appears relatively strong."
    if pct <= 50:
        return "Mild difficulties -- some areas may benefit from targeted strategies."
    if pct <= 75:
        return "Moderate difficulties -- structured routines and support strategies are recommended."
    return "Significant difficulties -- consider seeking professional assessment and support."


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


# ---------------------------------------------------------------------------
# Instruction / guide text (shared across CLI, GUI, and mobile)
# ---------------------------------------------------------------------------

BDEFS_INSTRUCTIONS = (
    "This is a brief executive-function self-assessment based on the "
    "Barkley Deficits in Executive Functioning Scale (BDEFS).\n\n"
    "It covers five domains: Time Management, Organisation & Problem-Solving, "
    "Self-Restraint, Self-Motivation, and Emotion Regulation.\n\n"
    "For each statement, rate how often it applies to you:\n"
    "  1 = Never   2 = Sometimes   3 = Often   4 = Very Often\n\n"
    "There are 15 questions and it takes about 2-3 minutes. "
    "Your results are stored locally and never shared."
)

STROOP_INSTRUCTIONS = (
    "The Stroop test measures inhibitory control -- your ability to override "
    "an automatic response.\n\n"
    "You will see a colour word (e.g. RED) displayed in a different ink colour "
    "(e.g. blue). Your task is to type the INK COLOUR, not the word itself.\n\n"
    "There are 10 trials. Try to answer as quickly and accurately as you can.\n\n"
    "Your accuracy and response time are recorded. Results are stored locally "
    "and never shared."
)

RESULTS_GUIDE = (
    "The radar chart shows your average scores across the five BDEFS domains. "
    "Higher values indicate greater difficulty in that area. "
    "Grey polygons show individual past assessments; the blue polygon is the mean.\n\n"
    "The line chart tracks your total BDEFS score over time. "
    "The dashed trend line shows the overall direction. "
    "A downward trend suggests improvement."
)


def interpret_stroop(correct: int, trials: int, avg_ms: int) -> str:
    """Return a plain-English interpretation of Stroop performance."""
    pct = correct / trials * 100 if trials else 0
    parts: list[str] = []
    if pct >= 90:
        parts.append("Excellent accuracy -- strong inhibitory control.")
    elif pct >= 70:
        parts.append("Good accuracy -- inhibitory control is adequate.")
    elif pct >= 50:
        parts.append(
            "Moderate accuracy -- you may benefit from impulse-management strategies."
        )
    else:
        parts.append("Low accuracy -- inhibitory control may be an area to work on.")

    if avg_ms <= 1000:
        parts.append("Your response time is fast.")
    elif avg_ms <= 2000:
        parts.append("Your response time is average.")
    else:
        parts.append(
            "Your response time is on the slower side; take your time and focus on accuracy."
        )

    return " ".join(parts)
