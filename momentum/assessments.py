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
    "The radar chart compares your two most recent BDEFS assessments. "
    "The blue polygon is your latest result; the grey polygon is your previous one. "
    "Higher values indicate greater difficulty in that area.\n\n"
    "The line chart tracks your total BDEFS score over time. "
    "The dashed trend line shows the overall direction. "
    "A downward trend suggests improvement."
)


# ---------------------------------------------------------------------------
# Domain-specific advice (science-backed, practical)
# ---------------------------------------------------------------------------

# Each domain maps to three tiers of advice: low, moderate, high difficulty.
# Thresholds: score <= 25% of max -> minimal, <= 50% -> mild, <= 75% -> moderate, else high.
# References:
#   Time Management: Barkley (2012) EF and self-regulation; Steel (2007) procrastination meta-analysis
#   Organisation: Risko & Gilbert (2016) cognitive offloading; Kirsh (2000) complementary strategies
#   Self-Restraint: Gollwitzer & Sheeran (2006) implementation intentions meta-analysis
#   Self-Motivation: Mazzucchelli et al. (2010) behavioural activation meta-analysis
#   Emotion Regulation: Gross (2015) emotion regulation handbook; Webb et al. (2012) reappraisal meta-analysis

_DOMAIN_ADVICE: dict[str, dict[str, str]] = {
    "Time Management": {
        "minimal": (
            "Your time management is a relative strength. Keep using whatever "
            "strategies are working for you."
        ),
        "mild": (
            "Try the two-minute rule: if a task takes less than two minutes, do it "
            "immediately rather than scheduling it. Use visual timers (e.g. Momentum's "
            "focus timer) to make the passage of time concrete -- research on time "
            "blindness shows that externalising time improves task completion."
        ),
        "moderate": (
            "Time-box your tasks: assign a fixed block (e.g. 15 minutes) and commit "
            "only to working for that block, not to finishing. Anchor tasks to existing "
            "routines (e.g. 'after morning coffee, I review my list'). Use Momentum's "
            "focus timer consistently -- structured intervals reduce the cognitive load "
            "of deciding when to start and stop."
        ),
        "high": (
            "Use external time cues extensively: alarms, visual countdown timers, and "
            "calendar blocks with reminders. Break every task into steps of 10 minutes "
            "or less. Consider body-doubling (working alongside someone, even virtually) "
            "to create accountability. Barkley's research shows that externalising time "
            "management to the environment is the single most effective strategy for "
            "severe time-management difficulties."
        ),
    },
    "Organisation & Problem-Solving": {
        "minimal": (
            "Your organisational skills are relatively strong. Continue leveraging "
            "whatever systems you have in place."
        ),
        "mild": (
            "Practise 'cognitive offloading': write things down rather than holding them "
            "in working memory. Use Momentum's task list and break-down feature to "
            "decompose projects into concrete next steps. Research shows that externalising "
            "information to the environment significantly reduces cognitive load."
        ),
        "moderate": (
            "Adopt a single, consistent capture system for all tasks and ideas (Momentum "
            "can be that system). Before starting any project, spend two minutes writing "
            "down the first three steps -- this 'pre-planning' bypasses the executive "
            "barrier of figuring out where to begin. Keep your workspace clear: physical "
            "clutter competes for attentional resources."
        ),
        "high": (
            "Use a structured daily review: each morning, identify your single most "
            "important task and break it into steps small enough that the first one takes "
            "under five minutes. Use visual organisers (checklists, colour-coded labels, "
            "physical bins) to reduce the planning demand. Consider working with a coach "
            "or accountability partner who can help you structure complex projects."
        ),
    },
    "Self-Restraint": {
        "minimal": (
            "Your inhibitory control appears relatively strong. Keep using strategies "
            "that help you pause before acting."
        ),
        "mild": (
            "Practise implementation intentions: create specific 'if-then' plans "
            "(e.g. 'if I feel the urge to check my phone, then I will take three breaths "
            "first'). Gollwitzer and Sheeran's meta-analysis found that implementation "
            "intentions have a medium-to-large effect on goal attainment by automating "
            "the response to impulse triggers."
        ),
        "moderate": (
            "Create physical distance from distractions: put your phone in another room "
            "during focus sessions, use website blockers, or change your environment. "
            "Build in a 'pause habit': before any impulsive action, count to five slowly. "
            "This brief delay engages the prefrontal cortex and weakens the automatic "
            "response."
        ),
        "high": (
            "Restructure your environment to make impulsive actions harder and planned "
            "actions easier (this is called 'choice architecture'). Remove or hide "
            "temptations entirely during work periods. Practise brief mindfulness pauses "
            "(even 30 seconds of focused breathing) -- research shows this strengthens "
            "the neural circuits involved in response inhibition. Consider whether "
            "professional assessment for ADHD may be appropriate, as persistent "
            "self-restraint difficulties are a hallmark."
        ),
    },
    "Self-Motivation": {
        "minimal": (
            "Your self-motivation is a relative strength. You seem able to initiate "
            "and sustain effort on tasks reasonably well."
        ),
        "mild": (
            "Pair tasks with small immediate rewards (e.g. a cup of tea after completing "
            "a step). Behavioural activation research shows that action often precedes "
            "motivation, not the other way round -- starting a task for just two minutes "
            "frequently generates the momentum to continue."
        ),
        "moderate": (
            "Schedule specific times for tasks rather than relying on feeling motivated. "
            "Connect each task to a personal value ('I am doing this because...'). "
            "Use Momentum's focus timer to commit to a short burst -- the key insight from "
            "behavioural activation is that reducing the 'activation energy' of starting "
            "is more important than willpower."
        ),
        "high": (
            "Use external accountability: tell someone your plan, use a body-double, or "
            "set up consequences for inaction. Break tasks into steps so small they feel "
            "almost trivial (the 'minimum viable effort' approach). Track your completions "
            "visibly -- Momentum's streak counter leverages the psychological principle "
            "that visible progress sustains effort. If low motivation is pervasive and "
            "persistent, it may reflect anhedonia -- a core depression symptom worth "
            "discussing with a healthcare provider."
        ),
    },
    "Emotion Regulation": {
        "minimal": (
            "Your emotion regulation appears relatively strong. Continue practising "
            "whatever helps you manage frustration and stress."
        ),
        "mild": (
            "Practise cognitive reappraisal: when you notice a strong emotional reaction, "
            "try re-framing the situation ('this is frustrating, but it is temporary and "
            "I can handle it'). Webb et al.'s meta-analysis found reappraisal has a "
            "reliable moderate effect on reducing negative emotions."
        ),
        "moderate": (
            "Use physiological soothing when emotions spike: box breathing (inhale 4s, "
            "hold 4s, exhale 4s, hold 4s) activates the parasympathetic nervous system "
            "within seconds. Label your emotions specifically ('I feel frustrated because "
            "X') -- affect labelling research shows that naming an emotion reduces its "
            "intensity. Be deliberate about self-compassion: treat yourself as you would "
            "a friend in the same situation."
        ),
        "high": (
            "Build a daily micro-practice: five minutes of mindfulness, journaling, or "
            "grounding exercises (5-4-3-2-1 senses technique). When overwhelmed, use the "
            "TIPP technique from DBT: change your Temperature (cold water on face), do "
            "Intense exercise briefly, practise Paced breathing, and use Paired muscle "
            "relaxation. If emotional reactions are frequently disproportionate and causing "
            "distress, consider seeking support from a therapist trained in emotion "
            "regulation strategies (e.g. DBT, CFT, or ACT)."
        ),
    },
}


def domain_advice(domain: str, score: int, max_domain_score: int) -> str:
    """Return practical, science-based advice for a specific BDEFS domain.

    Advice is tiered by severity (minimal / mild / moderate / high) based on
    the percentage of maximum score for that domain.
    """
    advice_map = _DOMAIN_ADVICE.get(domain)
    if advice_map is None:
        return ""
    pct = score / max_domain_score * 100 if max_domain_score else 0
    if pct <= 25:
        return advice_map["minimal"]
    if pct <= 50:
        return advice_map["mild"]
    if pct <= 75:
        return advice_map["moderate"]
    return advice_map["high"]


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
