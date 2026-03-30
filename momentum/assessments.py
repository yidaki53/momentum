"""Self-assessment instruments for executive-dysfunction screening.

Three tests are provided:

* **BDEFS-style self-report** -- a brief questionnaire modelled on the
  Barkley Deficits in Executive Functioning Scale.  It covers five domains
  (time management, organisation & problem-solving, self-restraint,
  self-motivation, and emotion regulation) with three items each, rated on
  a 1-4 Likert scale (Never / Sometimes / Often / Very Often).

* **Stroop colour-word test** -- a timed task that measures the ability to
  inhibit automatic responses.  The participant names the *colour* of
  colour-words that are printed in a mismatched colour.

* **BIS/BAS motivational profile** -- a brief self-report based on the
  Behavioural Inhibition / Behavioural Activation framework. It helps tailor
  task prompts, focus interval defaults, and encouragement style.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from momentum.models import AssessmentResult, AssessmentResultCreate, AssessmentType

# Try to import Cython-compiled versions; fall back to pure Python
_CYTHON_AVAILABLE = False
try:
    from momentum._assessments_cy import score_bdefs_cy, score_bisbas_cy

    _CYTHON_AVAILABLE = True
except ImportError:
    pass  # Cython not available; use pure Python implementations below

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
    # Use Cython-compiled version if available
    if _CYTHON_AVAILABLE:
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
    if _CYTHON_AVAILABLE:
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


def interpret_bisbas(score: int, max_score: int, domain_scores: dict[str, int]) -> str:
    """Return a plain-English interpretation of BIS/BAS profile data."""
    pct = score / max_score * 100 if max_score else 0
    bis = domain_scores.get("Behavioral Inhibition (BIS)", 0)
    drive = domain_scores.get("BAS Drive", 0)
    reward = domain_scores.get("BAS Reward Responsiveness", 0)
    fun = domain_scores.get("BAS Fun Seeking", 0)
    max_domain = len(BISBAS_QUESTIONS["Behavioral Inhibition (BIS)"]) * 4
    bis_pct = bis / max_domain * 100 if max_domain else 0
    drive_pct = drive / max_domain * 100 if max_domain else 0
    reward_pct = reward / max_domain * 100 if max_domain else 0
    fun_pct = fun / max_domain * 100 if max_domain else 0

    parts: list[str] = []
    if bis_pct >= 75:
        parts.append(
            "Your BIS score is high, which can make starting feel risky or pressured."
        )
    elif bis_pct <= 40:
        parts.append(
            "Your BIS score is low-to-moderate, suggesting threat-sensitivity is less dominant."
        )
    else:
        parts.append("Your BIS score is moderate.")

    if drive_pct >= 75:
        parts.append("You show strong goal-drive persistence.")
    elif drive_pct <= 50:
        parts.append(
            "Your BAS Drive is on the lower side, so external structure may help."
        )
    else:
        parts.append("Your BAS Drive is moderate.")

    if reward_pct >= 75:
        parts.append("You respond strongly to visible rewards and progress cues.")
    elif reward_pct <= 50:
        parts.append("Immediate rewards may be less activating for you than structure.")

    if fun_pct >= 75:
        parts.append("You likely benefit from variety and shorter work blocks.")
    elif fun_pct <= 50:
        parts.append("You may tolerate routine better than novelty-driven pacing.")

    if pct <= 33:
        parts.append("Overall endorsement is low.")
    elif pct >= 66:
        parts.append("Overall endorsement is high.")
    else:
        parts.append("Overall endorsement is moderate.")
    return " ".join(parts)


@dataclass(frozen=True)
class PersonalisationProfile:
    """Operational defaults derived from BIS/BAS profile data."""

    focus_minutes: int = 15
    break_minutes: int = 5
    nudge_style: str = "balanced"
    suggest_breakdown: bool = False
    encourage_reward: bool = False
    add_reassurance: bool = False


def personalise_from_bisbas(domain_scores: dict[str, int]) -> PersonalisationProfile:
    """Map BIS/BAS domain scores to app behavior defaults."""
    max_domain = len(BISBAS_QUESTIONS["Behavioral Inhibition (BIS)"]) * 4
    if max_domain <= 0:
        return PersonalisationProfile()

    bis = domain_scores.get("Behavioral Inhibition (BIS)", 0)
    drive = domain_scores.get("BAS Drive", 0)
    reward = domain_scores.get("BAS Reward Responsiveness", 0)
    fun = domain_scores.get("BAS Fun Seeking", 0)

    bis_pct = bis / max_domain * 100
    drive_pct = drive / max_domain * 100
    reward_pct = reward / max_domain * 100
    fun_pct = fun / max_domain * 100

    high_bis = bis_pct >= 75
    low_drive = drive_pct <= 50
    high_drive = drive_pct >= 75
    high_reward = reward_pct >= 75
    high_fun = fun_pct >= 75

    focus_minutes = 15
    break_minutes = 5

    if high_bis or low_drive:
        focus_minutes = 10
        break_minutes = 6

    if high_drive and high_reward and not high_bis and not low_drive:
        focus_minutes = 20
        break_minutes = 5

    if high_fun:
        focus_minutes = min(focus_minutes, 12)
        break_minutes = max(break_minutes, 6)

    nudge_style = "balanced"
    if high_bis:
        nudge_style = "reassuring"
    elif high_reward or high_drive:
        nudge_style = "reward"

    return PersonalisationProfile(
        focus_minutes=focus_minutes,
        break_minutes=break_minutes,
        nudge_style=nudge_style,
        suggest_breakdown=(high_bis or low_drive),
        encourage_reward=(high_reward or high_drive),
        add_reassurance=high_bis,
    )


def personalised_nudge(message: str, profile: PersonalisationProfile) -> str:
    """Return a profile-adjusted encouragement line."""
    if profile.nudge_style == "reassuring":
        return f"No pressure — one tiny step is enough. {message}"
    if profile.nudge_style == "reward":
        return f"Nice momentum. Small wins count. {message}"
    return message


def profile_from_latest_bisbas(
    latest_bisbas: AssessmentResult | None,
) -> PersonalisationProfile:
    """Build a behavior profile from the latest BIS/BAS result (or defaults)."""
    if latest_bisbas is None or latest_bisbas.assessment_type != AssessmentType.BISBAS:
        return PersonalisationProfile()
    return personalise_from_bisbas(latest_bisbas.domain_scores)


def profile_from_latest_assessments(
    *,
    latest_bisbas: AssessmentResult | None,
    latest_bdefs: AssessmentResult | None,
    latest_stroop: AssessmentResult | None,
) -> PersonalisationProfile:
    """Build a behavior profile from BIS/BAS plus BDEFS/Stroop context."""

    def _clamp(value: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, value))

    base = profile_from_latest_bisbas(latest_bisbas)
    focus_minutes = base.focus_minutes
    break_minutes = base.break_minutes
    nudge_style = base.nudge_style
    suggest_breakdown = base.suggest_breakdown
    encourage_reward = base.encourage_reward
    add_reassurance = base.add_reassurance

    if latest_bdefs is not None and latest_bdefs.max_score > 0:
        bdefs_pct = latest_bdefs.score / latest_bdefs.max_score * 100
        if bdefs_pct >= 75:
            focus_minutes -= 3
            break_minutes += 2
            suggest_breakdown = True
            add_reassurance = True
            nudge_style = "reassuring"
        elif bdefs_pct >= 50:
            focus_minutes -= 2
            break_minutes += 1
            suggest_breakdown = True
        elif bdefs_pct <= 30:
            focus_minutes += 2
            encourage_reward = True

    if latest_stroop is not None and latest_stroop.max_score > 0:
        accuracy_pct = latest_stroop.score / latest_stroop.max_score * 100
        avg_ms = latest_stroop.domain_scores.get("avg_time_ms", 0)
        if accuracy_pct < 65 or avg_ms >= 2200:
            focus_minutes = min(focus_minutes, 12)
            break_minutes = max(break_minutes, 6)
            add_reassurance = True
            suggest_breakdown = True
            if nudge_style != "reward":
                nudge_style = "reassuring"
        elif accuracy_pct >= 90 and 0 < avg_ms <= 1200:
            focus_minutes += 2
            break_minutes = max(4, break_minutes - 1)
            encourage_reward = True
            if nudge_style != "reassuring":
                nudge_style = "reward"

    return PersonalisationProfile(
        focus_minutes=_clamp(focus_minutes, 8, 25),
        break_minutes=_clamp(break_minutes, 4, 12),
        nudge_style=nudge_style,
        suggest_breakdown=suggest_breakdown,
        encourage_reward=encourage_reward,
        add_reassurance=add_reassurance,
    )


def bisbas_bespoke_guidance(domain_scores: dict[str, int]) -> str:
    """Return motivationally tailored practical suggestions from BIS/BAS scores."""
    profile = personalise_from_bisbas(domain_scores)
    max_domain = len(BISBAS_QUESTIONS["Behavioral Inhibition (BIS)"]) * 4
    bis_pct = (
        domain_scores.get("Behavioral Inhibition (BIS)", 0) / max_domain * 100
        if max_domain
        else 0
    )
    drive_pct = (
        domain_scores.get("BAS Drive", 0) / max_domain * 100 if max_domain else 0
    )
    reward_pct = (
        domain_scores.get("BAS Reward Responsiveness", 0) / max_domain * 100
        if max_domain
        else 0
    )
    fun_pct = (
        domain_scores.get("BAS Fun Seeking", 0) / max_domain * 100 if max_domain else 0
    )

    tips: list[str] = []
    if bis_pct >= 75:
        tips.append(
            "Start with a low-pressure micro-step and tell yourself completion is optional for the first 2 minutes."
        )
    elif bis_pct <= 40:
        tips.append("Use clear start cues and begin quickly before over-planning.")
    else:
        tips.append(
            "Use a brief start ritual (timer, breath, first sentence) to lower friction."
        )

    if drive_pct <= 50:
        tips.append("Pair each focus block with a pre-committed accountability check.")
    elif drive_pct >= 75:
        tips.append(
            "Leverage your persistence by batching one meaningful goal per block."
        )

    if reward_pct >= 75:
        tips.append(
            "Track visible wins and celebrate each completed step to sustain momentum."
        )
    else:
        tips.append(
            "Rely on consistent routines as much as rewards when motivation dips."
        )

    if fun_pct >= 75:
        tips.append("Use shorter, varied blocks to keep engagement high.")
    else:
        tips.append("Stable routines may work well—keep the daily flow predictable.")

    tips.append(
        f"Suggested default cadence right now: {profile.focus_minutes}m focus / {profile.break_minutes}m break."
    )
    return " ".join(tips)


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
BISBAS_INSTRUCTIONS = (
    "This is a BIS/BAS motivational profile questionnaire based on "
    "Carver and White's Reinforcement Sensitivity framework.\\n\\n"
    "It covers four domains: Behavioral Inhibition (BIS), BAS Drive, "
    "BAS Reward Responsiveness, and BAS Fun Seeking.\\n\\n"
    "For each statement, rate how true it is for you:\\n"
    "  1 = Very false for me   2 = Somewhat false for me\\n"
    "  3 = Somewhat true for me   4 = Very true for me\\n\\n"
    "There are 20 items and it takes about 2-4 minutes. "
    "Your profile is used to personalize timer defaults and encouragement style."
)

RESULTS_GUIDE = (
    "The radar chart compares your two most recent BDEFS assessments. "
    "The blue polygon is your latest result; the grey polygon is your previous one. "
    "Higher values indicate greater difficulty in that area.\n\n"
    "The line chart tracks your total BDEFS score over time. "
    "The dashed trend line shows the overall direction. "
    "A downward trend suggests improvement.\\n\\n"
    "BIS/BAS results are shown as domain totals and used to tailor task support "
    "defaults such as focus length and encouragement style."
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


_BISBAS_DOMAIN_ADVICE: dict[str, dict[str, str]] = {
    "Behavioral Inhibition (BIS)": {
        "low": (
            "Lower BIS suggests fewer threat-related start barriers. Keep using clear "
            "goals and routines to maintain consistency."
        ),
        "moderate": (
            "Moderate BIS suggests occasional avoidance when stakes feel high. Use "
            "micro-starts and low-pressure language when initiating tasks."
        ),
        "high": (
            "High BIS suggests threat-sensitivity at task start. Use very small first "
            "steps, reassurance prompts, and shorter focus blocks to reduce pressure."
        ),
    },
    "BAS Drive": {
        "low": (
            "Lower BAS Drive can make sustained pursuit harder. Time-box starts and "
            "external accountability can reduce reliance on willpower."
        ),
        "moderate": (
            "Moderate BAS Drive benefits from regular structure and visible progress "
            "cues to keep momentum."
        ),
        "high": (
            "High BAS Drive is a strength for persistence. You can often tolerate "
            "longer focus intervals when goals are clear."
        ),
    },
    "BAS Reward Responsiveness": {
        "low": (
            "Lower reward responsiveness means progress cues may feel less motivating. "
            "Use fixed routines and implementation intentions in addition to rewards."
        ),
        "moderate": (
            "Moderate reward responsiveness benefits from tracking completed steps and "
            "small planned rewards."
        ),
        "high": (
            "High reward responsiveness means visible progress and celebrations are "
            "likely to strongly improve engagement."
        ),
    },
    "BAS Fun Seeking": {
        "low": (
            "Lower fun-seeking suggests routine may be easier to sustain. Consistent "
            "daily workflows can work well for you."
        ),
        "moderate": (
            "Moderate fun-seeking benefits from occasional variety while preserving a "
            "stable task structure."
        ),
        "high": (
            "High fun-seeking often benefits from shorter, varied work blocks and "
            "frequent task switching within planned boundaries."
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


def bisbas_domain_advice(domain: str, score: int, max_domain_score: int) -> str:
    """Return practical advice for a BIS/BAS subscale score."""
    advice_map = _BISBAS_DOMAIN_ADVICE.get(domain)
    if advice_map is None:
        return ""
    pct = score / max_domain_score * 100 if max_domain_score else 0
    if pct <= 50:
        return advice_map["low"]
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
