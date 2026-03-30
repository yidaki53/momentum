"""Interpretation functions and domain-specific advice for assessments."""

from momentum.domain.assessments.scoring import (
    BISBAS_QUESTIONS,
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
# BDEFS Interpretation
# ---------------------------------------------------------------------------


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
# BIS/BAS Interpretation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Domain-specific advice (science-backed, practical)
# ---------------------------------------------------------------------------

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


def bisbas_bespoke_guidance(domain_scores: dict[str, int]) -> str:
    """Return motivationally tailored practical suggestions from BIS/BAS scores."""
    from momentum.domain.assessments.profile import personalise_from_bisbas

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
