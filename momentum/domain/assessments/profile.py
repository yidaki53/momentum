"""Personalization profile generation from assessment results."""

from __future__ import annotations

from dataclasses import dataclass

from momentum.domain.assessments.scoring import BISBAS_QUESTIONS
from momentum.models import AssessmentResult, AssessmentType


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
