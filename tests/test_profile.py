"""Tests for assessment-driven personalisation profiles."""

from __future__ import annotations

from momentum.domain.assessments.profile import (
    PersonalisationProfile,
    personalised_act_guidance,
    personalised_nudge,
    profile_from_latest_assessments,
    profile_from_latest_bisbas,
    should_show_act_support,
)
from momentum.models import AssessmentResult, AssessmentType


def _assessment(
    assessment_type: AssessmentType,
    *,
    score: int,
    max_score: int,
    domain_scores: dict[str, int],
) -> AssessmentResult:
    return AssessmentResult(
        id=1,
        assessment_type=assessment_type,
        score=score,
        max_score=max_score,
        domain_scores=domain_scores,
    )


class TestActSupportHelpers:
    def test_should_show_act_support_when_reassurance_needed(self) -> None:
        profile = PersonalisationProfile(add_reassurance=True)

        assert should_show_act_support(profile) is True

    def test_should_show_act_support_when_breakdown_needed(self) -> None:
        profile = PersonalisationProfile(suggest_breakdown=True)

        assert should_show_act_support(profile) is True

    def test_should_not_show_act_support_by_default(self) -> None:
        assert should_show_act_support(PersonalisationProfile()) is False

    def test_personalised_act_guidance_prefers_combined_message(self) -> None:
        profile = PersonalisationProfile(add_reassurance=True, suggest_breakdown=True)

        guidance = personalised_act_guidance(profile)

        assert "short ACT check-in" in guidance
        assert "tiny next action" in guidance

    def test_personalised_nudge_adjusts_reassuring_style(self) -> None:
        profile = PersonalisationProfile(nudge_style="reassuring")

        assert personalised_nudge("Try again.", profile).startswith("No pressure")


class TestProfileFromLatestAssessments:
    def test_profile_from_latest_bisbas_returns_defaults_when_missing(self) -> None:
        profile = profile_from_latest_bisbas(None)

        assert profile == PersonalisationProfile()

    def test_bdefs_overwhelm_branch_shortens_focus_and_adds_support(self) -> None:
        profile = profile_from_latest_assessments(
            latest_bisbas=None,
            latest_bdefs=_assessment(
                AssessmentType.BDEFS,
                score=48,
                max_score=60,
                domain_scores={},
            ),
            latest_stroop=None,
        )

        assert profile.focus_minutes == 12
        assert profile.break_minutes == 7
        assert profile.suggest_breakdown is True
        assert profile.add_reassurance is True
        assert profile.nudge_style == "reassuring"

    def test_stroop_slow_branch_caps_focus_and_adds_reassurance(self) -> None:
        profile = profile_from_latest_assessments(
            latest_bisbas=None,
            latest_bdefs=None,
            latest_stroop=_assessment(
                AssessmentType.STROOP,
                score=6,
                max_score=10,
                domain_scores={"avg_time_ms": 2400},
            ),
        )

        assert profile.focus_minutes == 12
        assert profile.break_minutes == 6
        assert profile.suggest_breakdown is True
        assert profile.add_reassurance is True
        assert profile.nudge_style == "reassuring"

    def test_stroop_strong_branch_rewards_fast_accurate_performance(self) -> None:
        profile = profile_from_latest_assessments(
            latest_bisbas=None,
            latest_bdefs=None,
            latest_stroop=_assessment(
                AssessmentType.STROOP,
                score=9,
                max_score=10,
                domain_scores={"avg_time_ms": 1100},
            ),
        )

        assert profile.focus_minutes == 17
        assert profile.break_minutes == 4
        assert profile.encourage_reward is True
        assert profile.nudge_style == "reward"
