"""Assessment domain module: scoring, interpretation, and personalization.

Exports:
- Scoring: BDEFS_*, BISBAS_*, STROOP_*, score_*, generate_stroop_trials, StroopTrial, StroopResult
- Interpretation: interpret_*, domain_advice, bisbas_domain_advice, bisbas_bespoke_guidance
- Profile: PersonalisationProfile, personalise_from_bisbas, profile_from_*, personalised_nudge

These are organized into sub-modules for clarity but re-exported here for backward compatibility.
"""

from momentum.domain.assessments.interpretation import (
    BDEFS_INSTRUCTIONS,
    BISBAS_INSTRUCTIONS,
    RESULTS_GUIDE,
    STROOP_INSTRUCTIONS,
    bisbas_bespoke_guidance,
    bisbas_domain_advice,
    domain_advice,
    interpret_bdefs,
    interpret_bisbas,
    interpret_stroop,
)
from momentum.domain.assessments.profile import (
    PersonalisationProfile,
    personalise_from_bisbas,
    personalised_act_guidance,
    personalised_nudge,
    profile_from_latest_assessments,
    profile_from_latest_bisbas,
    should_show_act_support,
)
from momentum.domain.assessments.scoring import (
    BDEFS_MAX_PER_ITEM,
    BDEFS_MIN_PER_ITEM,
    BDEFS_QUESTIONS,
    BDEFS_SCALE,
    BDEFS_SCALE_LABELS,
    BISBAS_MAX_PER_ITEM,
    BISBAS_MIN_PER_ITEM,
    BISBAS_QUESTIONS,
    BISBAS_SCALE,
    BISBAS_SCALE_LABELS,
    STROOP_COLOURS,
    STROOP_DEFAULT_TRIALS,
    StroopResult,
    StroopTrial,
    bdefs_max_score,
    bisbas_effective_domain_max_score,
    bisbas_effective_max_score,
    bisbas_max_score,
    bisbas_normalized_domain_score,
    bisbas_normalized_total_score,
    bisbas_total_min_score,
    generate_stroop_trials,
    score_bdefs,
    score_bisbas,
    score_stroop,
)

__all__ = [
    # Scoring
    "BDEFS_SCALE",
    "BDEFS_SCALE_LABELS",
    "BDEFS_QUESTIONS",
    "BDEFS_MIN_PER_ITEM",
    "BDEFS_MAX_PER_ITEM",
    "bdefs_max_score",
    "score_bdefs",
    "BISBAS_SCALE",
    "BISBAS_SCALE_LABELS",
    "BISBAS_QUESTIONS",
    "BISBAS_MIN_PER_ITEM",
    "BISBAS_MAX_PER_ITEM",
    "bisbas_total_min_score",
    "bisbas_effective_max_score",
    "bisbas_effective_domain_max_score",
    "bisbas_max_score",
    "bisbas_normalized_total_score",
    "bisbas_normalized_domain_score",
    "score_bisbas",
    "STROOP_COLOURS",
    "STROOP_DEFAULT_TRIALS",
    "StroopTrial",
    "StroopResult",
    "generate_stroop_trials",
    "score_stroop",
    # Interpretation
    "interpret_bdefs",
    "interpret_bisbas",
    "interpret_stroop",
    "domain_advice",
    "bisbas_domain_advice",
    "bisbas_bespoke_guidance",
    "BDEFS_INSTRUCTIONS",
    "STROOP_INSTRUCTIONS",
    "BISBAS_INSTRUCTIONS",
    "RESULTS_GUIDE",
    # Profile
    "PersonalisationProfile",
    "personalise_from_bisbas",
    "personalised_nudge",
    "personalised_act_guidance",
    "should_show_act_support",
    "profile_from_latest_bisbas",
    "profile_from_latest_assessments",
]
