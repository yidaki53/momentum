# cython: language_level=3, boundscheck=False, cdivision=True

"""Cython-compiled assessment scoring (performance-critical paths)."""

from typing import Dict, List
from momentum.models import AssessmentResult, AssessmentResultCreate, AssessmentType
from momentum.assessments import (
    BDEFS_QUESTIONS,
    BISBAS_QUESTIONS,
    bdefs_max_score,
    bisbas_max_score,
)


def score_bdefs_cy(answers: Dict[str, List[int]]) -> AssessmentResultCreate:
    """Cythonic BDEFS scoring: tighter loop, typed aggregation."""
    cdef int total = 0
    cdef int ds
    cdef list scores
    cdef int score_val

    domain_scores: Dict[str, int] = {}

    for domain, scores in answers.items():
        ds = 0
        for score_val in scores:
            ds += score_val
        domain_scores[domain] = ds
        total += ds

    return AssessmentResultCreate(
        assessment_type=AssessmentType.BDEFS,
        score=total,
        max_score=bdefs_max_score(),
        domain_scores=domain_scores,
    )


def score_bisbas_cy(answers: Dict[str, List[int]]) -> AssessmentResultCreate:
    """Cythonic BIS/BAS scoring."""
    cdef int total = 0
    cdef int ds
    cdef list scores
    cdef int score_val

    domain_scores: Dict[str, int] = {}

    for domain, scores in answers.items():
        ds = 0
        for score_val in scores:
            ds += score_val
        domain_scores[domain] = ds
        total += ds

    return AssessmentResultCreate(
        assessment_type=AssessmentType.BISBAS,
        score=total,
        max_score=bisbas_max_score(),
        domain_scores=domain_scores,
    )


def profile_from_latest_assessments_cy(
    assessments: List[AssessmentResult],
) -> Dict[str, float]:
    """Cythonic profile aggregation: faster averaging over cold data."""
    if not assessments:
        return {}

    cdef int n_assessments = len(assessments)
    cdef float count = <float>n_assessments
    cdef dict totals = {}
    cdef dict final_profiles = {}
    cdef str domain
    cdef float score_val
    cdef float avg

    for result in assessments:
        for domain, score_val in result.domain_scores.items():
            if domain not in totals:
                totals[domain] = 0
            totals[domain] += score_val

    for domain, total in totals.items():
        avg = total / count if count > 0 else 0.0
        final_profiles[domain] = avg

    return final_profiles
