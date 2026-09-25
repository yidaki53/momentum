# cython: language_level=3, boundscheck=False, cdivision=True

"""Cython-compiled assessment scoring (performance-critical paths)."""

from typing import Dict, List

from momentum.models import AssessmentResult, AssessmentResultCreate, AssessmentType

# Questionnaire helpers are imported lazily, inside the functions below.
#
# ``momentum.assessments`` is a backwards-compatibility shim for
# ``momentum.domain.assessments``, whose ``__init__`` imports
# ``momentum.domain.assessments.scoring`` -- the module that imports *this*
# extension. Importing the shim at module level therefore re-entered a
# half-built ``scoring`` (or a half-built ``momentum.assessments``, when this
# extension was imported first) and raised ImportError, which ``scoring.py``'s
# ``except ImportError`` fallback swallowed. The compiled fast path was thus
# silently disabled for every import order. Deferring the import to call time
# (when the package is fully built) removes the cycle.
_DEPENDENCY_CACHE = {}


def _questionnaire_deps() -> dict:
    """Return the questionnaire helpers, importing them on first use."""
    if not _DEPENDENCY_CACHE:
        from momentum.assessments import bdefs_max_score, bisbas_max_score

        _DEPENDENCY_CACHE["bdefs_max_score"] = bdefs_max_score
        _DEPENDENCY_CACHE["bisbas_max_score"] = bisbas_max_score
    return _DEPENDENCY_CACHE


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
        max_score=_questionnaire_deps()["bdefs_max_score"](),
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
        max_score=_questionnaire_deps()["bisbas_max_score"](),
        domain_scores=domain_scores,
    )


def profile_from_latest_assessments_cy(
    assessments: List[AssessmentResult],
) -> Dict[str, float]:
    """Cythonic profile aggregation: faster averaging over cold data."""
    if not assessments:
        return {}

    cdef int n_assessments = len(assessments)
    cdef double count = <double>n_assessments
    cdef dict totals = {}
    cdef dict final_profiles = {}
    cdef str domain
    # ``double`` (not ``float``): C ``float`` is 32-bit and would diverge from the
    # pure Python averaging in domain/assessments/profile.py.
    cdef double score_val
    cdef double avg

    for result in assessments:
        for domain, score_val in result.domain_scores.items():
            if domain not in totals:
                totals[domain] = 0
            totals[domain] += score_val

    for domain, total in totals.items():
        avg = total / count if count > 0 else 0.0
        final_profiles[domain] = avg

    return final_profiles
