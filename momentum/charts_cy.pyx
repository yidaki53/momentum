# cython: language_level=3, boundscheck=False, cdivision=True

"""Cython-compiled chart calculations (tight math loops)."""

from typing import List, Dict
from momentum.models import AssessmentResult
from momentum.assessments import BDEFS_QUESTIONS


def domain_percentages_cy(result: AssessmentResult) -> List[float]:
    """Cythonic percentage calculation: vectorized division."""
    cdef list percentages = []
    cdef str domain
    cdef list questions
    cdef int max_domain_score
    cdef float raw_score
    cdef float reserve
    cdef float pct

    domain_order: List[str] = list(BDEFS_QUESTIONS.keys())

    for domain in domain_order:
        questions = BDEFS_QUESTIONS[domain]
        max_domain_score = len(questions) * 4
        raw_score = <float>result.domain_scores.get(domain, 0)
        reserve = max(max_domain_score - raw_score, 0)
        pct = (reserve / <float>max_domain_score * 100.0) if max_domain_score else 0.0
        percentages.append(pct)

    return percentages
