# cython: language_level=3, boundscheck=False, cdivision=True

"""Cython-compiled chart calculations (tight math loops)."""

from typing import List, Dict

from momentum.models import AssessmentResult

# See the circular-import note in momentum/assessments_cy.pyx: questionnaire data
# is imported lazily so this extension stays importable from any entry point.
_DEPENDENCY_CACHE = {}


def _bdefs_questions() -> dict:
    """Return BDEFS_QUESTIONS, importing it on first use."""
    if "bdefs" not in _DEPENDENCY_CACHE:
        from momentum.assessments import BDEFS_QUESTIONS

        _DEPENDENCY_CACHE["bdefs"] = BDEFS_QUESTIONS
    return _DEPENDENCY_CACHE["bdefs"]


def domain_percentages_cy(result: AssessmentResult) -> List[float]:
    """Cythonic percentage calculation: vectorized division."""
    cdef list percentages = []
    cdef str domain
    cdef list questions
    cdef int max_domain_score
    # ``double`` (not ``float``) so the compiled path stays bit-for-bit
    # equivalent to the pure Python fallback in ui/charts.py: C ``float`` is
    # 32-bit and changed the results (66.666672 vs 66.666667).
    cdef double raw_score
    cdef double reserve
    cdef double pct

    cdef dict questions_by_domain = _bdefs_questions()
    domain_order: List[str] = list(questions_by_domain.keys())

    for domain in domain_order:
        questions = questions_by_domain[domain]
        max_domain_score = len(questions) * 4
        raw_score = <float>result.domain_scores.get(domain, 0)
        reserve = max(max_domain_score - raw_score, 0)
        pct = (reserve / <float>max_domain_score * 100.0) if max_domain_score else 0.0
        percentages.append(pct)

    return percentages
