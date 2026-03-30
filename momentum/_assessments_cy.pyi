"""Type stubs for momentum._assessments_cy (Cython-compiled assessments)."""

from typing import Dict, List

from momentum.models import AssessmentResult, AssessmentResultCreate

def score_bdefs_cy(answers: Dict[str, List[int]]) -> AssessmentResultCreate: ...
def score_bisbas_cy(answers: Dict[str, List[int]]) -> AssessmentResultCreate: ...
def profile_from_latest_assessments_cy(
    assessments: List[AssessmentResult],
) -> Dict[str, float]: ...
