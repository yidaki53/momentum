"""Tests for the charts module."""

from __future__ import annotations

from datetime import datetime, timedelta

from PIL import Image

from momentum.charts import bdefs_momentum_glow, bdefs_radar, bdefs_timeseries
from momentum.models import AssessmentResult, AssessmentType


def _make_bdefs_result(
    score: int = 30,
    taken_at: datetime | None = None,
) -> AssessmentResult:
    """Helper to build a fake BDEFS result."""
    return AssessmentResult(
        id=1,
        assessment_type=AssessmentType.BDEFS,
        score=score,
        max_score=60,
        domain_scores={
            "Time Management": 6,
            "Organisation & Problem-Solving": 6,
            "Self-Restraint": 6,
            "Self-Motivation": 6,
            "Emotion Regulation": 6,
        },
        taken_at=taken_at or datetime.now(),
    )


class TestBdefsRadar:
    def test_returns_image_with_latest(self) -> None:
        result = _make_bdefs_result()
        img = bdefs_radar(latest=result)
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_returns_image_previous_only(self) -> None:
        prev = _make_bdefs_result(score=20)
        img = bdefs_radar(previous=prev)
        assert isinstance(img, Image.Image)

    def test_returns_image_no_data(self) -> None:
        img = bdefs_radar()
        assert isinstance(img, Image.Image)

    def test_custom_size(self) -> None:
        img = bdefs_radar(latest=_make_bdefs_result(), size=(300, 300), dpi=50)
        assert isinstance(img, Image.Image)

    def test_with_latest_and_previous(self) -> None:
        latest = _make_bdefs_result(score=30)
        previous = _make_bdefs_result(score=20)
        img = bdefs_radar(latest=latest, previous=previous)
        assert isinstance(img, Image.Image)


class TestBdefsTimeseries:
    def test_returns_none_for_fewer_than_two(self) -> None:
        assert bdefs_timeseries([]) is None
        assert bdefs_timeseries([_make_bdefs_result()]) is None

    def test_returns_image_for_two_or_more(self) -> None:
        now = datetime.now()
        results = [
            _make_bdefs_result(score=20, taken_at=now - timedelta(days=7)),
            _make_bdefs_result(score=25, taken_at=now),
        ]
        img = bdefs_timeseries(results)
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0

    def test_ignores_non_bdefs(self) -> None:
        stroop = AssessmentResult(
            id=2,
            assessment_type=AssessmentType.STROOP,
            score=8,
            max_score=10,
            domain_scores={},
            taken_at=datetime.now(),
        )
        result = bdefs_timeseries([stroop, _make_bdefs_result()])
        assert result is None  # only 1 BDEFS result


class TestBdefsMomentumGlow:
    def test_returns_image_for_latest_result(self) -> None:
        img = bdefs_momentum_glow(_make_bdefs_result())
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_returns_image_with_previous_comparison(self) -> None:
        latest = _make_bdefs_result(score=24)
        previous = _make_bdefs_result(score=34)
        img = bdefs_momentum_glow(latest, previous=previous)
        assert isinstance(img, Image.Image)
