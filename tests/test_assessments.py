"""Tests for the assessments module."""

from __future__ import annotations

from momentum.assessments import (
    BDEFS_QUESTIONS,
    BDEFS_SCALE,
    StroopResult,
    StroopTrial,
    bdefs_max_score,
    generate_stroop_trials,
    interpret_bdefs,
    interpret_stroop,
    score_bdefs,
    score_stroop,
)
from momentum.models import AssessmentType


class TestBDEFS:
    def test_scale_has_four_levels(self) -> None:
        assert len(BDEFS_SCALE) == 4
        assert 1 in BDEFS_SCALE and 4 in BDEFS_SCALE

    def test_questions_has_five_domains(self) -> None:
        assert len(BDEFS_QUESTIONS) == 5

    def test_max_score(self) -> None:
        n_items = sum(len(qs) for qs in BDEFS_QUESTIONS.values())
        assert bdefs_max_score() == n_items * 4

    def test_score_bdefs_minimal(self) -> None:
        answers = {d: [1] * len(qs) for d, qs in BDEFS_QUESTIONS.items()}
        result = score_bdefs(answers)
        assert result.assessment_type == AssessmentType.BDEFS
        assert result.score == sum(len(qs) for qs in BDEFS_QUESTIONS.values())
        assert result.max_score == bdefs_max_score()

    def test_score_bdefs_maximal(self) -> None:
        answers = {d: [4] * len(qs) for d, qs in BDEFS_QUESTIONS.items()}
        result = score_bdefs(answers)
        assert result.score == bdefs_max_score()

    def test_score_bdefs_domain_scores(self) -> None:
        answers = {d: [2] * len(qs) for d, qs in BDEFS_QUESTIONS.items()}
        result = score_bdefs(answers)
        for domain, qs in BDEFS_QUESTIONS.items():
            assert result.domain_scores[domain] == 2 * len(qs)


class TestInterpretBDEFS:
    def test_minimal(self) -> None:
        text = interpret_bdefs(15, 60)  # 25%
        assert "Minimal" in text

    def test_mild(self) -> None:
        text = interpret_bdefs(20, 60)  # ~33%
        assert "Mild" in text

    def test_moderate(self) -> None:
        text = interpret_bdefs(40, 60)  # ~67%
        assert "Moderate" in text

    def test_significant(self) -> None:
        text = interpret_bdefs(55, 60)  # ~92%
        assert "Significant" in text


class TestStroop:
    def test_trial_mismatch(self) -> None:
        trial = StroopTrial(word="red", ink_colour="blue")
        assert trial.word != trial.ink_colour

    def test_generate_trials_default_count(self) -> None:
        trials = generate_stroop_trials()
        assert len(trials) == 10

    def test_generate_trials_custom_count(self) -> None:
        trials = generate_stroop_trials(5)
        assert len(trials) == 5

    def test_generate_trials_all_mismatched(self) -> None:
        trials = generate_stroop_trials(20)
        for t in trials:
            assert t.word != t.ink_colour

    def test_stroop_result_accuracy(self) -> None:
        r = StroopResult(trials=10, correct=7, total_time_s=15.0)
        assert r.accuracy_pct == 70.0

    def test_stroop_result_avg_time(self) -> None:
        r = StroopResult(trials=10, correct=7, total_time_s=15.0)
        assert r.avg_time_s == 1.5

    def test_stroop_result_empty(self) -> None:
        r = StroopResult(trials=0, correct=0, total_time_s=0.0)
        assert r.accuracy_pct == 0.0
        assert r.avg_time_s == 0.0

    def test_score_stroop(self) -> None:
        r = StroopResult(trials=10, correct=8, total_time_s=12.0)
        result = score_stroop(r)
        assert result.assessment_type == AssessmentType.STROOP
        assert result.score == 8
        assert result.max_score == 10
        assert result.domain_scores["avg_time_ms"] == 1200


class TestInterpretStroop:
    def test_excellent_fast(self) -> None:
        text = interpret_stroop(9, 10, 800)
        assert "Excellent" in text
        assert "fast" in text

    def test_good_average(self) -> None:
        text = interpret_stroop(8, 10, 1500)
        assert "Good" in text
        assert "average" in text

    def test_moderate_slow(self) -> None:
        text = interpret_stroop(6, 10, 2500)
        assert "Moderate" in text
        assert "slower" in text

    def test_low(self) -> None:
        text = interpret_stroop(3, 10, 1000)
        assert "Low" in text
