from __future__ import annotations

import random

from momentum.assessments import STROOP_COLOURS, StroopTrial
from momentum.ui.mobile_stroop import shuffled_stroop_options


def test_shuffled_stroop_options_include_all_colours() -> None:
    trial = StroopTrial(word="red", ink_colour="blue")
    options = shuffled_stroop_options(trial, rng=random.Random(3))

    assert len(options) == 4
    assert set(options) == set(STROOP_COLOURS)
    assert trial.word in options
    assert trial.ink_colour in options
