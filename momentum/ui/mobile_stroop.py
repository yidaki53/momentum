from __future__ import annotations

import random

from momentum.domain.assessments import STROOP_COLOURS, StroopTrial


def shuffled_stroop_options(
    trial: StroopTrial,
    *,
    rng: random.Random | None = None,
) -> list[str]:
    chooser = rng or random
    options = list(STROOP_COLOURS)
    return chooser.sample(options, len(options))
