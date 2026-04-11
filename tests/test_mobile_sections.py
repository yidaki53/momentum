from __future__ import annotations

import pytest

from momentum.ui.mobile_sections import next_home_section_state


@pytest.mark.parametrize(
    ("section", "current", "expected"),
    [
        ("tasks", (True, False, False), (False, False, False)),
        ("tasks", (False, True, False), (True, True, False)),
        ("timer", (True, False, False), (True, True, False)),
        ("timer", (False, True, False), (False, False, False)),
        ("journal", (False, True, False), (False, True, True)),
        ("journal", (False, False, True), (False, False, False)),
    ],
)
def test_next_home_section_state(
    section: str, current: tuple[bool, bool, bool], expected: tuple[bool, bool, bool]
) -> None:
    assert (
        next_home_section_state(
            section,
            tasks_expanded=current[0],
            timer_expanded=current[1],
            journal_expanded=current[2],
        )
        == expected
    )


def test_next_home_section_state_rejects_unknown_section() -> None:
    with pytest.raises(ValueError, match="Unknown home section"):
        next_home_section_state(
            "unknown",
            tasks_expanded=True,
            timer_expanded=False,
            journal_expanded=False,
        )
