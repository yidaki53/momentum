from __future__ import annotations

from typing import Literal

HomeSection = Literal["tasks", "timer", "journal"]


def next_home_section_state(
    section: HomeSection,
    *,
    tasks_expanded: bool,
    timer_expanded: bool,
    journal_expanded: bool,
) -> tuple[bool, bool, bool]:
    """Return the next home-section state, allowing multiple sections open."""
    if section == "tasks":
        return (not tasks_expanded, timer_expanded, journal_expanded)
    if section == "timer":
        return (tasks_expanded, not timer_expanded, journal_expanded)
    if section == "journal":
        return (tasks_expanded, timer_expanded, not journal_expanded)
    raise ValueError(f"Unknown home section: {section}")
