"""Curated encouragement messages grounded in CBT and self-compassion principles.

Messages are loaded from ``ENCOURAGEMENTS.md`` at the project root.
The user can freely add, edit, or remove messages in that file.
If the file is missing, a small built-in fallback list is used.
"""

from __future__ import annotations

import random
from pathlib import Path

_FALLBACK_MESSAGES: list[str] = [
    "Starting is the hardest part. You have already done that.",
    "Progress does not have to be perfect to count.",
    "You are allowed to do things slowly.",
    "One small step is still a step.",
    "Rest is not the opposite of productivity. It is part of it.",
    "Doing something imperfectly is better than not doing it at all.",
    "Your pace is valid.",
    "You have gotten through difficult days before. This is one of them.",
    "Be as kind to yourself as you would be to a friend.",
    "You are not lazy. You are dealing with something real.",
    "Showing up -- even like this -- matters.",
]


def _load_messages() -> list[str]:
    """Parse bullet points from ENCOURAGEMENTS.md, falling back to built-in list."""
    md_path = Path(__file__).resolve().parent.parent / "ENCOURAGEMENTS.md"
    if not md_path.exists():
        return _FALLBACK_MESSAGES

    messages: list[str] = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            msg = stripped[2:].strip()
            if msg:
                messages.append(msg)
    return messages if messages else _FALLBACK_MESSAGES


_MESSAGES: list[str] = _load_messages()


def get_nudge() -> str:
    """Return a single random encouragement message."""
    return random.choice(_MESSAGES)


def get_break_message() -> str:
    """Return a calming message for break time."""
    break_messages: list[str] = [
        "Step away from the screen for a moment.",
        "Take a few slow breaths.",
        "Stretch your shoulders and neck.",
        "Look at something far away for twenty seconds.",
        "Get some water if you can.",
        "Close your eyes for a moment. You have earned this pause.",
    ]
    return random.choice(break_messages)
