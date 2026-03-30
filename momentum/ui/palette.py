"""Shared theme palette for UI components.

Provides consistent color definitions across GUI, charts, and display layers.
Supports light/dark modes and high-contrast accessibility.
"""

from __future__ import annotations

from enum import Enum


class ThemeMode(Enum):
    """Theme mode selection."""

    LIGHT = "light"
    DARK = "dark"


def get_palette(
    theme_mode: ThemeMode = ThemeMode.DARK,
    high_contrast: bool = False,
) -> dict[str, str]:
    """Return a colour palette dictionary for the specified theme.

    Parameters
    ----------
    theme_mode : ThemeMode
        Light or dark theme.
    high_contrast : bool
        If True, apply high-contrast adaptation for accessibility.

    Returns
    -------
    dict[str, str]
        Mapping of palette keys (bg, fg, accent, etc.) to hex colour codes.
    """
    if theme_mode == ThemeMode.LIGHT:
        palette = {
            "bg": "#f5f6f8",
            "panel": "#ffffff",
            "fg": "#1f2933",
            "muted": "#5f6b76",
            "accent": "#2f6f8f",
            "timer": "#a86f00",
            "selection": "#9cc8e0",
            "progress_trough": "#d9e1e8",
        }
    else:  # DARK
        palette = {
            "bg": "#2b2b2b",
            "panel": "#333333",
            "fg": "#e0e0e0",
            "muted": "#b5b5b5",
            "accent": "#6a9fb5",
            "timer": "#e8c547",
            "selection": "#6a9fb5",
            "progress_trough": "#444444",
        }

    if high_contrast:
        if theme_mode == ThemeMode.LIGHT:
            palette.update(
                {
                    "fg": "#000000",
                    "muted": "#222222",
                    "accent": "#005fcc",
                    "selection": "#79b0ff",
                }
            )
        else:  # DARK
            palette.update(
                {
                    "fg": "#ffffff",
                    "muted": "#e6e6e6",
                    "accent": "#8cc9ff",
                    "selection": "#6dafff",
                }
            )
    return palette


# Dark theme palette for charts (default)
CHART_BG = "#2b2b2b"
CHART_FG = "#e0e0e0"
CHART_ACCENT = "#6a9fb5"
CHART_BLUE_FILL = (0.42, 0.62, 0.71, 0.30)  # translucent accent blue
CHART_BLUE_LINE = "#6a9fb5"
CHART_GREY_FILL = (0.70, 0.70, 0.70, 0.08)
CHART_GREY_LINE = "#555555"
CHART_GRID = "#444444"

__all__ = [
    "ThemeMode",
    "get_palette",
    "CHART_BG",
    "CHART_FG",
    "CHART_ACCENT",
    "CHART_BLUE_FILL",
    "CHART_BLUE_LINE",
    "CHART_GREY_FILL",
    "CHART_GREY_LINE",
    "CHART_GRID",
]
