"""Matplotlib charts for assessment visualisation.

All figures use a dark theme consistent with the Momentum GUI palette.
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- render to image buffers
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from momentum.assessments import BDEFS_QUESTIONS
from momentum.models import AssessmentResult, AssessmentType

# -- Palette (matches GUI dark theme) ------------------------------------
_BG = "#2b2b2b"
_FG = "#e0e0e0"
_ACCENT = "#6a9fb5"
_BLUE_FILL = (0.42, 0.62, 0.71, 0.30)   # translucent accent blue
_BLUE_LINE = "#6a9fb5"
_GREY_FILL = (0.70, 0.70, 0.70, 0.08)
_GREY_LINE = "#555555"
_GRID = "#444444"

# Canonical ordered list of BDEFS domain names (short labels for the axes).
_DOMAIN_ORDER: list[str] = list(BDEFS_QUESTIONS.keys())
_DOMAIN_MAX = max(len(qs) for qs in BDEFS_QUESTIONS.values()) * 4  # 12


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _fig_to_pil(fig: Figure, dpi: int = 100) -> Image.Image:
    """Render a matplotlib Figure to a PIL Image and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _domain_values(result: AssessmentResult) -> list[float]:
    """Extract domain scores in canonical order, defaulting to 0."""
    return [float(result.domain_scores.get(d, 0)) for d in _DOMAIN_ORDER]


# -----------------------------------------------------------------------
# Radar / spider chart
# -----------------------------------------------------------------------

def bdefs_radar(
    highlight: Optional[AssessmentResult] = None,
    past: Optional[list[AssessmentResult]] = None,
    *,
    title: str = "Executive Function Profile",
    size: tuple[int, int] = (540, 440),
    dpi: int = 100,
) -> Image.Image:
    """Draw a BDEFS radar chart and return it as a PIL Image.

    Parameters
    ----------
    highlight:
        The result to draw in blue (current result or the mean).
        If *None*, the mean of *past* is used as the highlight.
    past:
        Previous results rendered as grey polygons in the background.
    title:
        Chart title.
    """
    past = past or []

    # If no explicit highlight, compute the mean of past results.
    if highlight is None and past:
        mean_vals = [0.0] * len(_DOMAIN_ORDER)
        for r in past:
            for i, v in enumerate(_domain_values(r)):
                mean_vals[i] += v
            n = len(past)
        mean_vals = [v / n for v in mean_vals]
    elif highlight is not None:
        mean_vals = _domain_values(highlight)
    else:
        mean_vals = [0.0] * len(_DOMAIN_ORDER)

    n_axes = len(_DOMAIN_ORDER)
    angles = np.linspace(0, 2 * math.pi, n_axes, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]
    mean_vals_closed = mean_vals + mean_vals[:1]

    fig_w, fig_h = size[0] / dpi, size[1] / dpi
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=_BG)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(_BG)

    # Grid & ticks
    ax.set_ylim(0, _DOMAIN_MAX)
    ax.set_yticks(range(0, _DOMAIN_MAX + 1, 3))
    ax.set_yticklabels([str(v) for v in range(0, _DOMAIN_MAX + 1, 3)],
                       color=_FG, fontsize=7)
    ax.set_xticks(angles[:-1])
    short_labels = [d.replace(" & ", "\n& ").replace("Organisation", "Org.") for d in _DOMAIN_ORDER]
    ax.set_xticklabels(short_labels, color=_FG, fontsize=8)
    ax.tick_params(axis="x", pad=12)
    ax.spines["polar"].set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.5)

    # Past results (grey)
    for r in past:
        vals = _domain_values(r) + _domain_values(r)[:1]
        ax.plot(angles, vals, color=_GREY_LINE, linewidth=0.8, alpha=0.5)
        ax.fill(angles, vals, color=_GREY_FILL)

    # Highlight (blue)
    ax.plot(angles, mean_vals_closed, color=_BLUE_LINE, linewidth=2)
    ax.fill(angles, mean_vals_closed, color=_BLUE_FILL)

    ax.set_title(title, color=_FG, fontsize=11, fontweight="bold", pad=18)

    return _fig_to_pil(fig, dpi=dpi)


# -----------------------------------------------------------------------
# Timeseries
# -----------------------------------------------------------------------

def bdefs_timeseries(
    results: list[AssessmentResult],
    *,
    title: str = "Score Over Time",
    size: tuple[int, int] = (560, 240),
    dpi: int = 100,
) -> Optional[Image.Image]:
    """Line chart of BDEFS total score over time.

    Returns *None* if fewer than two results are provided.
    """
    bdefs = [r for r in results if r.assessment_type == AssessmentType.BDEFS]
    bdefs.sort(key=lambda r: r.taken_at)
    if len(bdefs) < 2:
        return None

    dates = [r.taken_at for r in bdefs]
    scores = [r.score for r in bdefs]
    max_score = bdefs[0].max_score

    fig_w, fig_h = size[0] / dpi, size[1] / dpi
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(_BG)

    ax.plot(dates, scores, color=_BLUE_LINE, linewidth=2, marker="o",
            markersize=5, markerfacecolor=_ACCENT, markeredgecolor="white",
            markeredgewidth=0.5)
    ax.fill_between(dates, scores, alpha=0.15, color=_ACCENT)

    ax.set_ylim(0, max_score)
    ax.set_ylabel("Total score", color=_FG, fontsize=9)
    ax.set_title(title, color=_FG, fontsize=11, fontweight="bold")

    ax.tick_params(colors=_FG, labelsize=8)
    ax.spines["bottom"].set_color(_GRID)
    ax.spines["left"].set_color(_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(color=_GRID, linewidth=0.5)

    fig.autofmt_xdate(rotation=30, ha="right")

    return _fig_to_pil(fig, dpi=dpi)
