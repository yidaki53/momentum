"""Matplotlib charts for assessment visualisation.

All figures use a dark theme consistent with the Momentum GUI palette.
"""

from __future__ import annotations

import io
import math
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend -- render to image buffers
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from momentum.assessments import BDEFS_QUESTIONS
from momentum.models import AssessmentResult, AssessmentType

# Try to import Cython-compiled chart functions; fall back to pure Python
_CYTHON_AVAILABLE = False
try:
    from momentum._charts_cy import domain_percentages_cy

    _CYTHON_AVAILABLE = True
except ImportError:
    pass  # Cython not available; use pure Python implementations below

# -- Palette (matches GUI dark theme) ------------------------------------
_BG = "#2b2b2b"
_FG = "#e0e0e0"
_ACCENT = "#6a9fb5"
_BLUE_FILL = (0.42, 0.62, 0.71, 0.30)  # translucent accent blue
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
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _domain_values(result: AssessmentResult) -> list[float]:
    """Extract domain scores in canonical order, defaulting to 0."""
    return [float(result.domain_scores.get(d, 0)) for d in _DOMAIN_ORDER]


def _domain_percentages(result: AssessmentResult) -> list[float]:
    """Convert raw BDEFS scores into momentum percentages where higher is better."""
    # Use Cython-compiled version if available
    if _CYTHON_AVAILABLE:
        return domain_percentages_cy(result)

    # Pure Python fallback
    percentages: list[float] = []
    for domain in _DOMAIN_ORDER:
        questions = BDEFS_QUESTIONS[domain]
        max_domain_score = len(questions) * 4
        raw_score = float(result.domain_scores.get(domain, 0))
        reserve = max(max_domain_score - raw_score, 0)
        percentages.append(
            (reserve / max_domain_score * 100) if max_domain_score else 0.0
        )
    return percentages


# -----------------------------------------------------------------------
# Radar / spider chart
# -----------------------------------------------------------------------


def bdefs_radar(
    latest: Optional[AssessmentResult] = None,
    previous: Optional[AssessmentResult] = None,
    *,
    title: str = "Executive Function Profile",
    size: tuple[int, int] = (540, 440),
    dpi: int = 100,
) -> Image.Image:
    """Draw a BDEFS radar chart and return it as a PIL Image.

    Parameters
    ----------
    latest:
        The most recent result, drawn in blue.
    previous:
        The second-most-recent result, drawn in grey.
    title:
        Chart title.
    """
    latest_vals = _domain_values(latest) if latest else [0.0] * len(_DOMAIN_ORDER)

    n_axes = len(_DOMAIN_ORDER)
    angles = np.linspace(0, 2 * math.pi, n_axes, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]
    latest_closed = latest_vals + latest_vals[:1]

    fig_w, fig_h = size[0] / dpi, size[1] / dpi
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=_BG)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(_BG)

    # Grid & ticks
    ax.set_ylim(0, _DOMAIN_MAX)
    ax.set_yticks(range(0, _DOMAIN_MAX + 1, 3))
    ax.set_yticklabels(
        [str(v) for v in range(0, _DOMAIN_MAX + 1, 3)], color=_FG, fontsize=7
    )
    ax.set_xticks(angles[:-1])
    short_labels = [
        d.replace(" & ", "\n& ").replace("Organisation", "Org.") for d in _DOMAIN_ORDER
    ]
    ax.set_xticklabels(short_labels, color=_FG, fontsize=8)
    ax.tick_params(axis="x", pad=12)
    ax.spines["polar"].set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.5)

    # Previous result (grey)
    if previous is not None:
        prev_vals = _domain_values(previous) + _domain_values(previous)[:1]
        ax.plot(
            angles,
            prev_vals,
            color=_GREY_LINE,
            linewidth=1.2,
            alpha=0.6,
            label="Previous",
        )
        ax.fill(angles, prev_vals, color=_GREY_FILL)

    # Latest result (blue)
    ax.plot(angles, latest_closed, color=_BLUE_LINE, linewidth=2, label="Latest")
    ax.fill(angles, latest_closed, color=_BLUE_FILL)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        fontsize=8,
        facecolor=_BG,
        edgecolor=_GRID,
        labelcolor=_FG,
    )
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

    ax.plot(
        dates,
        scores,
        color=_BLUE_LINE,
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor=_ACCENT,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax.fill_between(dates, scores, alpha=0.15, color=_ACCENT)

    # Line of best fit (linear regression)
    if len(dates) >= 2:
        import matplotlib.dates as mdates

        x_num = mdates.date2num(dates)
        coeffs = np.polyfit(x_num, scores, 1)
        trend_y = np.polyval(coeffs, x_num)
        ax.plot(
            dates,
            trend_y,
            color="#b5b5b5",
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
            label="Trend",
        )
        ax.legend(
            loc="upper right",
            fontsize=8,
            facecolor=_BG,
            edgecolor=_GRID,
            labelcolor=_FG,
        )

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


def bdefs_momentum_glow(
    latest: AssessmentResult,
    previous: Optional[AssessmentResult] = None,
    *,
    title: str = "Momentum Reserve by Domain",
    size: tuple[int, int] = (620, 360),
    dpi: int = 100,
) -> Image.Image:
    """Draw a motivating domain-by-domain BDEFS chart.

    The chart inverts BDEFS difficulty scores into a "momentum reserve" percentage,
    so taller, brighter markers indicate more available headroom in that domain.
    """
    latest_pct = _domain_percentages(latest)
    previous_pct = _domain_percentages(previous) if previous is not None else None
    x = np.arange(len(_DOMAIN_ORDER), dtype=float)

    fig_w, fig_h = size[0] / dpi, size[1] / dpi
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(_BG)

    gradient = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    ax.imshow(
        gradient,
        extent=(-0.7, len(_DOMAIN_ORDER) - 0.3, 0, 100),
        origin="lower",
        aspect="auto",
        cmap=matplotlib.colormaps["GnBu"],
        alpha=0.12,
        zorder=0,
    )

    rng = np.random.default_rng(17)
    star_x = rng.uniform(-0.45, len(_DOMAIN_ORDER) - 0.55, 32)
    star_y = rng.uniform(62, 102, 32)
    star_sizes = rng.uniform(6, 22, 32)
    ax.scatter(
        star_x, star_y, s=star_sizes, c="#f6f4d2", alpha=0.14, linewidths=0, zorder=0.5
    )

    for idx, value in enumerate(latest_pct):
        beam = FancyBboxPatch(
            (idx - 0.23, 0),
            0.46,
            value,
            boxstyle="round,pad=0.02,rounding_size=0.22",
            linewidth=0,
            facecolor="#5db6b0",
            alpha=0.2,
            zorder=1,
        )
        ax.add_patch(beam)
        ax.vlines(idx, 0, value, color="#8ee3ef", linewidth=2.2, alpha=0.55, zorder=2)

    if previous_pct is not None:
        ax.plot(
            x,
            previous_pct,
            color="#a0a0a0",
            linewidth=1.2,
            linestyle="--",
            alpha=0.65,
            zorder=3,
        )
        ax.scatter(
            x,
            previous_pct,
            s=96,
            facecolors=_BG,
            edgecolors="#c0c0c0",
            linewidths=1.4,
            alpha=0.9,
            zorder=4,
        )

    ax.plot(x, latest_pct, color="#f6d365", linewidth=2.3, alpha=0.95, zorder=5)
    ax.scatter(x, latest_pct, s=300, c="#f6d365", alpha=0.16, linewidths=0, zorder=5)
    ax.scatter(x, latest_pct, s=135, c="#ffd27d", alpha=0.95, linewidths=0, zorder=6)
    ax.scatter(x, latest_pct, s=34, c="#fff7dd", alpha=1.0, linewidths=0, zorder=7)

    for idx, value in enumerate(latest_pct):
        if previous_pct is None:
            label = f"{value:.0f}%"
            color = "#fff7dd"
        else:
            delta = value - previous_pct[idx]
            if delta > 0.9:
                label = f"+{delta:.0f}"
                color = "#9be28f"
            elif delta < -0.9:
                label = f"{delta:.0f}"
                color = "#ffb38a"
            else:
                label = "steady"
                color = "#d9dde3"
        ax.text(
            idx,
            min(value + 7.5, 98),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            fontweight="bold",
            zorder=8,
        )

    short_labels = [
        d.replace("Organisation & Problem-Solving", "Organisation")
        .replace("Self-Restraint", "Restraint")
        .replace("Self-Motivation", "Motivation")
        .replace("Emotion Regulation", "Emotion")
        for d in _DOMAIN_ORDER
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, color=_FG, fontsize=9)
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.6, len(_DOMAIN_ORDER) - 0.4)
    ax.set_yticks((15, 45, 75))
    ax.set_yticklabels(("Needs support", "Building", "Rolling"), color=_FG, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for y in (15, 45, 75):
        ax.axhline(y, color=_GRID, linewidth=0.8, alpha=0.45, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    latest_overall = (
        (latest.max_score - latest.score) / latest.max_score * 100
        if latest.max_score
        else 0.0
    )
    subtitle = f"Higher glow means more available headroom. Overall momentum reserve: {latest_overall:.0f}%"
    if previous is not None and previous.max_score:
        prev_overall = (previous.max_score - previous.score) / previous.max_score * 100
        delta = latest_overall - prev_overall
        if abs(delta) >= 1:
            direction = "up" if delta > 0 else "down"
            subtitle += f"  |  vs previous: {abs(delta):.0f}% {direction}"

    ax.set_title(title, color=_FG, fontsize=13, fontweight="bold", pad=16)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        color="#d9dde3",
        fontsize=8.5,
        ha="left",
        va="bottom",
    )

    return _fig_to_pil(fig, dpi=dpi)
