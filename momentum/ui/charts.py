"""Lightweight PIL chart rendering for desktop and Android."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from momentum.domain.assessments import (
    BDEFS_QUESTIONS,
    BISBAS_QUESTIONS,
    bisbas_effective_domain_max_score,
    bisbas_normalized_domain_score,
)
from momentum.models import AssessmentResult, AssessmentType
from momentum.ui.palette import (
    CHART_ACCENT,
    CHART_BG,
    CHART_BLUE_LINE,
    CHART_FG,
    CHART_GRID,
)

_DOMAIN_ORDER = list(BDEFS_QUESTIONS.keys())
_DOMAIN_MAX = max(len(questions) for questions in BDEFS_QUESTIONS.values()) * 4
_BISBAS_ORDER = list(BISBAS_QUESTIONS.keys())
_BISBAS_MAX = max(bisbas_effective_domain_max_score(d) for d in _BISBAS_ORDER)


def _configure_matplotlib_runtime() -> None:
    """Keep the historical Android runtime hook compatible after PIL migration."""
    private_dir = Path(os.environ.get("ANDROID_PRIVATE", "."))
    runtime_dir = private_dir / ".matplotlib"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(private_dir)
    os.environ["MPLCONFIGDIR"] = str(runtime_dir)
    os.environ["XDG_CACHE_HOME"] = str(runtime_dir)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _image(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", size, CHART_BG)
    return image, ImageDraw.Draw(image)


def _domain_values(result: AssessmentResult | None) -> list[float]:
    if result is None:
        return [0.0] * len(_DOMAIN_ORDER)
    return [float(result.domain_scores.get(domain, 0)) for domain in _DOMAIN_ORDER]


def _domain_percentages(result: AssessmentResult) -> list[float]:
    values = []
    for domain in _DOMAIN_ORDER:
        maximum = len(BDEFS_QUESTIONS[domain]) * 4
        score = float(result.domain_scores.get(domain, 0))
        values.append(max(maximum - score, 0) / maximum * 100 if maximum else 0.0)
    return values


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int = 12,
    bold: bool = False,
) -> None:
    draw.text(xy, value, fill=CHART_FG, font=_font(size, bold))


def bdefs_radar(
    latest: Optional[AssessmentResult] = None,
    previous: Optional[AssessmentResult] = None,
    *,
    title: str = "Executive Function Profile",
    size: tuple[int, int] = (540, 440),
    dpi: int = 100,
) -> Image.Image:
    image, draw = _image(size)
    width, height = size
    center = (width // 2, height // 2 + 15)
    radius = min(width, height) * 0.31
    count = len(_DOMAIN_ORDER)

    def point(index: int, value: float) -> tuple[int, int]:
        angle = -math.pi / 2 + index * 2 * math.pi / count
        scale = value / max(_DOMAIN_MAX, 1)
        return (
            int(center[0] + math.cos(angle) * radius * scale),
            int(center[1] + math.sin(angle) * radius * scale),
        )

    for level in range(1, 4):
        ring = [point(i, _DOMAIN_MAX * level / 3) for i in range(count)]
        draw.line(ring + [ring[0]], fill=CHART_GRID, width=1)
    for index, domain in enumerate(_DOMAIN_ORDER):
        edge = point(index, _DOMAIN_MAX)
        draw.line([center, edge], fill=CHART_GRID, width=1)
        label = domain.replace(" & ", "\n& ").replace("Organisation", "Org.")
        box = draw.textbbox((0, 0), label, font=_font(11))
        angle = -math.pi / 2 + index * 2 * math.pi / count
        position = (
            int(center[0] + math.cos(angle) * (radius + 24) - (box[2] - box[0]) / 2),
            int(center[1] + math.sin(angle) * (radius + 24) - 8),
        )
        draw.multiline_text(
            position, label, fill=CHART_FG, font=_font(11), align="center"
        )

    for values, color, width_px in (
        (_domain_values(previous), "#777777", 2),
        (_domain_values(latest), CHART_BLUE_LINE, 3),
    ):
        if latest is None and color == CHART_BLUE_LINE:
            continue
        polygon = [point(i, value) for i, value in enumerate(values)]
        draw.polygon(polygon, fill=color + "33" if color.startswith("#") else color)
        draw.line(polygon + [polygon[0]], fill=color, width=width_px)

    _text(draw, (18, 16), title, 16, True)
    return image


def bdefs_timeseries(
    results: list[AssessmentResult],
    *,
    title: str = "Score Over Time",
    size: tuple[int, int] = (560, 240),
    dpi: int = 100,
) -> Optional[Image.Image]:
    records = sorted(
        (r for r in results if r.assessment_type == AssessmentType.BDEFS),
        key=lambda r: r.taken_at,
    )
    if len(records) < 2:
        return None
    image, draw = _image(size)
    left, top, right, bottom = 55, 42, size[0] - 20, size[1] - 38
    maximum = max(records[0].max_score, 1)
    draw.line((left, top, left, bottom), fill=CHART_GRID, width=1)
    draw.line((left, bottom, right, bottom), fill=CHART_GRID, width=1)
    for level in range(0, 5):
        y = bottom - (bottom - top) * level / 4
        draw.line((left, int(y), right, int(y)), fill=CHART_GRID, width=1)
        _text(draw, (5, int(y) - 8), str(int(maximum * level / 4)), 10)
    points = []
    for index, record in enumerate(records):
        x = left + (right - left) * index / (len(records) - 1)
        y = bottom - (bottom - top) * float(record.score) / maximum
        points.append((int(x), int(y)))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=CHART_ACCENT, outline="white")
    draw.line(points, fill=CHART_BLUE_LINE, width=3)
    _text(draw, (18, 12), title, 15, True)
    return image


def bdefs_momentum_glow(
    latest: AssessmentResult,
    previous: Optional[AssessmentResult] = None,
    *,
    title: str = "Momentum Reserve by Domain",
    size: tuple[int, int] = (620, 360),
    dpi: int = 100,
) -> Image.Image:
    image, draw = _image(size)
    width, height = size
    values = _domain_percentages(latest)
    previous_values = _domain_percentages(previous) if previous else None
    left, top, right, bottom = 35, 52, width - 20, height - 55
    slot = (right - left) / max(len(values), 1)
    for index, value in enumerate(values):
        x = left + slot * (index + 0.5)
        y = bottom - (bottom - top) * value / 100
        draw.line((x, bottom, x, y), fill="#8ee3ef", width=4)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#ffd27d")
        _text(draw, (int(x - 15), max(int(y - 28), top)), f"{value:.0f}%", 10, True)
        label = (
            _DOMAIN_ORDER[index]
            .replace("Organisation & Problem-Solving", "Organisation")
            .replace("Self-", "")
        )
        draw.text(
            (int(x - slot / 2 + 5), bottom + 8), label, fill=CHART_FG, font=_font(9)
        )
        if previous_values:
            py = bottom - (bottom - top) * previous_values[index] / 100
            draw.ellipse((x - 4, py - 4, x + 4, py + 4), outline="#aaaaaa", width=2)
    for level, label in ((15, "Needs support"), (45, "Building"), (75, "Rolling")):
        y = bottom - (bottom - top) * level / 100
        draw.line((left, y, right, y), fill=CHART_GRID, width=1)
        _text(draw, (left, int(y) - 18), label, 9)
    _text(draw, (18, 14), title, 16, True)
    return image


def bisbas_profile_bars(
    result: AssessmentResult,
    *,
    title: str = "BIS/BAS Motivational Profile",
    size: tuple[int, int] = (560, 260),
    dpi: int = 100,
) -> Image.Image:
    image, draw = _image(size)
    labels = ["BIS", "Drive", "Reward", "Fun"]
    scores = [
        float(
            bisbas_normalized_domain_score(domain, result.domain_scores.get(domain, 0))
        )
        for domain in _BISBAS_ORDER
    ]
    left, top, right, bottom = 85, 48, size[0] - 24, size[1] - 25
    for index, (label, score) in enumerate(zip(labels, scores)):
        y = top + index * (bottom - top) / 4
        draw.text((12, int(y - 9)), label, fill=CHART_FG, font=_font(11))
        bar_right = left + (right - left) * score / max(_BISBAS_MAX, 1)
        draw.rounded_rectangle(
            (left, y - 9, bar_right, y + 9),
            radius=5,
            fill=("#d98c8c", CHART_BLUE_LINE, "#86c59d", "#d7ba7d")[index],
        )
        _text(
            draw,
            (int(min(bar_right + 8, right - 28)), int(y - 9)),
            f"{int(score)}",
            10,
            True,
        )
    _text(draw, (18, 15), title, 15, True)
    return image


__all__ = [
    "bdefs_momentum_glow",
    "bdefs_radar",
    "bdefs_timeseries",
    "bisbas_profile_bars",
]
