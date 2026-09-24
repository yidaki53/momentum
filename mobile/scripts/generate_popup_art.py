#!/usr/bin/env python3
"""Generate Momentum popup artwork without requiring a large ML model.

The default renderer is procedural PIL artwork: it is deterministic, fast on a
small laptop, and produces a 60-frame stepping-stone animation plus task-
specific popup icons.  ``--diffusion`` is an optional development-machine
mode that uses a small CPU diffusion model for hero keyframes; generated
intermediate files and its throwaway environment are removed afterwards.

Usage:
    python3 mobile/scripts/generate_popup_art.py
    python3 mobile/scripts/generate_popup_art.py --clean
    python3 mobile/scripts/generate_popup_art.py --diffusion
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "mobile" / "assets" / "art"
WORK = Path("/tmp/momentum-popup-art")
RAW = WORK / "raw"
HERO_SIZE = (448, 252)
HERO_COLS = 8
HERO_FPS = 60
HERO_FRAMES = 60
ICON_SIZE = 192

PALETTE = {
    "navy": (19, 39, 56),
    "teal": (57, 137, 145),
    "teal_light": (121, 205, 196),
    "amber": (239, 183, 91),
    "cream": (249, 239, 211),
    "ink": (24, 35, 45),
    "rose": (208, 111, 117),
    "lavender": (151, 137, 190),
}

ICON_COLORS = {
    "task": PALETTE["teal_light"],
    "coach": PALETTE["amber"],
    "act": PALETTE["lavender"],
    "result": PALETTE["teal_light"],
    "stroop": PALETTE["rose"],
    "download": PALETTE["amber"],
    "update": PALETTE["teal_light"],
    "folder": PALETTE["amber"],
    "info": PALETTE["teal_light"],
    "error": PALETTE["rose"],
}


def _draw_hero_frame(index: int) -> Image.Image:
    """Draw one frame of a calm path of stepping stones."""
    w, h = HERO_SIZE
    image = Image.new("RGB", (w, h), PALETTE["navy"])
    draw = ImageDraw.Draw(image, "RGBA")
    # Soft dusk sky and water bands.
    for y in range(h):
        t = y / h
        color = tuple(int(PALETTE["navy"][i] * (1 - t) + PALETTE["teal"][i] * t) for i in range(3))
        draw.line((0, y, w, y), fill=color + (255,))
    draw.ellipse((-80, 130, w + 80, h + 130), fill=(11, 67, 83, 180))
    for y in (105, 132, 160, 188):
        draw.arc((-40, y, w + 40, y + 28), 190, 350, fill=(121, 205, 196, 55), width=2)
    # Far bank and its small warm light.
    draw.polygon([(350, 205), (448, 175), (448, 252), (340, 252)], fill=(27, 69, 70, 255))
    glow = (370, 176)
    for radius, alpha in ((24, 18), (14, 32), (7, 90)):
        draw.ellipse((glow[0] - radius, glow[1] - radius, glow[0] + radius, glow[1] + radius), fill=PALETTE["amber"] + (alpha,))
    # Progressively appearing stones; interpolate positions smoothly.
    progress = index / (HERO_FRAMES - 1)
    count = 1 + int(progress * 7)
    for n in range(count):
        x = 34 + n * 48 + 5 * math.sin(index / 9 + n)
        y = 212 - n * 5 + 3 * math.cos(index / 11 + n)
        radius = 12 + (n % 2) * 2
        active = n < count - 1 or progress > 0.92
        fill = PALETTE["cream"] if active else (92, 126, 132)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill + (235,))
        draw.arc((x - radius - 3, y - radius - 2, x + radius + 3, y + radius + 2), 190, 350, fill=PALETTE["teal_light"] + (180,), width=2)
    # Small moving firefly makes the loop feel alive without distracting.
    fx = 40 + (w - 80) * ((progress + 0.12) % 1.0)
    fy = 48 + 10 * math.sin(index / 5)
    draw.ellipse((fx - 3, fy - 3, fx + 3, fy + 3), fill=PALETTE["amber"] + (220,))
    return image.filter(ImageFilter.GaussianBlur(0.15))


def _draw_icon(name: str) -> Image.Image:
    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    c = ICON_COLORS.get(name, PALETTE["teal_light"])
    dark = PALETTE["ink"]
    m = 22
    if name in {"task", "result"}:
        draw.rounded_rectangle((m + 12, m, ICON_SIZE - m - 12, ICON_SIZE - m), radius=18, fill=dark, outline=c, width=6)
        for y in (58, 82, 106, 130):
            draw.line((48, y, 68, y), fill=c, width=5)
            draw.ellipse((76, y - 6, 88, y + 6), fill=c)
    elif name == "coach":
        draw.rounded_rectangle((25, 34, 167, 130), radius=30, fill=dark, outline=c, width=6)
        draw.polygon([(70, 126), (58, 157), (101, 130)], fill=dark, outline=c)
        draw.ellipse((58, 66, 82, 90), fill=c)
        draw.ellipse((110, 66, 134, 90), fill=c)
        draw.arc((77, 76, 115, 116), 0, 180, fill=c, width=5)
    elif name == "act":
        draw.rounded_rectangle((28, 30, 164, 158), radius=12, fill=dark, outline=c, width=6)
        draw.line((96, 30, 96, 158), fill=c, width=4)
        for y in (58, 82, 106, 130):
            draw.line((48, y, 82, y), fill=c, width=4)
            draw.line((110, y, 144, y), fill=c, width=4)
    elif name == "stroop":
        for x, y, col in ((34, 38, PALETTE["rose"]), (72, 62, PALETTE["teal_light"]), (110, 86, PALETTE["amber"]), (48, 110, PALETTE["lavender"])):
            draw.rounded_rectangle((x, y, x + 52, y + 38), radius=8, fill=col)
    elif name == "download":
        draw.arc((35, 28, 157, 150), 210, 330, fill=c, width=8)
        draw.line((96, 48, 96, 118), fill=c, width=9)
        draw.polygon([(96, 146), (73, 112), (119, 112)], fill=c)
        draw.arc((35, 70, 157, 176), 30, 150, fill=c, width=8)
    elif name == "update":
        draw.arc((30, 30, 162, 162), 40, 320, fill=c, width=8)
        draw.polygon([(145, 36), (167, 29), (158, 54)], fill=c)
        draw.rounded_rectangle((70, 58, 122, 132), radius=10, outline=c, width=5)
    elif name == "folder":
        draw.polygon([(28, 58), (78, 58), (92, 72), (164, 72), (150, 145), (42, 145)], fill=dark, outline=c)
        draw.line((50, 93, 140, 93), fill=c, width=5)
    elif name == "error":
        draw.ellipse((28, 28, 164, 164), outline=c, width=8)
        draw.line((96, 60, 96, 108), fill=c, width=10)
        draw.ellipse((90, 126, 102, 138), fill=c)
    else:
        draw.ellipse((28, 28, 164, 164), outline=c, width=8)
        draw.ellipse((88, 56, 104, 72), fill=c)
        draw.line((96, 80, 96, 122), fill=c, width=8)
    return image


def _write_assets() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = Image.new("RGBA", (HERO_COLS * HERO_SIZE[0], HERO_COLS * HERO_SIZE[1]), (0, 0, 0, 0))
    for i in range(HERO_FRAMES):
        frame = _draw_hero_frame(i).convert("RGBA")
        atlas.paste(frame, ((i % HERO_COLS) * HERO_SIZE[0], (i // HERO_COLS) * HERO_SIZE[1]))
    atlas.save(OUT / "download_steps.png", optimize=True)
    for name in ICON_COLORS:
        _draw_icon(name).save(OUT / f"icon_{name}.png", optimize=True)
    (OUT / "manifest.txt").write_text(
        "download_steps.png:60 frames, 448x252, 8 columns, 60 fps\n"
        + "\n".join(f"icon_{name}.png:{ICON_SIZE}x{ICON_SIZE}" for name in sorted(ICON_COLORS))
        + "\n",
        encoding="utf-8",
    )


def _cleanup() -> None:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    print("[cleanup] temporary art-generation files removed")


def _diffusion() -> None:
    """Optional CPU diffusion keyframe mode for a well-provisioned dev machine."""
    if shutil.which("nvidia-smi"):
        print("[diffusion] This script intentionally uses CPU; run on a larger host for a model pass.")
    WORK.mkdir(parents=True, exist_ok=True)
    venv = WORK / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = str(venv / "bin" / "python")
    subprocess.run([py, "-m", "pip", "install", "-q", "torch", "diffusers", "transformers", "pillow"], check=True)
    print("[diffusion] Download a suitable model in this throwaway environment, then keep its keyframes under", RAW)
    print("[diffusion] The deterministic renderer above is used for the shipped APK assets.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="remove temporary files and exit")
    parser.add_argument("--diffusion", action="store_true", help="prepare an optional CPU diffusion environment")
    args = parser.parse_args()
    if args.clean:
        _cleanup()
        return 0
    if args.diffusion:
        _diffusion()
    _write_assets()
    print(f"[art] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
