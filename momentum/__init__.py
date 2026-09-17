"""Momentum: A gentle tool for executive dysfunction support."""

from __future__ import annotations

from momentum.build_info import BUILD_NUMBER

__version__ = "0.4.0"
if BUILD_NUMBER:
    __version__ += f"-build.{BUILD_NUMBER}"
