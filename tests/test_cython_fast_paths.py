"""Regression tests for the Cython-compiled fast paths.

``momentum/assessments_cy.pyx`` and ``momentum/charts_cy.pyx`` used to import
``momentum.assessments`` at module level. That module is a shim over
``momentum.domain.assessments``, whose ``__init__`` imports
``momentum.domain.assessments.scoring`` -- the very module that imports the
extensions. The import therefore re-entered a half-built module and raised
ImportError for *every* import order. ``scoring.py`` and ``ui/charts.py`` catch
ImportError to fall back to pure Python, so the failure was silent: CI built the
extensions and the shipped code never used them.

These tests pin the fixed behaviour: the extensions import cleanly from any entry
point, the fast path is actually selected, and the compiled results match the
pure Python reference. They skip when the extensions have not been built.
"""

from __future__ import annotations

import importlib.machinery
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "momentum"


def _extension_built(name: str) -> bool:
    """True when a compiled ``name`` extension exists for this interpreter."""
    return any(
        (PACKAGE_DIR / f"{name}{suffix}").is_file()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


_needs_compiled = pytest.mark.skipif(
    not _extension_built("_assessments_cy"),
    reason="Cython extensions not built; run: python setup.py build_ext --inplace",
)

_SCORING_ORDER_PROBE = """
import {first}
import momentum.domain.assessments.scoring as scoring
print(
    int(scoring._CYTHON_AVAILABLE),
    int(scoring.score_bdefs_cy is not None),
    int(scoring.score_bisbas_cy is not None),
)
"""

_CHARTS_ORDER_PROBE = """
import {first}
import momentum.ui.charts as charts
print(int(charts._CYTHON_AVAILABLE), int(charts.domain_percentages_cy is not None))
"""

_RESULT_PROBE = """
import sys
import types

if {block}:
    # A ``None`` entry makes ``from x import y`` raise ImportError, which is
    # what scoring.py / ui/charts.py catch to select the pure Python fallback.
    sys.modules["momentum._assessments_cy"] = None
    sys.modules["momentum._charts_cy"] = None

import momentum.domain.assessments.scoring as scoring
import momentum.ui.charts as charts

bdefs = {{domain: [3] * len(qs) for domain, qs in scoring.BDEFS_QUESTIONS.items()}}
bisbas = {{domain: [2] * len(qs) for domain, qs in scoring.BISBAS_QUESTIONS.items()}}

for label, result in (
    ("bdefs", scoring.score_bdefs(bdefs)),
    ("bisbas", scoring.score_bisbas(bisbas)),
):
    print(label, result.assessment_type.value, result.score, result.max_score,
          sorted(result.domain_scores.items()))

fake = types.SimpleNamespace(
    domain_scores={{domain: 4 for domain in scoring.BDEFS_QUESTIONS}},
)
print("percentages", [round(value, 6) for value in charts._domain_percentages(fake)])
"""


def _run(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


# Every import order that the real entry points use. ``scoring`` first is what
# plain ``import momentum.gui``/``momentum.cli`` does; ``_assessments_cy`` first
# is what a direct ``from momentum._assessments_cy import ...`` does.
_SCORING_IMPORT_ORDERS = [
    "momentum.domain.assessments.scoring",
    "momentum.domain.assessments",
    "momentum.assessments",
    "momentum.models",
    "momentum._assessments_cy",
    "momentum",
]

_CHARTS_IMPORT_ORDERS = [
    "momentum.ui.charts",
    "momentum.charts",
    "momentum.domain.assessments.scoring",
    "momentum._charts_cy",
]


@_needs_compiled
@pytest.mark.parametrize("first", _SCORING_IMPORT_ORDERS)
def test_scoring_fast_path_selected_for_every_import_order(first: str) -> None:
    """The compiled scoring helpers are reachable from any entry point."""
    assert _run(_SCORING_ORDER_PROBE.format(first=first)).strip() == "1 1 1"


@_needs_compiled
@pytest.mark.parametrize("first", _CHARTS_IMPORT_ORDERS)
def test_charts_fast_path_selected_for_every_import_order(first: str) -> None:
    """The compiled chart helper is reachable from any entry point."""
    assert _run(_CHARTS_ORDER_PROBE.format(first=first)).strip() == "1 1"


@_needs_compiled
def test_compiled_results_match_pure_python_fallback() -> None:
    """The compiled path must agree with the fallback it replaces."""
    pure = _run(_RESULT_PROBE.format(block=True))
    compiled = _run(_RESULT_PROBE.format(block=False))
    assert pure == compiled


@_needs_compiled
def test_fast_path_engages_on_the_real_desktop_entry_point() -> None:
    """Importing the GUI must leave both compiled fast paths active."""
    code = """
import momentum.gui
import momentum.domain.assessments.scoring as scoring
import momentum.ui.charts as charts
print(int(scoring._CYTHON_AVAILABLE), int(charts._CYTHON_AVAILABLE))
"""
    assert _run(code).strip() == "1 1"
