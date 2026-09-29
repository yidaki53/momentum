"""Regression tests for the Cython-compiled scoring fast path.

``momentum/assessments_cy.pyx`` used to import ``momentum.assessments`` at
module level. That module is a shim over ``momentum.domain.assessments``,
whose ``__init__`` imports ``momentum.domain.assessments.scoring`` -- the very
module that imports the extension. The import therefore re-entered a
half-built module and raised ImportError for *every* import order.
``scoring.py`` catches ImportError to fall back to pure Python, so the failure
was silent: CI built the extension and the shipped code never used it.

These tests pin the fixed behaviour: the extension imports cleanly from any
entry point, the fast path is actually selected, and the compiled results
match the pure Python reference. They skip when the extension has not been
built. ``ui/charts.py`` used to carry the same dual path; it now renders with
PIL directly, so there is no compiled chart path to pin.
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

_RESULT_PROBE = """
import sys

if {block}:
    # A ``None`` entry makes ``from x import y`` raise ImportError, which is
    # what scoring.py catches to select the pure Python fallback.
    sys.modules["momentum._assessments_cy"] = None

import momentum.domain.assessments.scoring as scoring

bdefs = {{domain: [3] * len(qs) for domain, qs in scoring.BDEFS_QUESTIONS.items()}}
bisbas = {{domain: [2] * len(qs) for domain, qs in scoring.BISBAS_QUESTIONS.items()}}

for label, result in (
    ("bdefs", scoring.score_bdefs(bdefs)),
    ("bisbas", scoring.score_bisbas(bisbas)),
):
    print(label, result.assessment_type.value, result.score, result.max_score,
          sorted(result.domain_scores.items()))
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


@_needs_compiled
@pytest.mark.parametrize("first", _SCORING_IMPORT_ORDERS)
def test_scoring_fast_path_selected_for_every_import_order(first: str) -> None:
    """The compiled scoring helpers are reachable from any entry point."""
    assert _run(_SCORING_ORDER_PROBE.format(first=first)).strip() == "1 1 1"


@_needs_compiled
def test_compiled_results_match_pure_python_fallback() -> None:
    """The compiled path must agree with the fallback it replaces."""
    pure = _run(_RESULT_PROBE.format(block=True))
    compiled = _run(_RESULT_PROBE.format(block=False))
    assert pure == compiled


@_needs_compiled
def test_fast_path_engages_on_the_real_desktop_entry_point() -> None:
    """Importing the GUI must leave the compiled scoring path active."""
    code = """
import momentum.gui
import momentum.domain.assessments.scoring as scoring
print(int(scoring._CYTHON_AVAILABLE))
"""
    assert _run(code).strip() == "1"
