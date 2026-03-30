# Architecture And Performance Learnings

## Domain layout (current)
- Assessments are split into deep modules under `momentum/domain/assessments/`:
  - `scoring.py`
  - `interpretation.py`
  - `profile.py`
  - `__init__.py` (public aggregation)
- `momentum/assessments.py` is a backward-compatibility shim re-exporting domain APIs.
- Theme palette constants live in `momentum/ui/palette.py` and should be reused by chart/UI layers.

## Cython learnings (current)
- Cython-accelerated modules are integrated with safe Python fallbacks:
  - `momentum/_assessments_cy`
  - `momentum/_charts_cy`
  - `momentum/_timer_cy`
- Keep import-guard fallbacks (`try/except ImportError`) so pure-Python remains functional.
- Prefer optimizing hot loops and numeric transformations while keeping Python-facing APIs unchanged.

## Refactor guidance
- Avoid recreating monolithic modules; prefer focused files by responsibility.
- Do not duplicate business logic in CLI/GUI/mobile layers; call shared domain modules.
- Keep chart rendering logic separate from palette/theme definitions.
