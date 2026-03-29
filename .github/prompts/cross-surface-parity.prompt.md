Use this prompt when a change affects user-facing behavior that should stay aligned across CLI, desktop GUI, and mobile app.

Parity checks:
1. Entry points exist on each surface
   - CLI command/flags in `momentum/cli.py`
   - Desktop action/menu path in `momentum/gui.py`
   - Mobile action/navigation path in `mobile/main.py`
2. Shared logic is reused
   - Pull from shared modules (`models.py`, `db.py`, `assessments.py`, `charts.py`, `timer.py`) instead of duplicating logic per UI.
3. Terminology and messaging are consistent
   - Keep labels, setting names, and explanatory text equivalent across surfaces.
   - Keep BIS/BAS reference-line disclaimer and ACT wording consistent if touched.
4. Error handling quality
   - Ensure each surface reports actionable errors without crashing.
5. Persistence and defaults
   - Confirm settings/state are persisted and read through shared config/db paths.
   - Confirm personalization defaults are applied consistently.
6. Results/history rendering
   - If a test/feature has result views, update all relevant result surfaces (immediate result + historical view).
7. Regression tests
   - Add/update tests for shared behavior and CLI surface.

Final verification:
- Run ruff, mypy, and pytest suite.
- Summarize any intentional parity gaps explicitly.
