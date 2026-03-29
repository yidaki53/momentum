Use this prompt when implementing a new feature or non-trivial enhancement in Momentum.

Goal:
- Deliver a complete feature with consistent architecture, tests, docs, and version updates.

Workflow:
1. Scope and impacted layers
   - Identify whether the change touches models, db, assessments/charts, timer service, CLI, desktop GUI, and mobile UI.
   - Call out anything intentionally out of scope.
2. Core domain first
   - Add/adjust types in `momentum/models.py`.
   - Add/adjust persistence in `momentum/db.py` with typed model inputs/outputs.
   - Add/adjust logic in `momentum/assessments.py`, `momentum/charts.py`, or `momentum/timer.py` as needed.
3. Surface integration
   - Wire feature into `momentum/cli.py`.
   - Wire corresponding UI behavior into `momentum/gui.py` and `mobile/main.py` unless explicitly scoped otherwise.
   - Ensure callback failures are surfaced safely in UI layers.
4. Tests
   - Add or update tests in `tests/` for all changed behavior.
   - Include CLI behavior coverage with `CliRunner` and temporary DB patching.
5. Documentation and assets
   - Update `README.md` and `SCIENCE.md` when user-visible behavior or rationale changes.
   - Keep root/mobile markdown copies in sync when both are edited.
6. Versioning
   - For user-facing changes, update version in:
     - `pyproject.toml`
     - `momentum/__init__.py`
     - `mobile/buildozer.spec`
     - About/version strings in CLI and UIs
7. Validation before handoff
   - `poetry run ruff check momentum/ tests/ mobile/main.py`
   - `poetry run mypy momentum/`
   - `poetry run pytest tests/ -v`

Output checklist for handoff:
- Files changed (by layer)
- Behavior changes
- Test coverage added/updated
- Validation command results
- Version/doc updates (if applicable)
