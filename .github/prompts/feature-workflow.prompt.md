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
6. Versioning and release branching
   - Determine bump type using SemVer:
     - Docs-only (no behavior change): no bump.
     - Internal-only maintenance (tests/refactors/tooling, no user-visible behavior): no bump unless release is explicitly requested.
     - Backward-compatible bug fixes/small improvements: PATCH.
     - Backward-compatible feature additions/substantial improvements: MINOR.
     - Breaking/incompatible changes: MAJOR.
   - For version bumps, update all version touchpoints together:
     - `pyproject.toml`
     - `momentum/__init__.py`
     - `mobile/buildozer.spec`
     - About/version strings in CLI and UIs
   - For release prep, use branch naming `release/MAJOR.MINOR.PATCH`.
   - In this repository, cut release branches from `master` (or `develop` if introduced later).
   - Keep release branches short-lived and stabilization-only (QA fixes, release docs/changelog, version metadata); no new features on release branches.
   - Tag shipped releases as `vMAJOR.MINOR.PATCH`.
   - For hotfixes on older versions, branch from the relevant release tag, patch, retag, and merge/cherry-pick fixes into `master` and active newer release branches.
   - After release, merge the release branch back into `master` (and `develop` if it exists).
   - For Android releases, keep package identity stable and ensure version code progression (`android.numeric_version`) is monotonic.
7. Android release safety (when APK delivery is affected)
   - Build distributable APKs via `buildozer android release` (not debug).
   - Ensure stable release signing is configured via repository secrets:
     - `ANDROID_RELEASE_KEYSTORE_B64`
     - `ANDROID_RELEASE_KEYSTORE_PASSWD`
     - `ANDROID_RELEASE_KEYALIAS`
     - `ANDROID_RELEASE_KEYALIAS_PASSWD`
   - Ensure CI fails fast if any signing secret is missing.
   - Ensure release artifact selection prefers release APK outputs.
8. Validation before handoff
   - `poetry run ruff check momentum/ tests/ mobile/main.py`
   - `poetry run mypy momentum/`
   - `poetry run pytest tests/ -v`

Output checklist for handoff:
- Files changed (by layer)
- Behavior changes
- Test coverage added/updated
- Validation command results
- Version/doc updates (if applicable)
- Android upgrade safety checks/results (if APK path affected)
