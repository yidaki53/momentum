# Momentum agent instructions

## Hard rule
- Never commit or push changes.

## Command execution
- Use Poetry-first Python commands (`poetry run ...`) for linting, typing, tests, scripts, and app entry points.

## Versioning and release-branch policy
- Use semantic versioning (`MAJOR.MINOR.PATCH`) for releases.
- Version bump type by change impact:
  1. Docs-only changes (no behavior change): no bump.
  2. Internal-only maintenance (tests/refactors/tooling, no user-visible behavior): no bump unless a release is explicitly requested.
  3. Backward-compatible bug fixes and small improvements: PATCH bump.
  4. Backward-compatible features or substantial user-visible improvements: MINOR bump.
  5. Breaking/incompatible changes: MAJOR bump.
- Release branch naming convention: `release/MAJOR.MINOR.PATCH` (example: `release/1.2.0`).
- In this repository, cut release branches from `master` (if a `develop` branch is introduced later, cut from `develop`).
- Release branches are short-lived stabilization branches; allow only QA fixes, release docs/changelog updates, and version metadata changes.
- Do not perform new feature development directly on release branches.
- Tag every shipped release as `vMAJOR.MINOR.PATCH`.
- For hotfixes on older production versions, branch from the relevant release tag, apply the patch, retag, and merge/cherry-pick the fix into `master` and any active newer release branches.
- After release, merge the release branch back into `master` (and into `develop` as well if `develop` exists).
- When a version bump is required, update all version touchpoints together:
  - `pyproject.toml`
  - `momentum/__init__.py`
  - `mobile/buildozer.spec` (`version`)
  - Any CLI/desktop/mobile visible version/about strings
  - Use feature-workflow prompt when updating user-visible behavior to identify all impacted surfaces and ensure consistent updates.

## Android safe-upgrade requirements
- Android releases must always be installable as an upgrade over the prior release (never as a downgrade).
- In `mobile/buildozer.spec`, keep `version` aligned with project semver and maintain a strictly increasing Android version code (`android.numeric_version`) for each shipped build.
- Never change `package.name` or `package.domain` for an existing app lineage.
- For persistence/model changes, preserve backward compatibility or add explicit migration handling so existing local data/settings remain readable after upgrade.
- Before declaring a release ready, validate an upgrade path from the previous APK build and confirm startup, settings, timer flow, and history/results screens still work.
- Treat `package could not be installed / package conflicts with another package` as a likely signing-certificate mismatch first, then verify package identity/version code.

## Android CI signing and artifact rules
- CI-distributed Android artifacts must be built with `buildozer android release` (not debug) to avoid signer drift between builds.
- Use one stable release keystore across all shipped Android builds for this app lineage.
- Required repository secrets for Android release signing:
  - `ANDROID_RELEASE_KEYSTORE_B64`
  - `ANDROID_RELEASE_KEYSTORE_PASSWD`
  - `ANDROID_RELEASE_KEYALIAS`
  - `ANDROID_RELEASE_KEYALIAS_PASSWD`
- Android build jobs should fail fast with a clear error when signing secrets are missing.
- Keep CI Android version codes monotonic (e.g., deterministic formula using run number) and never decrease them.

## Architecture constraints
- `momentum/models.py` is the source of truth for domain models and shared data structures.
- Keep persistence/query logic in `momentum/db.py`, using typed inputs/outputs and explicit model mappings.
- Keep timer and cycle behavior centralized in `momentum/timer.py`.
- Keep assessment scoring/interpretation in `momentum/assessments.py`.
- Keep chart generation and chart-data shaping in `momentum/charts.py`.
- Avoid re-implementing business logic in UI layers; call shared modules instead.

## Cross-surface parity
- User-facing features should maintain parity across:
  - CLI (`momentum/cli.py`)
  - Desktop GUI (`momentum/gui.py`)
  - Mobile UI (`mobile/main.py`)
- Keep naming, messaging, defaults, and result interpretation consistent unless a deliberate surface-specific deviation is documented.
- Use cross-surface-parity prompt when updating user-visible behavior to identify all impacted surfaces and ensure consistent updates.

## UI callback hardening
- UI callbacks must fail safely and surface actionable errors instead of crashing the app.
- Guard async/scheduled callbacks and event handlers so exceptions are captured and user-visible.
- Keep UI state transitions resilient when callbacks fail.

## Testing and validation gates
- Add or update tests for any changed behavior.
- Run and pass:
  - `poetry run ruff check momentum/ tests/ mobile/main.py`
  - `poetry run mypy momentum/`
  - `poetry run pytest tests/ -v`

## Docs and assets sync
- Update `README.md` and `SCIENCE.md` for user-visible behavior, rationale, or workflow changes.
- Keep root/mobile documentation and related assets consistent when behavior is shared across surfaces.

## GitHub RFC guidance
- For non-trivial product, architecture, data-model, or cross-surface behavior changes, raise an RFC before implementation.
- RFCs should include: problem statement, goals/non-goals, proposed approach, alternatives considered, risks, rollout/migration plan, and acceptance criteria.
- Link implementation issues/PRs back to the RFC, and keep RFC status explicit (proposed, accepted, superseded, rejected).
- Record accepted decisions in long-lived docs when relevant (`README.md`, `SCIENCE.md`, or other project docs).

## GitHub issues: raising and resolving
- Raise an issue for each bug, feature, or maintenance task before or alongside implementation.
- New issues should include clear context: expected vs actual behavior, reproduction steps (for bugs), scope, and definition of done.
- Keep issue state current: triage, priority, blockers, and owner should be visible in the issue thread.
- Link related RFCs, follow-up issues, and implementation PRs so planning and delivery are traceable.
- Resolve issues only after acceptance criteria are met, tests/validation pass, and documentation updates are completed when required.
- When closing an issue, include a concise resolution summary and note any deferred follow-up work.

## GitHub auth and workflow-permission rules
- For this repository, use the `yidaki53` GitHub account for `gh` and git-authenticated GitHub operations.
- If a push updates `.github/workflows/*`, ensure the active token has `workflow` scope before pushing.
- If push is rejected with OAuth workflow-scope errors, refresh token scope before retrying.
