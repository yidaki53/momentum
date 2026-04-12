# Release And Versioning

- Use semantic versioning (`MAJOR.MINOR.PATCH`).
- Bump policy:
  1. Docs-only changes: no bump.
  2. Internal-only maintenance (tests/refactors/tooling, no user-visible behavior): no bump unless explicitly requested.
  3. Backward-compatible bug fixes/small improvements: PATCH.
  4. Backward-compatible features/substantial user-visible improvements: MINOR.
  5. Breaking changes: MAJOR.

When version bump is required, update all touchpoints together:
- `pyproject.toml`
- `momentum/__init__.py`
- `mobile/momentum/__init__.py`
- `mobile/buildozer.spec` (`version`)
- Any CLI/desktop/mobile visible version/about strings

Release branch policy:
- Branch naming: `release/MAJOR.MINOR.PATCH`
- Cut from `master` (or `develop` if introduced later)
- Release branch is stabilization-only
- Tag shipped releases as `vMAJOR.MINOR.PATCH`

## Paid launch checklist (Play rollout/rollback gates)

### Pre-rollout gates
1. Semver/version touchpoints are aligned (`pyproject.toml`, desktop/mobile `__init__`, and `mobile/buildozer.spec`).
2. CI is green for tests plus Android artifact build.
3. Internal-track upload succeeds from the dedicated Play publish workflow.
4. Smoke checks pass on at least one clean-install and one upgrade-install Android device.
5. Release notes, privacy copy, and support contact path are ready.

### Staged rollout gates
1. Start with internal track verification, then production staged rollout at a low fraction.
2. Promote only after crash-free session trend, startup success, and core home/settings/timer flows remain healthy.
3. Hold rollout if any severe regression appears in onboarding, timer continuity, settings persistence, or assessment rendering.

### Rollback gates
1. Immediate halt criteria: startup crash spike, data-loss signal, timer/session corruption, or signing/install failures.
2. If halt criteria trigger, stop rollout in Play Console and keep current release status non-completed.
3. Ship a patched build with incremented semver and monotonic `android.numeric_version` before resuming rollout.
4. Record incident, trigger condition, and mitigation in release notes/history for future launches.
