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
