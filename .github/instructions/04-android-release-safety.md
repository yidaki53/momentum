# Android Release Safety

- Android upgrades must be installable over prior release builds.
- Keep `mobile/buildozer.spec` package identity stable:
  - `package.name`
  - `package.domain`
- Keep app semver aligned and Android version code monotonic:
  - `version`
  - `android.numeric_version`
- For CI release artifacts, use `buildozer android release` (not debug).
- Keep one stable release keystore lineage for shipped builds.
- If install conflict occurs, verify signing certificate lineage before other debugging.

## Signing and CI caveats
- Do not trust `keytool -list` alone for release-key validation; it does not prove key password correctness.
- CI signing preflight must include:
  - keystore decode succeeds,
  - alias exists,
  - private key entry can be decrypted with provided key password.
- For keystore secrets, prefer file-based upload paths over shell-piped transforms to avoid silent base64 corruption.
- Keep decode scripts simple and fail-fast with clear stderr messages; avoid fragile heredoc indentation inside workflow `run` blocks.

## Maintainer signing and secrets runbook

### Canonical GitHub secrets (preferred names)
- `ANDROID_UPLOAD_KEYSTORE_B64`
- `ANDROID_UPLOAD_KEYSTORE_PASSWD`
- `ANDROID_UPLOAD_KEYALIAS`
- `ANDROID_UPLOAD_KEYALIAS_PASSWD`
- `GOOGLE_PLAY_SERVICE_ACCOUNT_JSON` (Play publish lane only)

Legacy fallback names remain accepted in CI (`ANDROID_RELEASE_*`) to avoid breaking older setups, but maintainers should migrate to the upload-key naming.

### One-time setup
1. Export the release/upload keystore to base64 without line wrapping.
2. Add the canonical secrets to repository or environment-scoped secrets.
3. Restrict Play publishing secrets to the protected workflow environment used by the publish job.
4. Run the Play publish workflow once with `dry_run=true` to validate signing preflight without uploading.

### Rotation or break-glass recovery
1. Rotate the keystore only if Play App Signing lineage allows it and recovery material is confirmed.
2. Update all four upload-key secrets in one change window.
3. Re-run workflow with `dry_run=true` and confirm keystore decode, alias check, and key decrypt check pass.
4. Run one internal-track publish (`dry_run=false`, `release_status=draft`) before any wider rollout.

### Fast triage map
- `not valid base64 after normalization`: malformed keystore secret value.
- `cannot open keystore`: wrong store password.
- `alias not found`: wrong alias or wrong keystore file.
- `cannot decrypt key entry`: wrong key entry password (common root cause of Gradle "final block not properly padded").

## Runtime caveats after build success
- "APK built" does not imply runtime safety; validate app startup and first-screen render on device/emulator.
- Avoid heavy eager imports on Android startup paths; use lazy imports for optional charting/visual modules.
- Treat settings callbacks as crash boundaries: wrap UI callbacks and surface recoverable errors in-app instead of crashing to black screen.
