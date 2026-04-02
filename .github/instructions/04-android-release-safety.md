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

## Runtime caveats after build success
- "APK built" does not imply runtime safety; validate app startup and first-screen render on device/emulator.
- Avoid heavy eager imports on Android startup paths; use lazy imports for optional charting/visual modules.
- Treat settings callbacks as crash boundaries: wrap UI callbacks and surface recoverable errors in-app instead of crashing to black screen.
