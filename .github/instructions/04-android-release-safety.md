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
