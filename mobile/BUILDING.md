# Local Android builds

Run `poetry install --with dev`, install Android SDK/NDK and JDK 17+, then
use `make mobile-apk` (debug APK) or `make mobile-aab` (signed release AAB;
requires signing credentials). `python3 mobile/scripts/build_android.py --check`
validates the environment without compiling.

The launcher works without shell activation: it selects the project `.venv`,
sets `VIRTUAL_ENV` and `PATH`, removes `PYTHONHOME`, chooses a JDK 17+ (overriding
an incompatible Java 8 `JAVA_HOME`), and exposes the committed pure-Python
MarkupSafe wheel using a file URI (safe for paths containing spaces).
It runs from `mobile/` and restores the spec after an AAB build, including failures.
No shell startup files or system Java alternatives are changed.

If p4a's generated helper venv has a mixed/broken pip installation, stop all
builds first and remove **only** the generated `build/venv` beneath the active
Buildozer storage directory. Do not delete the project `.venv` or the SDK.
For the default CPU build this generated directory is
`/tmp/buildozer-build/android/platform/build-arm64-v8a/build/venv`.
The next build recreates it. The pip-upgrade fallback patch is not a repair
for an already-corrupt pip; do not keep retrying the same failed environment.

## Cython extensions in the APK

`momentum/_assessments_cy`, `momentum/_charts_cy`, and `momentum/_timer_cy` are
compiled into the app by `mobile/scripts/build_android_ext.py` from the
Cython-generated `.c` files committed alongside the `.pyx` sources.
python-for-android only cythonizes *recipe* sources, so an app's own `.pyx` files
are never built; without this step the APK quietly uses the pure-Python fallbacks
in `domain/assessments/scoring.py` and `ui/charts.py`.

CI runs the script between buildozer's two passes, once pass 1 has produced the
target CPython headers. To inspect what it would do in a local build:

```bash
python3 mobile/scripts/build_android_ext.py \
  --build-dir /tmp/buildozer-build/android/platform \
  --arch arm64-v8a --min-api 26 --dry-run
```

`--dry-run` prints the clang command lines and is the quickest way to confirm the
target headers, `EXT_SUFFIX`, and NDK driver were all discovered. The script
reports and skips instead of failing, so a build without the toolchain keeps
working on the pure-Python paths. `source.include_exts` in `buildozer.spec` lists
`so` so the compiled modules survive buildozer's source filter.

After building, run `python3 mobile/scripts/verify_coach_apk.py <apk>`.
This checks package contents, not Android imports or inference. A USB device
must appear as `device` in `adb devices -l` (enable USB debugging and authorize
the computer) before runtime validation. A debug-signed APK may not update a
release-signed installation; do not uninstall the app to work around this
without backing up data and getting explicit approval.

The local CPU debug APK was built and passed the package verifier. The Vulkan
variant and release signing still need their own build/runtime validation.
